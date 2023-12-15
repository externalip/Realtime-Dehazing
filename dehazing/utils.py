import cv2
from threading import Thread, Lock
import time
import numpy as np
from dehazing.dehazing import *
from PyQt5.QtCore import pyqtSignal, QThread, QObject
import logging
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque


class CameraStream(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, url) -> None:
        super(CameraStream, self).__init__()
        self.url = url
        self.status = None
        self.frame_count = 0
        self.start_time = time.time()
        self.logger = self.setup_logger()
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.thread_lock = Lock()
        self.init_video_capture()
        self.width = 640
        self.height = 480
        self.inter = cv2.INTER_AREA
        self.stop_thread = False  # Flag to signal the threads to stop
        self.threads_count = min(psutil.cpu_count(
            logical=False), psutil.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.threads_count)

    def init_video_capture(self):
        try:
            self.capture = cv2.VideoCapture(self.url)
            if not self.capture.isOpened():
                raise ValueError(
                    f"Error: Unable to open video capture from {self.url}")
        except Exception as e:
            print(f"Error initializing video capture: {e}")
            self.status = False

    def setup_logger(self):
        logger = logging.getLogger("CameraStreamLogger")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        # Log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def grab_frames(self):
        while not self.stop_thread:
            if self.capture.isOpened():
                self.capture.grab()
            else:
                self.stop_thread = True

    def update(self):
        grab_thread = Thread(target=self.grab_frames, args=())
        grab_thread.daemon = True
        grab_thread.start()
        while not self.stop_thread:
            try:
                if self.capture.isOpened():
                    self.status, frame = self.capture.retrieve()
                    if self.status:
                        self.img = cv2.resize(
                            frame, (self.width, self.height), self.inter)
                        future = self.executor.submit(
                            self.process_and_emit_frame, self.img)
                    else:
                        self.status = False  # Ensure status is False if the capture is not opened
            except Exception as e:
                print(f"Error processing frame: {e}")
                self.status = False  # Set status to False in case of error
        grab_thread.join()

    def process_and_emit_frame(self, frame):
        try:
            if not self.use_cuda:
                dehazing_instance = DehazingCPU()
                self.frame = dehazing_instance.image_processing(frame)
            else:
                dehazing_instance = DehazingCPU()
                self.frame = dehazing_instance.image_processing(frame)

            with self.thread_lock:
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time
                self.logger.debug(f"FPS: {fps}")
                self.frame_processed.emit(self.frame)
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    def start(self) -> None:
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        # time.sleep(1)
        with self.thread_lock:
            self.stop_thread = True  # Set the flag to stop the threads
            if self.thread is not None:
                self.thread.join()
            if self.capture is not None:
                self.capture.release()
            self.terminate()


class VideoProcessor(QObject):
    update_progress_signal = pyqtSignal(int)

    def __init__(self, input_file, output_file):
        super(VideoProcessor, self).__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.total_frames = 0
        self.frames_processed = 0
        self.status_lock = Lock()
        self.threads_count = min(psutil.cpu_count(
            logical=False), psutil.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.threads_count)

    @staticmethod
    def process_frame(frame):
        dehazing_instance = DehazingCPU()
        processed_frame = dehazing_instance.image_processing(frame)
        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=(255.0))
        return processed_frame

    def process_video(self):
        start_time = time.time()
        cap = cv2.VideoCapture(self.input_file)
        if not cap.isOpened():
            print('Error opening video file')
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'mp4v'),
                              original_fps, (frame_width, frame_height))

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with ProcessPoolExecutor(max_workers=self.threads_count) as executor:
            futures = deque()

            while True:
                with self.status_lock:
                    if self.frames_processed >= self.total_frames:
                        break

                ret, frame = cap.read()
                if not ret:
                    break

                future = executor.submit(self.process_frame, frame)
                future.add_done_callback(self.update_progress)
                futures.append(future)
                if len(futures) > original_fps:
                    processed_frame = futures.popleft().result()
                    out.write(processed_frame)
                    del processed_frame  # Explicitly delete the processed frame to free up memory
            while futures:
                processed_frame = futures.popleft().result()
                out.write(processed_frame)
                del processed_frame

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing took {time.time() - start_time} seconds")

    def start_processing(self):
        processing_thread = Thread(target=self.process_video)
        processing_thread.start()

    def update_progress(self, future):
        with self.status_lock:
            self.frames_processed += 1
            progress_percentage = int(
                (self.frames_processed / self.total_frames) * 100)
            print(
                f"Outputting frame {self.frames_processed} of {self.total_frames}")
            print(self.threads_count)
            self.update_progress_signal.emit(progress_percentage)
