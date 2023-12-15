import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from threading import Lock, Thread
import cv2
import numpy as np
import time
import logging
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal


class CameraStream(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, src=0):
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        self.lock = Lock()  # Lock for thread safety
        self.logger = self.setup_logger()
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.status = None
        self.thread.start()

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

    def update(self):
        while True:
            if self.capture.isOpened():
                # Grab the frame
                if self.capture.grab():
                    # Retrieve the frame
                    self.status, self.img = self.capture.retrieve()
            time.sleep(0.01)  # Adjust this delay as needed

    def DarkChannel(self, im, sz):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)  # Use CPU min function

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)

        return dark

    # @jit
    def EstimateA(self, img, darkChannel):
        h, w, _ = img.shape
        length = h * w
        num = max(int(length * 0.0001), 1)
        # convert to a row vector
        darkChannVec = np.reshape(darkChannel, length)
        index = darkChannVec.argsort()[length - num:]
        rowIdx = index // w
        colIdx = index % w
        coords = np.stack((rowIdx, colIdx), axis=1)

        sumA = np.zeros((1, 3))
        for coord in coords:
            row, col = coord
            sumA += img[row, col, :]
        A = sumA / num
        return A

    def TransmissionEstimate(self, im, A, sz):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)

        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]

        transmission = 1 - omega * self.DarkChannel(im3, sz)
        return transmission

    def GaussianTransmissionRefine(self, et):
        r = 89  # radius of the Gaussian filter

        # Apply Gaussian filtering to the transmission map
        t = cv2.GaussianBlur(et, (r, r), 0)

        return t

    def Recover(self, im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)

        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

        return res

    def process_frame(self):
        size = 15
        I = self.img.astype('float64') / 255
        dark_channel = self.DarkChannel(I, size)
        A = self.EstimateA(I, dark_channel)
        TE = self.GaussianTransmissionRefine(
            self.TransmissionEstimate(I, A, size))
        self.frame = self.Recover(I, TE, A)

    def show_frame(self):
        frame_count = 0
        start_time = time.time()

        try:
            self.process_frame()

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            self.logger.debug(f"FPS: {fps}")

            # Display dehazed frame

            self.frame_processed.emit(self.frame)

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")


# Create a QMainWindow with a QLabel to display the frames
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.video_label)

        self.stream = CameraStream('http://192.168.1.5:4747/video?1280x720')
        self.stream.frame_processed.connect(self.display_frame)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Set the timeout interval in milliseconds

        self.screenshot_button = QPushButton("Take Screenshot", self)
        self.screenshot_button.clicked.connect(self.take_screenshot)
        layout.addWidget(self.screenshot_button)

    @pyqtSlot(np.ndarray)
    def display_frame(self, frame):
        # Convert the frame to a QImage
        scaled_image = (
            frame * 255.0).clip(0, 255).astype(np.uint8)
        # self.take_screenshot(scaled_image, "Dehazed")
        rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                        rgb_image.shape[1] * 3, QImage.Format_RGB888)
        # Convert the image to QPixmap
        pixmap = QPixmap.fromImage(qimage)

        # Scale the pixmap while keeping the aspect ratio
        pixmap = pixmap.scaled(self.video_label.width(),
                               self.video_label.height(), Qt.KeepAspectRatio)

        # Display the QImage in the QLabel
        self.video_label.setPixmap(pixmap)

    def take_screenshot(self):
        # Capture the current frames
        original_frame = self.stream.img
        processed_frame = self.stream.frame
        timestamp = time.time()
        # Save the original and processed frames as images
        cv2.imwrite(f"original_screenshot_{timestamp}.png", original_frame)
        cv2.imwrite(
            f"processed_screenshot_{timestamp}.png", processed_frame * 255)
        print("Screenshots saved successfully.")

    def update_frame(self):
        self.stream.show_frame()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
