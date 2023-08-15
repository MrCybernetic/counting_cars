import time
from threading import Thread
import cv2


class VideoStreamWidget(object):
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.frame = None
        if self.capture.isOpened():
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            (self.status, self.frame) = self.capture.read()
            time.sleep(1/30)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def get_frame(self):
        # Display frames in main program
        return self.frame
