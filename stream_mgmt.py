import cv2
from multiprocessing import Process, Queue
import time


class VideoStreamWidget(object):
    def __init__(self, src, buffer_size=10):
        self.frame_queue = Queue(maxsize=buffer_size)
        self.process = Process(target=self.update, args=(src, self.frame_queue))
        self.process.daemon = True
        self.process.start()

    def update(self, src, frame_queue):
        capture = cv2.VideoCapture(src)
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            (status, frame) = capture.read()
            if not status:
                break
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)
            time.sleep(1/fps)
        capture.release()

    def get_frame(self):
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()
