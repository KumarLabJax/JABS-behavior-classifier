from threading import Thread
from queue import Queue
import cv2

_FRAME_LABEL_COLOR = (0, 255, 102)


class VideoStream:
    """
    video stream
    """
    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        if not self.stream.isOpened():
            raise Exception(f"unable to open {path}")

        self.frame_index = 0
        self.queue_size = queue_size
        self.frame_queue = Queue(maxsize=self.queue_size)

        self._num_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.stream.get(cv2.CAP_PROP_FPS)
        self._duration = 1 / self._fps

        # used to signal the reading thread to terminate
        self._stopped = False

        self.thread = None

        # load the first frame:
        self.load_next_frame()

        self._width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        self.stream.release()

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def stopped(self):
        return self._stopped

    @property
    def dimensions(self):
        return self._width, self._height

    def seek(self, frame):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self.frame_index = frame
        self.frame_queue = Queue(maxsize=self.queue_size)
        self.load_next_frame()

    def load_next_frame(self):
        (grabbed, frame) = self.stream.read()

        if grabbed:
            self.frame_queue.put({
                'data': frame,
                'index': self.frame_index,
                'duration': self._duration
            })
            self.frame_index += 1

    def start(self):
        """ start a thread to read frames from the file video stream """
        self._stopped = False
        self.thread = Thread(target=self._main, args=())
        self.thread.start()
        return self

    def stop(self):
        """stop the thread """
        self._stopped = True
        self.thread.join()
        self.thread = None

    def read(self):
        """ return next frame in the queue """
        frame = self.frame_queue.get()
        if frame['index'] != -1:
            self._add_frame_num(frame['data'], frame['index'])
        return frame

    def empty(self):
        """ is the frame queue empty """
        return self.frame_queue.qsize() == 0

    @staticmethod
    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    @staticmethod
    def _add_frame_num(frame, frame_num):
        """ add the frame number to bottom right of frame """
        label = f"{frame_num}"

        # figure out text origin
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_origin = (frame.shape[1] - label_size[0][0] - 5, frame.shape[0] - 5)

        cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, _FRAME_LABEL_COLOR, 1, cv2.LINE_AA)

    def _main(self):
        """ main loop for thread """
        while True:
            # do we need to stop?
            if self._stopped:
                return

            # otherwise if the queue is not full grab a frame insert it
            if not self.frame_queue.full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    # we reached the end of file
                    self.frame_queue.put({
                        'data': None,
                        'index': -1
                    })
                    return

                self.frame_queue.put({
                    'data': frame,
                    'index': self.frame_index,
                    'duration': self._duration
                })
                self.frame_index += 1
