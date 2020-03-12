from threading import Thread
from queue import Queue
import cv2
import time

_FRAME_LABEL_COLOR = (0, 255, 102)


class VideoStream:
    """
    VideoStream.

    Uses OpenCV to open a video file and read frames.

    Allows for reading one frame at a time, or starting a thread to buffer
    frames.
    """

    _EOF = {'data': None, 'index': -1}

    def __init__(self, path, frame_buffer_size=128):
        """

        :param path: path to video file
        :param frame_buffer_size: max number of frames to buffer
        """

        # open video file
        self.stream = cv2.VideoCapture(path)
        if not self.stream.isOpened():
            # TODO use a less general exception
            raise Exception(f"unable to open {path}")

        self._frame_index = 0
        self._queue_size = frame_buffer_size
        self._frame_queue = Queue(maxsize=frame_buffer_size)

        self._num_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.stream.get(cv2.CAP_PROP_FPS)
        self._duration = 1 / self._fps

        # used to signal the reading thread to terminate
        self._stopped = False

        # flag used to indicate we've read to the end of the file
        # used so that load_next_frame() won't add more than one EOF frame
        self._eof = False

        # buffering thread
        self._thread = None

        # load the first frame into the buffer so the player can show the
        # first frame:
        self.load_next_frame()

        # get frame dimensions
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

    def seek(self, index):
        """
        Seek to a specific frame.
        This will clear the buffer and insert the frame at the new position.

        NOTE:
            some video formats might not be able to set an exact frame position
            this is not a problem with our AVI files, but should be kept in
            mind if we eventually accommodate other types of files.
        """
        if self.stream.set(cv2.CAP_PROP_POS_FRAMES, index):
            self._frame_index = index
            self._eof = False

            # clear the buffer
            self._frame_queue = Queue(maxsize=self._queue_size)

            # load frame at current position into buffer
            self.load_next_frame()

    def load_next_frame(self):
        """ grab the next frame and add it to the queue """

        # queue is already full, don't do anything
        if self._frame_queue.full():
            return

        (grabbed, frame) = self.stream.read()
        if grabbed:
            self._frame_queue.put({
                'data': frame,
                'index': self._frame_index,
                'duration': self._duration
            })
            self._frame_index += 1
        elif not self._eof:
            # if the _EOF frame hasn't already been written to the buffer
            # do so now
            self._frame_queue.put(self._EOF)
            self._eof = True

    def start(self):
        """
        start a thread to read frames from the file video stream from the
        current position
        """
        self._stopped = False
        self._thread = Thread(target=self._stream, args=())
        self._thread.start()
        return self

    def stop(self):
        """stop the buffering thread """
        if self._thread:
            self._stopped = True
            self._thread.join()
            self._thread = None

    def read(self):
        """
        return next frame in the queue

        Note: this is blocking! Don't call this without starting the stream
        or explicitly calling load_next_frame().
        """
        frame = self._frame_queue.get()

        # add labels with frame number and time
        # TODO remove, currently for debugging
        if frame['index'] != -1:
            self._add_frame_num(frame['data'], frame['index'])
            self._add_time_overlay(frame['data'], frame['index'] * frame['duration'])

        return frame

    @staticmethod
    def _resize_image(image, width=None, height=None, interpolation=None):
        """
        resize an image, allow passing only desired width or height to
        maintain current aspect ratio

        :param image: image to resize
        :param width: new width, if None compute to maintain aspect ratio
        :param height: new height, if None compute to maintain aspect ratio
        :param interpolation: type of interpolation to use for resize. If None,
        we will default to cv2.INTER_AREA for shrinking cv2.INTER_CUBIC when
        expanding
        :return: resized image
        """
        # current size
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        elif height is None:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        else:
            dim = (width, height)

        if interpolation is None:
            if dim[0] * dim[1] < w * h:
                # new image size is larger than the original, default to
                # INTER_AREA
                inter = cv2.INTER_AREA
            else:
                inter = cv2.INTER_CUBIC
        else:
            inter = interpolation

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    @staticmethod
    def _add_frame_num(frame, frame_num):
        """
        add the frame number to bottom right of frame
        :param frame: frame image
        :param frame_num: frame index
        """
        label = f"{frame_num}"

        # figure out text origin
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_origin = (frame.shape[1] - label_size[0][0] - 5,
                       frame.shape[0] - 5)

        cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _FRAME_LABEL_COLOR, 1, cv2.LINE_AA)

    @staticmethod
    def _add_time_overlay(frame, frame_time):
        """
        add a time overlay in the format HH:MM:SS to the lower left of the frame
        :param frame: frame image
        :param frame_time: time in seconds to use for overlay
        :return: frame with text added
        """
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(frame_time))

        text_origin = (5, frame.shape[0] - 5)
        cv2.putText(frame, formatted_time, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _FRAME_LABEL_COLOR, 1, cv2.LINE_AA)


    def _stream(self):
        """
        main loop for thread

        reads frames in and buffers them in our frame queue
        """
        while not self._stopped:
            # if the queue is not full grab a frame insert it
            if not self._frame_queue.full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    # We've reached the end of file.
                    # Add EOF indicator to end of buffer
                    self._frame_queue.put(self._EOF)
                    self._eof = True
                else:
                    # we got a new frame, add it to the queue
                    self._frame_queue.put({
                        'data': frame,
                        'index': self._frame_index,
                        'duration': self._duration
                    })
                    self._frame_index += 1
