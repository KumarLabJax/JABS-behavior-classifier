from threading import Thread
from queue import Queue
import cv2
import time


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

        # get framerate
        self._fps = self.stream.get(cv2.CAP_PROP_FPS)

        # calculate duration in seconds of each frame based on framerate
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
        """ get total number of frames in the video """
        return self._num_frames

    @property
    def stopped(self):
        """ return True if the stream is stopped, false otherwise """
        return self._stopped

    @property
    def dimensions(self):
        """ return width, height of video frames """
        return self._width, self._height

    def get_frame_time(self, frame_number):
        """ return a formatted string of the time of a given frame """
        return time.strftime('%H:%M:%S',
                             time.gmtime(frame_number * self._duration))

    def seek(self, index):
        """
        Seek to a specific frame.
        This will clear the buffer and insert the frame at the new position.

        NOTE:
            some video formats might not be able to seek to an exact frame
            position so this could be slow in those cases. Our avi files have
            reasonable seek times.
        """
        if self.stream.set(cv2.CAP_PROP_POS_FRAMES, index):
            self._frame_index = index
            self._eof = False

            # clear the buffer
            self._frame_queue = Queue(maxsize=self._queue_size)

            # load frame at current position into buffer
            self.load_next_frame()

    def load_next_frame(self):
        """ grab the next frame from the stream and add it to the buffer """

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
        current position and insert them into the buffer
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
        return self._frame_queue.get()

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
