import time
import typing
from pathlib import Path

import cv2


class VideoReader:
    """VideoReader.

    Uses OpenCV to open a video file and read frames.
    """

    _EOF: typing.ClassVar[dict] = {"data": None, "index": -1}

    def __init__(self, path: Path):
        """Initialize a VideoReader object.

        Args:
            path: path to video file
        """
        # open video file
        self.stream = cv2.VideoCapture(str(path))
        if not self.stream.isOpened():
            raise OSError(f"unable to open {path}")

        self._frame_index = 0
        self._num_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        # get frame rate
        self._fps = round(self.stream.get(cv2.CAP_PROP_FPS))

        # calculate duration in seconds of each frame based on frame rate
        self._duration = 1.0 / self._fps

        # get frame dimensions
        self._width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._filename = path.name

    @property
    def num_frames(self):
        """get total number of frames in the video"""
        return self._num_frames

    @property
    def fps(self):
        """get frames per second from video"""
        return self._fps

    @property
    def dimensions(self):
        """return width, height of video frames"""
        return self._width, self._height

    @property
    def filename(self):
        """return the name of the video file"""
        return self._filename

    def get_frame_time(self, frame_number):
        """return a formatted string of the time of a given frame"""
        return time.strftime("%H:%M:%S", time.gmtime(frame_number * self._duration))

    def seek(self, index):
        """Seek to a specific frame.

        This will clear the buffer and insert the frame at the new position.

        Note:
            some video formats might not be able to seek to an exact frame
            position so this could be slow in those cases. Our avi files have
            reasonable seek times.
        """
        if self.stream.set(cv2.CAP_PROP_POS_FRAMES, index):
            self._frame_index = index

    def load_next_frame(self) -> dict:
        """grab the next frame from the file"""
        (grabbed, frame) = self.stream.read()
        if grabbed:
            data = {
                "data": frame,
                "index": self._frame_index,
                "duration": self._duration,
            }
            self._frame_index += 1
        else:
            data = self._EOF
        return data

    @staticmethod
    def _resize_image(image, width=None, height=None, interpolation=None):
        """resize an image, allow passing only desired width or height to maintain current aspect ratio

        Args:
            image: image to resize
            width: new width, if None compute to maintain aspect ratio
            height: new height, if None compute to maintain aspect ratio
            interpolation: type of interpolation to use for resize. If
                None, we will default to cv2.INTER_AREA for shrinking cv2.INTER_CUBIC when
                expanding

        Returns:
            resized image
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
            inter = cv2.INTER_AREA if dim[0] * dim[1] < w * h else cv2.INTER_CUBIC
        else:
            inter = interpolation

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    @classmethod
    def get_nframes_from_file(cls, path: Path):
        """get the number of frames by inspecting the video file"""
        # open video file
        stream = cv2.VideoCapture(str(path))
        if not stream.isOpened():
            raise OSError(f"unable to open {path}")

        return int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
