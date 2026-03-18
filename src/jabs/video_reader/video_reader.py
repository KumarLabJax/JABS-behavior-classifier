"""Video file reading and frame extraction using OpenCV."""

import time
import weakref
from pathlib import Path
from typing import ClassVar

import cv2


class VideoReader:
    """VideoReader.

    Uses OpenCV to open a video file and read frames.
    """

    _EOF: ClassVar[dict[str, object]] = {"data": None, "index": -1}

    def __init__(self, path: Path) -> None:
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
        if self._fps <= 0:
            raise ValueError(f"invalid frame rate ({self._fps}) for video {path}")

        # calculate duration in seconds of each frame based on frame rate
        self._duration = 1.0 / self._fps

        # get frame dimensions
        self._width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._filename = path.name
        self._finalizer = weakref.finalize(self, self.stream.release)

    @property
    def num_frames(self) -> int:
        """Get total number of frames in the video.

        Returns:
            Total frame count.
        """
        return self._num_frames

    @property
    def fps(self) -> int:
        """Get frames per second from video.

        Returns:
            Frame rate rounded to the nearest integer.
        """
        return self._fps

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return width and height of video frames.

        Returns:
            Tuple of (width, height) in pixels.
        """
        return self._width, self._height

    @property
    def filename(self) -> str:
        """Return the name of the video file.

        Returns:
            Filename component of the video path.
        """
        return self._filename

    def close(self) -> None:
        """Release the video capture resources."""
        self._finalizer()  # no-op if already called; calls stream.release() otherwise
        self.stream = None

    def __enter__(self) -> "VideoReader":
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release resources on context manager exit."""
        self.close()

    def get_frame_time(self, frame_number: int) -> str:
        """Return a formatted string of the time of a given frame.

        Args:
            frame_number: Frame index to get the frame time for.

        Returns:
            Formatted time string (%H:%M:%S) of the given frame.
        """
        return time.strftime("%H:%M:%S", time.gmtime(frame_number * self._duration))

    def seek(self, index: int) -> None:
        """Seek to a specific frame.

        This will clear the buffer and insert the frame at the new position.

        Args:
            index: Frame index to seek to.

        Raises:
            OSError: If the video stream has been closed.

        Note:
            some video formats might not be able to seek to an exact frame
            position so this could be slow in those cases. Our avi files have
            reasonable seek times.
        """
        if self.stream is None:
            raise OSError("video stream is closed")

        if self.stream.set(cv2.CAP_PROP_POS_FRAMES, index):
            self._frame_index = index

    def load_next_frame(self) -> dict[str, object]:
        """Grab the next frame from the file.

        Raises:
            OSError: If the video stream has been closed.
        """
        if self.stream is None:
            raise OSError("video stream is closed")

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

    @classmethod
    def get_nframes_from_file(cls, path: Path) -> int:
        """get the number of frames by inspecting the video file"""
        stream = cv2.VideoCapture(str(path))
        if not stream.isOpened():
            raise OSError(f"unable to open {path}")

        num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        stream.release()  # Always release the stream
        return num_frames
