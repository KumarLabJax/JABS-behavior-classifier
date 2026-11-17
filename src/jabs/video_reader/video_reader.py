"""Video file reading and frame extraction using OpenCV."""

import logging
import time
import weakref
from pathlib import Path
from typing import ClassVar

import cv2

from .utilities import get_fps_and_nframes

logger = logging.getLogger(__name__)

# How many times seek() lands again after overshooting the requested frame before it
# gives up and trusts OpenCV's position.
_MAX_SEEK_ATTEMPTS = 4

# How far, as a fraction, a frame duration measured from the file's timestamps may be
# from the one implied by its reported frame rate before the measurement is distrusted
# (for example because the frame count is an estimate). A difference of 1/(frames - 1)
# is allowed on top of this: that is the discrepancy that throws OpenCV's seeking off,
# and it is large for short videos.
#
# Known gap: a container whose frame count is one higher than the frames that decode
# skews the measured duration by the same 1/(frames - 1) and passes this check. Frames
# past the midpoint are then numbered one too high, so seeks there land a frame early
# without a warning, as they did before seek() corrected landings. The check cannot
# tell that file apart from the reported-rate error it corrects, whose discrepancy is
# the same size with the opposite sign.
_TIMING_TOLERANCE = 0.01

# How many frames _frame_timing() decodes after landing near the end before deciding
# the frame count is an underestimate, rather than decoding the rest of the video.
_MAX_TIMING_TAIL_FRAMES = 10


class VideoReader:
    """VideoReader.

    Uses OpenCV to open a video file and read frames.

    Frames are numbered by their position in the file, counting from 0, which is how
    pose files are indexed. OpenCV's own frame seeking does not always respect that:
    it converts a frame number to a time using the file's reported frame rate, so in a
    file whose reported rate is slightly off it lands a frame early or late. ``seek()``
    corrects for this by checking each landing against the frame's timestamp.
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
        # True when seek() has already decoded the next frame to return
        self._frame_pending = False
        # (first frame timestamp, frame duration) in milliseconds, measured on first seek
        self._timing: tuple[float, float] | None = None
        self._timing_measured = False
        # warn about an inexact seek only once per file, since scrubbing seeks repeatedly
        self._warned_inexact_seek = False
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

        The next call to ``load_next_frame()`` returns frame ``index``.

        OpenCV can land a frame or two away from the requested frame: it converts the
        frame number to a time using the reported frame rate, and some files report a
        rate that is slightly off. For example, a 30 fps recording whose duration
        covers one frame interval fewer than its frame count reports 30.000278 fps,
        and every seek past its midpoint lands one frame early. So this lands, works
        out which frame it actually reached from that frame's timestamp, and steps
        forward to the requested frame, or lands again further back if it overshot.

        Args:
            index: Frame index to seek to.

        Raises:
            OSError: If the video stream has been closed.

        Note:
            The correction assumes frames are evenly spaced in time. For a file with
            a variable frame rate it can still land a frame away from ``index``.
        """
        if self.stream is None:
            raise OSError("video stream is closed")

        self._frame_pending = False
        timing = self._frame_timing()
        if timing is None:
            # no reliable timing to check against, so trust OpenCV's position
            if self.stream.set(cv2.CAP_PROP_POS_FRAMES, index):
                self._frame_index = index
            return

        land_at = index
        for _ in range(_MAX_SEEK_ATTEMPTS):
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, max(land_at, 0))
            reached = self._grab_frame_number(timing)
            if reached is None and index < self._num_frames:
                # landed past the end of the file, overshooting a frame that exists
                land_at -= 1
                continue
            while reached is not None and reached < index:
                reached = self._grab_frame_number(timing)

            if reached is None:
                # past the last frame: the next load returns the end-of-file marker
                self._frame_index = index
                return
            if reached == index:
                self._frame_index = index
                self._frame_pending = True
                return

            # overshot: land again far enough back to come up short of index
            land_at -= reached - index + 1

        if not self._warned_inexact_seek:
            self._warned_inexact_seek = True
            logger.warning(
                "Unable to seek %s to frame %d exactly (variable frame rate?); frames may "
                "be offset after seeking in this video",
                self._filename,
                index,
            )
        if self.stream.set(cv2.CAP_PROP_POS_FRAMES, index):
            self._frame_index = index

    def _grab_frame_number(self, timing: tuple[float, float]) -> int | None:
        """Decode the next frame and return its frame number, or None at end of file."""
        if not self.stream.grab():
            return None
        first_ms, frame_ms = timing
        return round((self.stream.get(cv2.CAP_PROP_POS_MSEC) - first_ms) / frame_ms)

    def _frame_timing(self) -> tuple[float, float] | None:
        """Return ``(first frame timestamp, frame duration)`` in milliseconds.

        Measured once, from the timestamps of the first and last frames, rather than
        derived from the reported frame rate, which is what can be off. Returns None,
        and stays None, if the file cannot be measured or the measurement disagrees
        with the reported frame rate (for example because the frame count is only an
        estimate). Moves the stream position, so callers must seek afterwards.
        """
        if self._timing_measured:
            return self._timing
        self._timing_measured = True

        if self._num_frames < 2 or not self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0):
            return None
        if not self.stream.grab():
            return None
        first_ms = self.stream.get(cv2.CAP_PROP_POS_MSEC)

        # land near the end and read to the last frame; where exactly it lands does
        # not matter, only the timestamp of the final frame
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, self._num_frames - 3)
        last_ms = None
        for _ in range(_MAX_TIMING_TAIL_FRAMES):
            if not self.stream.grab():
                break
            last_ms = self.stream.get(cv2.CAP_PROP_POS_MSEC)
        else:
            # still decoding well past the reported last frame: the count is an
            # underestimate, so don't decode the rest of the video to measure
            logger.warning(
                "Frame count of %s is too low to measure frame timing; seeking may be "
                "off by a frame",
                self._filename,
            )
            return None
        if last_ms is None or last_ms <= first_ms:
            logger.debug("Unable to measure frame timing of %s", self._filename)
            return None

        frame_ms = (last_ms - first_ms) / (self._num_frames - 1)
        reported_ms = 1000.0 / self.stream.get(cv2.CAP_PROP_FPS)
        tolerance = _TIMING_TOLERANCE + 1.0 / (self._num_frames - 1)
        if abs(frame_ms - reported_ms) > tolerance * reported_ms:
            logger.warning(
                "Frame timing of %s (%.4f ms/frame) disagrees with its frame rate "
                "(%.4f ms/frame); seeking may be off by a frame",
                self._filename,
                frame_ms,
                reported_ms,
            )
            return None

        self._timing = (first_ms, frame_ms)
        logger.debug("Measured frame timing of %s: %.6f ms/frame", self._filename, frame_ms)
        return self._timing

    def load_next_frame(self) -> dict[str, object]:
        """Grab the next frame from the file.

        Raises:
            OSError: If the video stream has been closed.
        """
        if self.stream is None:
            raise OSError("video stream is closed")

        if self._frame_pending:
            # seek() already decoded this frame while checking where it landed
            self._frame_pending = False
            (grabbed, frame) = self.stream.retrieve()
        else:
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
        """Get the number of frames by inspecting the video file.

        Args:
            path: path to video file.

        Returns:
            Total frame count.

        Raises:
            OSError: if unable to open the specified video.
        """
        _fps, num_frames = get_fps_and_nframes(path)
        return num_frames
