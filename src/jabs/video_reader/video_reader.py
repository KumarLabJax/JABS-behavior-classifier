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

# How many frames past the first _frame_timing() decodes to measure the frame
# duration. More frames average out the rounding of coarse timestamps.
_TIMING_HEAD_FRAMES = 30

# How far, in frames, the measured position of a frame may be from its true position,
# given the rounding of the timestamps it is measured from, for it to be numbered.
_MAX_TIMING_UNCERTAINTY_FRAMES = 0.25

# How much further, in frames, a measured position may be from a whole frame number
# than that rounding explains before the frames are taken to be unevenly spaced.
_TIMING_SLACK_FRAMES = 0.05

# How far, in frames, the frame duration measured so far may have drifted by the frame
# _frame_timing() numbers next. Coarse timestamps (e.g. whole milliseconds) leave too
# much error to number the last frame of a long video straight from the first frames,
# so it numbers frames in stages, each as far out as this allows, refining the
# duration from each one.
_TIMING_STAGE_DRIFT_FRAMES = 0.2

# How many stages _frame_timing() takes to reach the last frame before giving up.
_MAX_TIMING_STAGES = 12

# How many frames past the reported end _frame_timing() decodes after landing near it
# before deciding the frame count is an underestimate, rather than decoding the rest
# of the video.
_MAX_TIMING_TAIL_FRAMES = 10

# How many frames before the reported end _frame_timing() lands to find the last
# frame, trying each in turn while landings decode nothing (a count that is too high).
_TIMING_TAIL_BACKOFF = (3, 10, 30, 100)

# How many frames seek() decodes to step forward from where OpenCV landed. OpenCV
# normally lands within a frame or two; a landing further short is treated as a failed
# seek rather than decoded through on the GUI thread.
_MAX_CATCH_UP_FRAMES = 60


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
        # True when seek() went past the last frame, so the next load is end of file
        self._at_end = False
        # (first frame timestamp, frame duration) in milliseconds, measured on first seek
        self._timing: tuple[float, float] | None = None
        self._timing_measured = False
        # number of the last frame, measured with the timing
        self._last_frame_index: int | None = None
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

    def seek(self, index: int) -> bool:
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
            index: Frame index to seek to. A negative index seeks to frame 0.

        Returns:
            True if the next ``load_next_frame()`` returns frame ``index``, or the
            end-of-file marker for an index past the last frame. False if the video
            could not be positioned; the next load then returns the frame the stream
            is at, numbered by OpenCV.

        Raises:
            OSError: If the video stream has been closed.

        Note:
            The correction assumes frames are evenly spaced in time. For a file with
            a variable frame rate it can still land a frame away from ``index``.
        """
        if self.stream is None:
            raise OSError("video stream is closed")

        index = max(index, 0)
        self._frame_pending = False
        self._at_end = False
        timing = self._frame_timing()
        if timing is None:
            # no reliable timing to check against, so trust OpenCV's position
            return self._plain_seek(index)
        if self._last_frame_index is not None and index > self._last_frame_index:
            # past the last frame: the next load returns the end-of-file marker
            self._frame_index = index
            self._at_end = True
            return True

        land_at = index
        for _ in range(_MAX_SEEK_ATTEMPTS):
            if not self.stream.set(cv2.CAP_PROP_POS_FRAMES, max(land_at, 0)):
                return self._plain_seek(index)
            reached = self._grab_frame_number(timing)
            if reached is None:
                # landed past the end of the file, overshooting a frame that exists
                land_at -= 1
                continue
            for _ in range(_MAX_CATCH_UP_FRAMES):
                if reached is None or reached >= index:
                    break
                reached = self._grab_frame_number(timing)
            if reached is None or reached < index:
                # landed too far short to step forward to index
                break
            if reached == index:
                self._frame_index = index
                self._frame_pending = True
                return True

            # overshot: land again far enough back to come up short of index
            land_at -= reached - index + 1

        # the correction doesn't work for this file, so stop paying for it on every seek
        self._timing = None
        logger.warning(
            "Unable to seek %s to frame %d exactly (variable frame rate?); seeking in this "
            "video may be off by a frame",
            self._filename,
            index,
        )
        return self._plain_seek(index)

    def _plain_seek(self, index: int) -> bool:
        """Seek with OpenCV alone, keeping the frame number in step if it fails."""
        if self.stream.set(cv2.CAP_PROP_POS_FRAMES, index):
            self._frame_index = index
            return True
        # the stream is wherever the failed seek, or measuring the timing, left it
        self._frame_index = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
        logger.warning("Unable to seek %s to frame %d", self._filename, index)
        return False

    def _grab_frame_number(self, timing: tuple[float, float]) -> int | None:
        """Decode the next frame and return its frame number, or None at end of file."""
        if not self.stream.grab():
            return None
        first_ms, frame_ms = timing
        return round((self.stream.get(cv2.CAP_PROP_POS_MSEC) - first_ms) / frame_ms)

    def _frame_timing(self) -> tuple[float, float] | None:
        """Return ``(first frame timestamp, frame duration)`` in milliseconds.

        Measured once, from frame timestamps rather than the reported frame rate or
        frame count, either of which can be off by a frame. Also records the number of
        the last frame. Returns None, and stays None, if the file cannot be measured
        precisely enough to number every frame. Moves the stream position, so callers
        must seek afterwards.
        """
        if self._timing_measured:
            return self._timing
        self._timing_measured = True

        measured = self._measure_timing()
        if measured is None:
            return None
        first_ms, frame_ms, self._last_frame_index = measured
        if self._last_frame_index != self._num_frames - 1:
            logger.debug(
                "%s reports %d frames but its last frame is frame %d",
                self._filename,
                self._num_frames,
                self._last_frame_index,
            )
        self._timing = (first_ms, frame_ms)
        logger.debug("Measured frame timing of %s: %.6f ms/frame", self._filename, frame_ms)
        return self._timing

    def _measure_timing(self) -> tuple[float, float, int] | None:
        """Measure ``(first frame timestamp, frame duration, last frame number)``.

        The first frames give a rough frame duration. That numbers a frame further
        on, whose timestamp gives a more precise duration, and so on until the last
        frame is numbered; with precise timestamps that takes one stage.
        """
        if self._num_frames < 2 or not self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0):
            return None
        head_ms = []
        for _ in range(_TIMING_HEAD_FRAMES + 1):
            if not self.stream.grab():
                break
            head_ms.append(self.stream.get(cv2.CAP_PROP_POS_MSEC))
        if len(head_ms) < 2:
            return None
        first_ms = head_ms[0]
        numbered = len(head_ms) - 1
        frame_ms = (head_ms[-1] - first_ms) / numbered
        if frame_ms <= 0:
            logger.debug("Unable to measure frame timing of %s", self._filename)
            return None
        if len(head_ms) <= _TIMING_HEAD_FRAMES:
            if len(head_ms) < self._num_frames - 1:
                # decoding stopped well before the reported end: an error, not the end
                logger.debug("Unable to decode the first frames of %s", self._filename)
                return None
            # decoded the whole video, so the head is exact
            return first_ms, frame_ms, numbered

        # how far a timestamp may be from even spacing, e.g. from a coarse time base
        rounding = max(abs(t - first_ms - i * frame_ms) for i, t in enumerate(head_ms))
        last_ms = self._last_frame_ms()
        if last_ms is None:
            return None
        for _ in range(_MAX_TIMING_STAGES):
            # frame_ms is off by at most error_ms, so frame n is off by n * error_ms
            error_ms = 2 * rounding / numbered
            reach = (
                _TIMING_STAGE_DRIFT_FRAMES * frame_ms / error_ms if error_ms > 0 else float("inf")
            )
            final = last_ms - first_ms <= reach * frame_ms
            if final:
                stage_ms = last_ms
            elif reach < 1.5 * numbered:
                break  # timestamps too coarse to make progress
            else:
                stage_ms = self._frame_ms_near(int(reach))
                if stage_ms is None:
                    return None
            position = (stage_ms - first_ms) / frame_ms
            index = round(position)
            # drift of the duration so far, plus the rounding of this timestamp
            uncertainty = (position * error_ms + rounding) / frame_ms
            if (
                index <= numbered
                or uncertainty > _MAX_TIMING_UNCERTAINTY_FRAMES
                or abs(position - index) > uncertainty + _TIMING_SLACK_FRAMES
            ):
                break
            frame_ms = (stage_ms - first_ms) / index
            numbered = index
            if final:
                return first_ms, frame_ms, index
        logger.warning(
            "Unable to measure frame timing of %s precisely; seeking may be off by a frame",
            self._filename,
        )
        return None

    def _frame_ms_near(self, index: int) -> float | None:
        """Return the timestamp of a frame at or near ``index``, or None on failure."""
        if not self.stream.set(cv2.CAP_PROP_POS_FRAMES, index) or not self.stream.grab():
            logger.debug("Unable to read frame %d of %s", index, self._filename)
            return None
        return self.stream.get(cv2.CAP_PROP_POS_MSEC)

    def _last_frame_ms(self) -> float | None:
        """Return the timestamp of the last frame, or None if it can't be reached."""
        # land near the end and read to the last frame; where exactly it lands does
        # not matter, only the timestamp of the final frame. A count that is too high
        # can land past the end, so back up further until a frame decodes.
        for back in _TIMING_TAIL_BACKOFF:
            if not self.stream.set(cv2.CAP_PROP_POS_FRAMES, max(self._num_frames - back, 0)):
                logger.debug("Unable to seek to the end of %s", self._filename)
                return None
            last_ms = None
            for _ in range(back + _MAX_TIMING_TAIL_FRAMES):
                if not self.stream.grab():
                    break
                last_ms = self.stream.get(cv2.CAP_PROP_POS_MSEC)
            else:
                # still decoding well past the reported last frame: the count is an
                # underestimate, or the seek failed, so don't decode the rest
                logger.warning(
                    "Unable to find the last frame of %s to measure frame timing; seeking "
                    "may be off by a frame",
                    self._filename,
                )
                return None
            if last_ms is not None:
                return last_ms
        logger.debug("No frames decode near the end of %s", self._filename)
        return None

    def load_next_frame(self) -> dict[str, object]:
        """Grab the next frame from the file.

        Raises:
            OSError: If the video stream has been closed.
        """
        if self.stream is None:
            raise OSError("video stream is closed")

        if self._at_end:
            return self._EOF
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
