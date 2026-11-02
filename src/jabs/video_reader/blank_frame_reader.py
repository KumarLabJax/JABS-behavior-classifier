"""Frame source that produces solid-color frames in place of a video file."""

import time
from typing import ClassVar

import numpy as np
import numpy.typing as npt


class BlankFrameReader:
    """Stands in for a :class:`VideoReader` when there is no video to read.

    Implements the same reading interface as ``VideoReader`` but every frame is the
    same solid color, so pose data can be played back without its video.
    """

    _EOF: ClassVar[dict[str, object]] = {"data": None, "index": -1}

    def __init__(
        self,
        num_frames: int,
        fps: int,
        width: int,
        height: int,
        color: tuple[int, int, int] = (0, 0, 0),
        filename: str = "",
    ) -> None:
        """Initialize a BlankFrameReader.

        Args:
            num_frames: Number of frames to produce.
            fps: Frame rate to play the frames at.
            width: Frame width in pixels.
            height: Frame height in pixels.
            color: Frame color as (blue, green, red), matching OpenCV's channel order.
            filename: Name reported by :attr:`filename`, e.g. the source pose file.

        Raises:
            ValueError: If fps, width or height is not positive, or num_frames is
                negative.
        """
        if fps <= 0:
            raise ValueError(f"invalid frame rate ({fps})")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid frame size ({width}x{height})")
        if num_frames < 0:
            raise ValueError(f"invalid frame count ({num_frames})")

        self._num_frames = num_frames
        self._fps = fps
        self._duration = 1.0 / fps
        self._width = width
        self._height = height
        self._filename = filename
        self._frame_index = 0
        self._frame: npt.NDArray[np.uint8] = np.empty((height, width, 3), dtype=np.uint8)
        self._frame[:] = color

    @property
    def num_frames(self) -> int:
        """Get total number of frames."""
        return self._num_frames

    @property
    def fps(self) -> int:
        """Get frames per second."""
        return self._fps

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return width and height of the frames."""
        return self._width, self._height

    @property
    def filename(self) -> str:
        """Return the name given for this frame source."""
        return self._filename

    def close(self) -> None:
        """No-op; there are no resources to release."""

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

        Args:
            index: Frame index to seek to.
        """
        self._frame_index = index

    def load_next_frame(self) -> dict[str, object]:
        """Return the next frame, or the end-of-file marker past the last frame.

        The returned frame is a copy, so callers may modify it freely.
        """
        if not 0 <= self._frame_index < self._num_frames:
            return self._EOF
        data = {
            "data": self._frame.copy(),
            "index": self._frame_index,
            "duration": self._duration,
        }
        self._frame_index += 1
        return data
