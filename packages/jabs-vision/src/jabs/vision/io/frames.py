"""Video frame reading for pose inference (imageio + ffmpeg).

imageio[ffmpeg] is used deliberately to match the reference decoder (mtr), so
decoded pixels - and therefore argmax coordinates - are consistent with the
legacy pipeline.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 30


def read_frames(video: str | Path) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield ``(H, W, 3)`` RGB uint8 frames from a video.

    Args:
        video: Path to the input video.

    Yields:
        Successive RGB frames as uint8 arrays.
    """
    logger.info("reading frames from %s", video)
    reader = imageio.get_reader(str(video), format="ffmpeg")
    try:
        yield from reader.iter_data()
    finally:
        reader.close()


def video_fps(video: str | Path) -> int:
    """Return the video frame rate rounded to an int (default 30 if unknown).

    Args:
        video: Path to the input video.

    Returns:
        Frames per second, rounded to the nearest integer.
    """
    reader = imageio.get_reader(str(video), format="ffmpeg")
    try:
        fps = reader.get_meta_data().get("fps")
    finally:
        reader.close()
    if fps is None:
        logger.warning("no fps in %s metadata; defaulting to %d", video, _DEFAULT_FPS)
        return _DEFAULT_FPS
    return round(fps)
