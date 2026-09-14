"""Read metadata (frame rate, frame count) from a video file without decoding it."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2


@contextmanager
def _open_video(video_path: str | Path) -> Iterator[cv2.VideoCapture]:
    """Open a video for metadata reads, releasing the capture on the way out.

    Args:
        video_path: path to the video file.

    Yields:
        An open ``cv2.VideoCapture`` for the video.

    Raises:
        OSError: if unable to open the specified video.
    """
    stream = cv2.VideoCapture(str(video_path))
    try:
        if not stream.isOpened():
            raise OSError(f"unable to open {video_path}")
        yield stream
    finally:
        stream.release()


def get_frame_count(video_path: str | Path) -> int:
    """Get the number of frames in a video file.

    Args:
        video_path: path to video file

    Returns:
        Integer number of frames in video.

    Raises:
        OSError: if unable to open specified video
    """
    with _open_video(video_path) as stream:
        return int(stream.get(cv2.CAP_PROP_FRAME_COUNT))


def get_fps(video_path: str | Path) -> int:
    """Get the frames per second from a video file.

    Args:
        video_path: path to video file

    Returns:
        Frame rate rounded to the nearest integer.

    Raises:
        OSError: if unable to open specified video
    """
    with _open_video(video_path) as stream:
        return round(stream.get(cv2.CAP_PROP_FPS))


def get_fps_and_nframes(video_path: str | Path) -> tuple[int, int]:
    """Get the frames per second and frame count from a video in a single open.

    Reads both properties from one ``cv2.VideoCapture`` handle so callers that
    already need the FPS can obtain the frame count without a second file open.

    Args:
        video_path: path to video file.

    Returns:
        Tuple of ``(fps, num_frames)`` where ``fps`` is rounded to an int.

    Raises:
        OSError: if unable to open the specified video.
    """
    with _open_video(video_path) as stream:
        fps = round(stream.get(cv2.CAP_PROP_FPS))
        num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, num_frames
