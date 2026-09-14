"""Read metadata (frame rate, frame count) from a video file without decoding it."""

from pathlib import Path

import cv2


def get_fps_and_nframes(video_path: str | Path) -> tuple[int, int]:
    """Get the frames per second and frame count from a video in a single open.

    Reads both properties from one ``cv2.VideoCapture`` handle so callers that
    need only one of them still pay for a single file open. The capture is
    always released, including when the video cannot be opened.

    Args:
        video_path: path to video file.

    Returns:
        Tuple of ``(fps, num_frames)`` where ``fps`` is rounded to an int.

    Raises:
        OSError: if unable to open the specified video.
    """
    stream = cv2.VideoCapture(str(video_path))
    try:
        if not stream.isOpened():
            raise OSError(f"unable to open {video_path}")
        fps = round(stream.get(cv2.CAP_PROP_FPS))
        num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        stream.release()
    return fps, num_frames
