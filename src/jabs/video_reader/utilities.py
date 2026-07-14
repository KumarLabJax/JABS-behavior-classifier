import cv2


def get_frame_count(video_path: str):
    """Get the number of frames in a video file.

    Args:
        video_path: string containing path to video file

    Returns:
        Integer number of frames in video.

    Raises:
        OSError: if unable to open specified video
    """
    # open video file
    stream = cv2.VideoCapture(video_path)
    if not stream.isOpened():
        raise OSError(f"unable to open {video_path}")

    return int(stream.get(cv2.CAP_PROP_FRAME_COUNT))


def get_fps(video_path: str):
    """get the frames per second from a video file"""
    # open video file
    stream = cv2.VideoCapture(video_path)
    if not stream.isOpened():
        raise OSError(f"unable to open {video_path}")

    return round(stream.get(cv2.CAP_PROP_FPS))


def get_fps_and_nframes(video_path: str) -> tuple[int, int]:
    """Get the frames per second and frame count from a video in a single open.

    Reads both properties from one ``cv2.VideoCapture`` handle so callers that
    already need the FPS can obtain the frame count without a second file open.

    Args:
        video_path: string containing path to video file.

    Returns:
        Tuple of ``(fps, num_frames)`` where ``fps`` is rounded to an int.

    Raises:
        OSError: if unable to open the specified video.
    """
    stream = cv2.VideoCapture(video_path)
    if not stream.isOpened():
        raise OSError(f"unable to open {video_path}")

    fps = round(stream.get(cv2.CAP_PROP_FPS))
    num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    stream.release()
    return fps, num_frames
