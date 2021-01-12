import cv2


def get_frame_count(video_path: str):
    """
    Get the number of frames in a video file. Raises an IOError if unable to
    open the video specified
    :param video_path: string containing path to video file
    :return: Integer number of frames in video.
    :raises: IOError if unable to open video file
    """
    # open video file
    stream = cv2.VideoCapture(video_path)
    if not stream.isOpened():
        raise IOError(f"unable to open {video_path}")

    return int(stream.get(cv2.CAP_PROP_FRAME_COUNT))


def get_fps(video_path: str):
    # open video file
    stream = cv2.VideoCapture(video_path)
    if not stream.isOpened():
        raise IOError(f"unable to open {video_path}")

    return stream.get(cv2.CAP_PROP_FPS)
