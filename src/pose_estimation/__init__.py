from pathlib import Path

from .pose_est_v3 import PoseEstimationV3
from .pose_est_v2 import PoseEstimationV2


class PoseEstFactory:
    """
    class that can take a h5 file path and instantiate the correct object
    based on the version
    """

    @staticmethod
    def open(path: Path):
        if path.name.endswith('v2.h5'):
            return PoseEstimationV2(path)
        elif path.name.endswith('v3.h5'):
            return PoseEstimationV3(path)
        else:
            raise ValueError("not a valid pose estimate filename")


def get_pose_path(video_path: Path):
    """
    take a path to a video file and return the path to the corresponding
    pose_est h5 file
    :param video_path: Path to video file in project
    :return: Path object representing location of corresponding pose_est h5 file
    :raises ValueError: if video_path does not have corresponding pose_est file
    """

    file_base = video_path.with_suffix('')

    if video_path.with_name(file_base.name + '_pose_est_v3.h5').exists():
        return video_path.with_name(file_base.name + '_pose_est_v3.h5')
    elif video_path.with_name(file_base.name + '_pose_est_v2.h5').exists():
        return video_path.with_name(file_base.name + '_pose_est_v2.h5')
    else:
        raise ValueError("Video does not have pose file")

