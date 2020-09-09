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
            print(path)
            raise ValueError("not a valid pose estimate filename")


def get_pose_path(video_path: Path):

    file_base = video_path.with_suffix('')

    if video_path.with_name(file_base.name + '_pose_est_v3.h5').exists():
        return video_path.with_name(file_base.name + '_pose_est_v3.h5')
    elif video_path.with_name(file_base.name + '_pose_est_v2.h5').exists():
        return video_path.with_name(file_base.name + '_pose_est_v2.h5')
    else:
        raise ValueError("Video does not have pose file")

