import typing
from pathlib import Path

import h5py

from .pose_est import PoseEstimation
from .pose_est_v2 import PoseEstimationV2
from .pose_est_v3 import PoseEstimationV3


def open_pose_file(path: Path, cache_dir: typing.Optional[Path]=None):
    """
    open a pose file using the correct PoseEstimation subclass based on
    the version implied by the filename
    """
    if path.name.endswith('v2.h5'):
        return PoseEstimationV2(path, cache_dir)
    elif path.name.endswith('v3.h5'):
        return PoseEstimationV3(path, cache_dir)
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


def get_frames_from_file(path: Path):
    """ peak into a pose_est file to count number of frames """

    with h5py.File(path, 'r') as pose_h5:
        vid_grp = pose_h5['poseest']
        return vid_grp['points'].shape[0]
