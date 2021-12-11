import re
import typing
from pathlib import Path

import h5py

from .pose_est import PoseEstimation, PoseHashException
from .pose_est_v2 import PoseEstimationV2
from .pose_est_v3 import PoseEstimationV3
from .pose_est_v4 import PoseEstimationV4
from .pose_est_v5 import PoseEstimationV5


def open_pose_file(path: Path, cache_dir: typing.Optional[Path]=None):
    """
    open a pose file using the correct PoseEstimation subclass based on
    the version implied by the filename
    """
    if path.name.endswith('v2.h5'):
        return PoseEstimationV2(path, cache_dir)
    elif path.name.endswith('v3.h5'):
        return PoseEstimationV3(path, cache_dir)
    elif path.name.endswith('v4.h5'):
        return PoseEstimationV4(path, cache_dir)
    elif path.name.endswith('v5.h5'):
        return PoseEstimationV5(path, cache_dir)
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

    # default to the highest version pose file for a video
    if video_path.with_name(file_base.name + '_pose_est_v5.h5').exists():
        return video_path.with_name(file_base.name + '_pose_est_v5.h5')
    elif video_path.with_name(file_base.name + '_pose_est_v4.h5').exists():
        return video_path.with_name(file_base.name + '_pose_est_v4.h5')
    elif video_path.with_name(file_base.name + '_pose_est_v3.h5').exists():
        return video_path.with_name(file_base.name + '_pose_est_v3.h5')
    elif video_path.with_name(file_base.name + '_pose_est_v2.h5').exists():
        return video_path.with_name(file_base.name + '_pose_est_v2.h5')
    else:
        raise ValueError("Video does not have pose file")


def get_pose_file_major_version(path: Path):
    """
    get the major version of a pose file from the _filename_
    (does not inspect contents of file), assumes file name matches
    video_name_v[version number].h5
    :param path: path of pose file
    :return: integer major version number
    """
    v = re.search(r'_v([0-9])+\.h5', str(path)).group(1)
    return int(v)


def get_frames_from_file(path: Path):
    """ peek into a pose_est file to count number of frames """
    with h5py.File(path, 'r') as pose_h5:
        vid_grp = pose_h5['poseest']
        return vid_grp['points'].shape[0]


def get_static_objects_in_file(path: Path):
    """
    peek into a pose file to get a list of the static objects it contains
    :param path: path of pose file
    :return: list of static object names contained in pose file
    """

    if get_pose_file_major_version(path) >= 5:
        with h5py.File(path, 'r') as pose_h5:
            if 'static_objects' in pose_h5:
                return list(pose_h5['static_objects'].keys())
    return []
