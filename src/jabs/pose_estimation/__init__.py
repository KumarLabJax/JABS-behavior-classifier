"""JABS pose file handler module"""

import re
from pathlib import Path

import h5py

from jabs.core.abstract.pose_est import MINIMUM_CONFIDENCE, PoseEstimation
from jabs.core.exceptions import PoseHashException

from .pose_est_v2 import PoseEstimationV2
from .pose_est_v3 import PoseEstimationV3
from .pose_est_v4 import PoseEstimationV4
from .pose_est_v5 import PoseEstimationV5
from .pose_est_v6 import PoseEstimationV6
from .pose_est_v7 import PoseEstimationV7
from .pose_est_v8 import PoseEstimationV8


def open_pose_file(path: Path, cache_dir: Path | None = None):
    """open a pose file using the correct PoseEstimation subclass based on the version implied by the filename"""
    if path.name.endswith("v2.h5"):
        return PoseEstimationV2(path, cache_dir)
    elif path.name.endswith("v3.h5"):
        return PoseEstimationV3(path, cache_dir)
    elif path.name.endswith("v4.h5"):
        return PoseEstimationV4(path, cache_dir)
    elif path.name.endswith("v5.h5"):
        return PoseEstimationV5(path, cache_dir)
    elif path.name.endswith("v6.h5"):
        return PoseEstimationV6(path, cache_dir)
    elif path.name.endswith("v7.h5"):
        return PoseEstimationV7(path, cache_dir)
    elif path.name.endswith("v8.h5"):
        return PoseEstimationV8(path, cache_dir)
    else:
        raise ValueError("not a valid pose estimate filename")


def get_pose_path(video_path: Path, pose_dir: Path | None = None):
    """take a path to a video file and return the path to the corresponding pose_est h5 file

    Args:
        video_path: Path to video file in project
        pose_dir: Optional directory to search for pose files. If omitted,
            search beside ``video_path``.

    Returns:
        Path object representing location of corresponding pose_est h5 file

    Raises:
        ValueError: if video_path does not have corresponding pose_est file
    """
    file_base = video_path.with_suffix("")
    search_dir = pose_dir if pose_dir is not None else video_path.parent

    # default to the highest version pose file for a video
    supported_versions = [8, 7, 6, 5, 4, 3, 2]
    for version in supported_versions:
        pose_file = search_dir / f"{file_base.name}_pose_est_v{version}.h5"
        if pose_file.exists():
            return pose_file
    raise ValueError("Video does not have pose file")


# matches the version suffix of a pose file name, e.g. "_v6.h5" in
# "video_pose_est_v6.h5". Anchored at the end so only the filename's suffix can
# supply the version.
_POSE_VERSION_RE = re.compile(r"_v(\d+)\.h5$")


def get_pose_file_major_version(path: Path) -> int:
    """get the major version of a pose file from the _filename_

    Note: does not inspect contents of file, assumes the file name follows the
    JABS convention ``<video name>_pose_est_v<major version>.h5``.

    Args:
        path: path of pose file

    Returns:
        integer major version number

    Raises:
        ValueError: if the file name does not end with a ``_v<major version>.h5``
            suffix
    """
    match = _POSE_VERSION_RE.search(path.name)
    if match is None:
        raise ValueError(f"'{path.name}' is not a valid pose file name")
    return int(match.group(1))


def get_frames_from_file(path: Path):
    """peek into a pose_est file to count number of frames"""
    with h5py.File(path, "r") as pose_h5:
        vid_grp = pose_h5["poseest"]
        return vid_grp["points"].shape[0]


def get_static_objects_in_file(path: Path):
    """peek into a pose file to get a list of the static objects it contains

    Args:
        path: path of pose file

    Returns:
        list of static object names contained in pose file
    """
    if get_pose_file_major_version(path) >= 5:
        with h5py.File(path, "r") as pose_h5:
            if "static_objects" in pose_h5:
                return list(pose_h5["static_objects"].keys())
    return []


def get_points_per_lixit(path: Path) -> int:
    """inspect a pose file to get the number of keypoints per lixit

    returns zero if the pose file does not have any lixit keypoints.
    """
    points_per_lixit = 0
    if get_pose_file_major_version(path) >= 5:
        with h5py.File(path, "r") as pose_h5:
            if "static_objects" in pose_h5 and "lixit" in pose_h5["static_objects"]:
                points_per_lixit = 3 if pose_h5["static_objects"]["lixit"].ndim == 3 else 1
    return points_per_lixit
