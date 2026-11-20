"""JABS pose file handler module"""

import os
import re
from pathlib import Path

import h5py

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.exceptions import PoseHashException
from jabs.core.pose_skeleton import MINIMUM_CONFIDENCE
from jabs.core.utils.static_objects import lixit_keypoint_count

from .conversion_cache import open_unified
from .discovery import find_pose_file, is_legacy_pose_file, is_unified_pose_file
from .errors import PoseOpenError
from .pose import Pose
from .pose_est_v2 import PoseEstimationV2
from .pose_est_v3 import PoseEstimationV3
from .pose_est_v4 import PoseEstimationV4
from .pose_est_v5 import PoseEstimationV5
from .pose_est_v6 import PoseEstimationV6
from .pose_est_v7 import PoseEstimationV7
from .pose_est_v8 import PoseEstimationV8

# matches the version suffix of a pose file name, e.g. "_v6.h5" in
# "video_pose_est_v6.h5". Anchored at the end so only the filename's suffix can
# supply the version.
_POSE_VERSION_RE = re.compile(r"_v(\d+)\.h5$")

# when "1", legacy pose files open through the conversion cache as Pose instead
# of their legacy reader
_FORCE_UNIFIED_ENV = "JABS_FORCE_UNIFIED"

# reader class for each legacy pose file major version JABS can open;
# open_pose_file() dispatches legacy files on it.
_POSE_READERS: dict[int, type[PoseEstimation]] = {
    2: PoseEstimationV2,
    3: PoseEstimationV3,
    4: PoseEstimationV4,
    5: PoseEstimationV5,
    6: PoseEstimationV6,
    7: PoseEstimationV7,
    8: PoseEstimationV8,
}


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


def open_pose_file(path: Path, cache_dir: Path | None = None) -> PoseEstimation | Pose:
    """Open a pose file.

    A unified ``*_pose.h5`` file opens as :class:`Pose`. A legacy
    ``*_pose_est_v<N>.h5`` file opens with the legacy reader for the version
    its name declares (see :func:`get_pose_file_major_version`), unless the
    environment variable ``JABS_FORCE_UNIFIED`` is ``1``, which converts it
    through the conversion cache and opens it as :class:`Pose` too.

    Args:
        path: path of pose file
        cache_dir: optional directory the reader may use to cache the converted
            pose data

    Returns:
        :class:`Pose` for a unified (or forced) file, otherwise the
        PoseEstimation subclass instance for the file's format version

    Raises:
        PoseOpenError: if a unified (or forced) file cannot be opened
        ValueError: if the file name does not declare a major version, or declares
            a version JABS is unable to open
    """
    if is_unified_pose_file(path) or (
        is_legacy_pose_file(path) and os.environ.get(_FORCE_UNIFIED_ENV) == "1"
    ):
        return open_unified(path, cache_dir)
    version = get_pose_file_major_version(path)
    try:
        reader = _POSE_READERS[version]
    except KeyError:
        raise ValueError(
            f"'{path.name}': pose file major version {version} is not supported"
        ) from None
    return reader(path, cache_dir)


def get_pose_path(video_path: Path, pose_dir: Path | None = None) -> Path:
    """take a path to a video file and return the path to its pose file

    The unified ``<video>_pose.h5`` wins; otherwise the highest version legacy
    ``<video>_pose_est_v<N>.h5`` (see :func:`find_pose_file`).

    Args:
        video_path: Path to video file in project
        pose_dir: Optional directory to search for pose files. If omitted,
            search beside ``video_path``.

    Returns:
        Path object representing location of corresponding pose file

    Raises:
        ValueError: if video_path does not have corresponding pose file
    """
    pose_file = find_pose_file(video_path, pose_dir)
    if pose_file is None:
        raise ValueError("Video does not have pose file")
    return pose_file


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

    returns zero if the pose file does not have any lixit keypoints, which
    includes a lixit dataset that is present but empty -- the same rule the
    v5+ readers and the project scan apply.
    """
    points_per_lixit = 0
    if get_pose_file_major_version(path) >= 5:
        with h5py.File(path, "r") as pose_h5:
            if "static_objects" in pose_h5 and "lixit" in pose_h5["static_objects"]:
                points_per_lixit = lixit_keypoint_count(pose_h5["static_objects"]["lixit"].shape)
    return points_per_lixit
