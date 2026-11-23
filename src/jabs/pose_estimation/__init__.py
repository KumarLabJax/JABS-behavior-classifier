"""JABS pose file handler module"""

from pathlib import Path

from jabs.core.exceptions import PoseHashException
from jabs.core.pose_skeleton import MINIMUM_CONFIDENCE

from .conversion_cache import open_unified
from .discovery import find_pose_file, is_legacy_pose_file, is_pose_file, is_unified_pose_file
from .errors import PoseOpenError
from .metadata import (
    PoseMetadata,
    scan_model_metadata,
    scan_pose_metadata,
    scan_raw_model_metadata,
)
from .pose import Pose


def open_pose_file(path: Path, cache_dir: Path | None = None) -> Pose:
    """Open a pose file as :class:`Pose`.

    A unified ``*_pose.h5`` file opens directly. A legacy ``*_pose_est_v<N>.h5``
    file is converted through the conversion cache (see
    :func:`~jabs.pose_estimation.conversion_cache.open_unified`) and its
    conversion opens.

    Args:
        path: path of pose file
        cache_dir: optional project cache directory; conversions and convex
            hulls are cached here, or in the per-user cache when None

    Returns:
        The opened pose; the caller closes it.

    Raises:
        PoseOpenError: if the name is not a pose file name, or the file cannot
            be read, converted or opened
    """
    return open_unified(path, cache_dir)


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
