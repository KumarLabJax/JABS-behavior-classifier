"""parallel_workers.py

Provides stateless, picklable worker functions for parallel feature extraction
and per-video metadata scanning. These functions are designed to
be executed by ProcessPoolExecutor workers, managed by Project.
"""

import json
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import h5py
import numpy as np
import pandas as pd

import jabs.feature_extraction as fe
from jabs.core.enums import CacheFormat
from jabs.pose_estimation import open_pose_file
from jabs.video_reader import VideoReader
from jabs.video_reader.utilities import get_fps

from .track_labels import TrackLabels
from .video_labels import VideoLabels

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation


class FeatureLoadJobSpec(TypedDict):
    """Specification of a single video feature extraction job.

    This TypedDict encapsulates all necessary information for loading
    a single video's features *for labeled frames* in a parallel worker.
    """

    video: str
    video_path: Path
    pose_path: Path
    annotations_path: Path
    feature_dir: Path
    cache_dir: Path | None
    behavior_settings: dict[str, object]
    behavior_name: str | None
    cache_format: str


class CollectFeatureLoadResult(TypedDict):
    """Result of collecting features from a single video."""

    per_frame: list[pd.DataFrame]
    window: list[pd.DataFrame]
    labels: list[np.ndarray]
    group_keys: list[tuple[str, int]]


class VideoScanJobSpec(TypedDict):
    """Specification for a single-video metadata scan job.

    All fields needed by :func:`scan_video_metadata` to open one HDF5 file
    and collect everything :class:`~jabs.project.video_manager.VideoManager`
    and :class:`~jabs.project.feature_manager.FeatureManager` require at
    project-load time.
    """

    video: str
    video_path: Path
    pose_path: Path
    pose_major_version: int
    scan_frame_counts: bool


class VideoScanResult(TypedDict):
    """Per-video metadata collected by :func:`scan_video_metadata`."""

    video: str
    hdf5_frame_count: int
    video_frame_count: int | None
    identity_count: int
    static_objects: list[str]
    lixit_keypoints: int
    has_cm_per_pixel: bool


def _get_identity_count(pose_h5: "h5py.File", major_version: int) -> int:
    """Read identity count from an open HDF5 file without loading keypoint data.

    Args:
        pose_h5: Open h5py File object for a pose estimation HDF5 file.
        major_version: Pose file major version (from filename).

    Returns:
        Number of identities encoded in the pose file.
    """
    if major_version <= 2:
        return 1
    pose_grp = pose_h5["poseest"]
    if major_version == 3:
        # Shape is (n_frames, n_instances, n_keypoints, 2); shape read only.
        return pose_grp["points"].shape[1]
    # V4+: prefer instance_id_center (shape attribute only, no data load).
    if "instance_id_center" in pose_grp:
        return pose_grp["instance_id_center"].shape[0]
    # Fallback: compute max identity from instance_embed_id (smaller dataset).
    if "instance_embed_id" in pose_grp and "id_mask" in pose_grp:
        id_mask = pose_grp["id_mask"][:]
        instance_embed_id = pose_grp["instance_embed_id"][:]
        if instance_embed_id.shape[1] > 0:
            valid = id_mask == 0
            if valid.any():
                return int(instance_embed_id[valid].max())
    return 0


def scan_video_metadata(job: VideoScanJobSpec) -> VideoScanResult:
    """Collect per-video metadata by opening each HDF5 pose file exactly once.

    Reads everything :class:`~jabs.project.video_manager.VideoManager` and
    :class:`~jabs.project.feature_manager.FeatureManager` need at project-load
    time — frame counts, identity count, static objects, lixit keypoints, and
    distance-unit attributes — in a single ``h5py.File`` open.

    This function is stateless and picklable so it can be dispatched to
    :class:`~jabs.core.utils.process_pool_manager.ProcessPoolManager` workers.

    Args:
        job: Scan job specification, including paths and options.

    Returns:
        Per-video metadata collected from the pose HDF5 file (and optionally
        the video file when ``scan_frame_counts`` is ``True``).
    """
    video = job["video"]
    video_path = job["video_path"]
    pose_path = job["pose_path"]
    major_version = job["pose_major_version"]
    scan_frame_counts = job["scan_frame_counts"]

    with h5py.File(pose_path, "r") as pose_h5:
        hdf5_frame_count: int = pose_h5["poseest"]["points"].shape[0]
        identity_count: int = _get_identity_count(pose_h5, major_version)

        # Static objects are only present in V5+.
        static_objects: list[str] = []
        if major_version >= 5 and "static_objects" in pose_h5:
            static_objects = list(pose_h5["static_objects"].keys())

        # Lixit keypoint count (0 if no lixit present).
        lixit_keypoints: int = 0
        if "lixit" in static_objects:
            lixit_keypoints = 3 if pose_h5["static_objects"]["lixit"].ndim == 3 else 1

        # Distance unit: check for cm_per_pixel in poseest group attributes.
        has_cm_per_pixel: bool = pose_h5["poseest"].attrs.get("cm_per_pixel") is not None

    video_frame_count: int | None = None
    if scan_frame_counts:
        video_frame_count = VideoReader.get_nframes_from_file(video_path)

    return VideoScanResult(
        video=video,
        hdf5_frame_count=hdf5_frame_count,
        video_frame_count=video_frame_count,
        identity_count=identity_count,
        static_objects=static_objects,
        lixit_keypoints=lixit_keypoints,
        has_cm_per_pixel=has_cm_per_pixel,
    )


def _load_video_labels(annotations_path: Path, pose_est: "PoseEstimation") -> VideoLabels | None:
    """Load VideoLabels from a JSON file if present; else None."""
    ap = annotations_path
    if not ap.exists():
        return None
    with ap.open("r") as f:
        data = json.load(f)
    return VideoLabels.load(data, pose_est)


def collect_labeled_features(job: FeatureLoadJobSpec) -> CollectFeatureLoadResult:
    """Extracts features for labeled frames for a single video.

    This function loads per-frame and window features for a given video. If features
    are not pre-computed then this will result in features being computed directly
    from pose. It is intended to be used in parallel in Project.get_labeled_features().
    Returns features for labeled frame only, features for unlabeled frames are discarded.

    Note: this function is a standalone function to facilitate pickling for parallel
    processing via ProcessPoolExecutor. It should not rely on any instance-specific
    state, and is passed all necessary data via the JobSpec argument. A Project instance
    maintains a pool of workers that call this function in parallel from
    Project.load_labeled_features() in order to speed up feature extraction across
    multiple videos.

    Args:
        job (FeatureLoadJobSpec): Specification of the video and settings for feature extraction.

    Returns:
        CollectFeatureLoadResult: Collected per-frame and window features, labels, and
            identity mapping for the video.
    """
    video: str = job["video"]
    video_path = job["video_path"]
    pose_path = job["pose_path"]
    annotations_path = job["annotations_path"]
    feature_dir = job["feature_dir"]
    cache_dir = job["cache_dir"]
    behavior_settings: dict = job["behavior_settings"]
    behavior_name = job.get("behavior_name")

    # On macOS, scipy.linalg.lstsq (called by signal.stft's "linear" detrend)
    # uses Apple's Accelerate LAPACK, which segfaults when invoked from a
    # forked child process.  Switch to the pure-numpy detrend path only on
    # macOS to avoid this.  This flag is process-local so the main process
    # and non-macOS workers are unaffected.
    if sys.platform == "darwin" and multiprocessing.parent_process() is not None:
        fe.feature_base_class._use_numpy_detrend = True

    pose_est = open_pose_file(pose_path, cache_dir)
    fps = get_fps(str(video_path))

    # Get labels for video (might be None)
    # this loads all labels from the annotations file for any labeled behavior
    labels_obj = _load_video_labels(annotations_path, pose_est)
    if labels_obj is None:
        return {"per_frame": [], "window": [], "labels": [], "group_keys": []}

    per_frame_list: list[pd.DataFrame] = []
    window_list: list[pd.DataFrame] = []
    labels_list: list[np.ndarray] = []
    group_keys: list[tuple[str, int]] = []

    for identity in pose_est.identities:
        # Extract labels for this (video, identity) pair for the specified behavior
        labels = labels_obj.get_track_labels(str(identity), behavior_name).get_labels()

        # Exclude frames where identity does not exist
        # NOTE: in the future we might want to handle this differently, since we can still predict
        # behavior even when the identity is not detected in a frame (e.g., occluded) due to
        # temporal context from surrounding frames provided by window features
        labels[pose_est.identity_mask(identity) == 0] = TrackLabels.Label.NONE

        # Skip identities without any BEHAVIOR/NOT_BEHAVIOR labels
        if (
            (labels == TrackLabels.Label.BEHAVIOR) | (labels == TrackLabels.Label.NOT_BEHAVIOR)
        ).sum() == 0:
            continue

        # Feature extraction for this identity
        cache_format = CacheFormat(job["cache_format"])
        features = fe.IdentityFeatures(
            video,
            identity,
            feature_dir,
            pose_est,
            fps=fps,
            op_settings=behavior_settings,
            cache_format=cache_format,
        )

        # Per-frame features
        per_frame = features.get_per_frame_flat(labels)

        # Window features
        window_size: int = behavior_settings["window_size"]
        window_features = features.get_window_features(window_size, labels)
        window_features = fe.IdentityFeatures.merge_window_features(window_features)

        # Keep only labeled frames
        per_frame_list.append(pd.DataFrame(per_frame))
        window_list.append(pd.DataFrame(window_features))
        labels_list.append(labels[labels != TrackLabels.Label.NONE])
        group_keys.append((video, int(identity)))

    return {
        "per_frame": per_frame_list,
        "window": window_list,
        "labels": labels_list,
        "group_keys": group_keys,
    }
