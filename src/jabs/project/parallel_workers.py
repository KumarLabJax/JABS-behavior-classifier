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
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import CacheFormat
from jabs.pose_estimation import open_pose_file
from jabs.video_reader import VideoReader
from jabs.video_reader.utilities import get_fps

from .track_labels import TrackLabels
from .video_labels import VideoLabels

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation


class _BaseFeatureLoadJobSpec(TypedDict):
    """Shared fields for binary and multi-class feature-load jobs."""

    video: str
    video_path: Path
    pose_path: Path
    annotations_path: Path
    feature_dir: Path
    cache_dir: Path | None
    behavior_settings: dict[str, object]
    cache_format: str


class BinaryFeatureLoadJobSpec(_BaseFeatureLoadJobSpec):
    """Single-video feature-load job for binary classification."""

    behavior_name: str | None


class MulticlassFeatureLoadJobSpec(_BaseFeatureLoadJobSpec):
    """Single-video feature-load job for multi-class classification."""

    behavior_names: list[str]


class BinaryFeatureResult(TypedDict):
    """Result of collecting binary-labeled features from a single video."""

    per_frame: list[pd.DataFrame]
    window: list[pd.DataFrame]
    labels: list[np.ndarray]
    group_keys: list[tuple[str, int]]


class MulticlassFeatureResult(TypedDict):
    """Result of collecting multi-class-labeled features from a single video."""

    per_frame: list[pd.DataFrame]
    window: list[pd.DataFrame]
    labels_by_behavior: list[dict[str, np.ndarray]]
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
    time - frame counts, identity count, static objects, lixit keypoints, and
    distance-unit attributes - in a single ``h5py.File`` open.

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


def _apply_macos_fork_lapack_workaround() -> None:
    """Avoid Accelerate LAPACK segfaults in forked children on macOS.

    scipy.linalg.lstsq (called by signal.stft's "linear" detrend) uses Apple's
    Accelerate LAPACK, which segfaults when invoked from a forked child
    process. Switch to the pure-numpy detrend path only on macOS to avoid this.
    The flag is process-local, so the main process and non-macOS workers are
    unaffected.
    """
    if sys.platform == "darwin" and multiprocessing.parent_process() is not None:
        fe.feature_base_class._use_numpy_detrend = True


def _open_pose_and_labels(
    job: _BaseFeatureLoadJobSpec,
) -> "tuple[PoseEstimation, VideoLabels | None, float]":
    """Set up the macOS workaround and open pose + label resources for a job."""
    _apply_macos_fork_lapack_workaround()
    pose_est = open_pose_file(job["pose_path"], job["cache_dir"])
    fps = get_fps(str(job["video_path"]))
    labels_obj = _load_video_labels(job["annotations_path"], pose_est)
    return pose_est, labels_obj, fps


def _extract_identity_features(
    video: str,
    identity: int,
    pose_est: "PoseEstimation",
    feature_dir: Path,
    behavior_settings: dict,
    cache_format: str,
    fps: float,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-frame and window features for one identity's labeled frames.

    Args:
        video: Video filename (used to locate cached features).
        identity: Identity index within the video.
        pose_est: Open pose estimation object.
        feature_dir: Project feature directory.
        behavior_settings: Behavior-scoped settings dict (must include ``window_size``).
        cache_format: Cache format string from project settings.
        fps: Video frames per second.
        labels: Per-frame label vector for this identity; rows with
            ``TrackLabels.Label.NONE`` are dropped from the output features.

    Returns:
        Tuple ``(per_frame_df, window_df)`` containing only the labeled rows.
    """
    cache_format_enum = CacheFormat(cache_format)
    features = fe.IdentityFeatures(
        video,
        identity,
        feature_dir,
        pose_est,
        fps=fps,
        op_settings=behavior_settings,
        cache_format=cache_format_enum,
    )
    per_frame = features.get_per_frame_flat(labels)
    window_size: int = behavior_settings["window_size"]
    window_features = features.get_window_features(window_size, labels)
    window_features = fe.IdentityFeatures.merge_window_features(window_features)
    return pd.DataFrame(per_frame), pd.DataFrame(window_features)


def collect_binary_labeled_features(job: BinaryFeatureLoadJobSpec) -> BinaryFeatureResult:
    """Extract per-frame and window features for one video, binary mode.

    For every identity in the video, frames not assigned the requested behavior
    label are dropped before feature extraction. Identities with no labeled
    frames are skipped entirely.

    This function is stateless and picklable so it can be dispatched to
    process-pool workers from :class:`~jabs.project.Project`.

    Args:
        job: Video and feature-extraction settings.

    Returns:
        Collected per-frame/window features, label arrays, and per-identity
        group keys for the video.
    """
    pose_est, labels_obj, fps = _open_pose_and_labels(job)
    if labels_obj is None:
        return {"per_frame": [], "window": [], "labels": [], "group_keys": []}

    video = job["video"]
    behavior_name = job["behavior_name"]
    behavior_settings: dict = job["behavior_settings"]
    feature_dir = job["feature_dir"]
    cache_format = job["cache_format"]

    per_frame_list: list[pd.DataFrame] = []
    window_list: list[pd.DataFrame] = []
    labels_list: list[np.ndarray] = []
    group_keys: list[tuple[str, int]] = []

    for identity in pose_est.identities:
        identity_mask = pose_est.identity_mask(identity).astype(bool)
        labels = labels_obj.get_track_labels(str(identity), behavior_name).get_labels()
        # Exclude frames where the identity does not exist.
        # NOTE: in the future we might want to handle this differently, since we
        # can still predict behavior even when the identity is not detected in
        # a frame (e.g., occluded) thanks to window features.
        labels[~identity_mask] = TrackLabels.Label.NONE

        if (labels != TrackLabels.Label.NONE).sum() == 0:
            continue

        per_frame_df, window_df = _extract_identity_features(
            video, identity, pose_est, feature_dir, behavior_settings, cache_format, fps, labels
        )
        per_frame_list.append(per_frame_df)
        window_list.append(window_df)
        labels_list.append(labels[labels != TrackLabels.Label.NONE])
        group_keys.append((video, int(identity)))

    return {
        "per_frame": per_frame_list,
        "window": window_list,
        "labels": labels_list,
        "group_keys": group_keys,
    }


def collect_multiclass_labeled_features(
    job: MulticlassFeatureLoadJobSpec,
) -> MulticlassFeatureResult:
    """Extract per-frame and window features for one video, multi-class mode.

    Frames are included only when they have an explicit
    ``TrackLabels.Label.BEHAVIOR`` label in at least one class track (including
    the reserved ``MULTICLASS_NONE_BEHAVIOR`` background track). Identities
    with no labeled frames are skipped entirely.

    This function is stateless and picklable so it can be dispatched to
    process-pool workers from :class:`~jabs.project.Project`.

    Args:
        job: Video and feature-extraction settings (must include
            ``behavior_names``).

    Returns:
        Collected per-frame/window features, per-behavior label arrays, and
        per-identity group keys for the video.
    """
    pose_est, labels_obj, fps = _open_pose_and_labels(job)
    if labels_obj is None:
        return {"per_frame": [], "window": [], "labels_by_behavior": [], "group_keys": []}

    behavior_names = job["behavior_names"]
    if not behavior_names:
        raise ValueError("behavior_names is required for multiclass feature collection")

    video = job["video"]
    behavior_settings: dict = job["behavior_settings"]
    feature_dir = job["feature_dir"]
    cache_format = job["cache_format"]
    behavior_tracks = [MULTICLASS_NONE_BEHAVIOR, *behavior_names]

    per_frame_list: list[pd.DataFrame] = []
    window_list: list[pd.DataFrame] = []
    labels_by_behavior_list: list[dict[str, np.ndarray]] = []
    group_keys: list[tuple[str, int]] = []

    for identity in pose_est.identities:
        identity_mask = pose_est.identity_mask(identity).astype(bool)

        labels_by_behavior: dict[str, np.ndarray] = {}
        include_mask = np.zeros(identity_mask.shape, dtype=bool)
        for behavior_key in behavior_tracks:
            behavior_labels = (
                labels_obj.get_track_labels(str(identity), behavior_key).get_labels().copy()
            )
            behavior_labels[~identity_mask] = TrackLabels.Label.NONE
            labels_by_behavior[behavior_key] = behavior_labels
            include_mask |= behavior_labels == TrackLabels.Label.BEHAVIOR

        # Include only frames with explicit BEHAVIOR labels in any class.
        labels = np.full(identity_mask.shape, TrackLabels.Label.NONE, dtype=np.int8)
        labels[include_mask] = TrackLabels.Label.BEHAVIOR

        if (labels != TrackLabels.Label.NONE).sum() == 0:
            continue

        per_frame_df, window_df = _extract_identity_features(
            video, identity, pose_est, feature_dir, behavior_settings, cache_format, fps, labels
        )
        per_frame_list.append(per_frame_df)
        window_list.append(window_df)
        labels_by_behavior_list.append(
            {key: arr[labels != TrackLabels.Label.NONE] for key, arr in labels_by_behavior.items()}
        )
        group_keys.append((video, int(identity)))

    return {
        "per_frame": per_frame_list,
        "window": window_list,
        "labels_by_behavior": labels_by_behavior_list,
        "group_keys": group_keys,
    }
