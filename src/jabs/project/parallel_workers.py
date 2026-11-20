"""parallel_workers.py

Provides stateless, picklable worker functions for parallel feature extraction
and per-video metadata scanning. These functions are designed to
be executed by ProcessPoolExecutor workers, managed by Project.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd

import jabs.feature_extraction as fe
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import CacheFormat
from jabs.io.annotations import read_document
from jabs.pose_estimation import PoseOpenError, open_pose_file
from jabs.pose_estimation.metadata import scan_pose_metadata
from jabs.video_reader import VideoReader
from jabs.video_reader.utilities import get_fps_and_nframes

from .track_labels import TrackLabels
from .video_labels import VideoLabels

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation


def preload_worker_modules() -> None:
    """Process-pool initializer that warms each worker's import cache.

    Intended to be passed as the ``initializer`` for the application
    ``ProcessPoolManager``. A worker must import this module to resolve this
    reference, and importing it (with its transitive dependencies) pulls in the
    heavy stack workers need: numpy and pandas directly, plus
    ``jabs.feature_extraction`` (which brings in scipy) and the pose/video
    readers (which bring in OpenCV). Doing that once at pool warm-up means the
    first project-load scan and the first training run don't pay the import
    cost. The function body is intentionally empty; the import triggered by
    resolving it is the whole point.
    """


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

    All fields needed by :func:`scan_video_metadata` to scan one pose file
    and collect everything :class:`~jabs.project.video_manager.VideoManager`
    and :class:`~jabs.project.feature_manager.FeatureManager` require at
    project-load time.
    """

    video: str
    video_path: Path
    pose_path: Path
    scan_frame_counts: bool


class VideoScanResult(TypedDict):
    """Per-video metadata collected by :func:`scan_video_metadata`.

    When the pose file cannot be scanned, ``error`` holds the reason and the
    other pose fields are zero or empty.
    """

    video: str
    hdf5_frame_count: int
    video_frame_count: int | None
    identity_count: int
    static_objects: list[str]
    lixit_keypoints: int
    has_cm_per_pixel: bool
    capabilities: list[str]
    error: str | None


def scan_video_metadata(job: VideoScanJobSpec) -> VideoScanResult:
    """Collect per-video metadata without reading any keypoint data.

    Reads everything :class:`~jabs.project.video_manager.VideoManager` and
    :class:`~jabs.project.feature_manager.FeatureManager` need at project-load
    time - frame count, identity count, static objects, lixit keypoints,
    distance-unit availability and capabilities - through
    :func:`~jabs.pose_estimation.metadata.scan_pose_metadata`, for unified and
    legacy pose files alike.

    A pose file that cannot be scanned does not raise: the result carries the
    reason in ``error`` so the project can open and mark the video unusable.

    This function is stateless and picklable so it can be dispatched to
    :class:`~jabs.core.utils.process_pool_manager.ProcessPoolManager` workers.

    Args:
        job: Scan job specification, including paths and options.

    Returns:
        Per-video metadata collected from the pose file (and optionally the
        video file when ``scan_frame_counts`` is ``True``).
    """
    video = job["video"]
    try:
        metadata = scan_pose_metadata(job["pose_path"])
    except PoseOpenError as error:
        return VideoScanResult(
            video=video,
            hdf5_frame_count=0,
            video_frame_count=None,
            identity_count=0,
            static_objects=[],
            lixit_keypoints=0,
            has_cm_per_pixel=False,
            capabilities=[],
            error=error.reason,
        )

    video_frame_count: int | None = None
    if job["scan_frame_counts"]:
        video_frame_count = VideoReader.get_nframes_from_file(job["video_path"])

    return VideoScanResult(
        video=video,
        hdf5_frame_count=metadata.frames,
        video_frame_count=video_frame_count,
        identity_count=metadata.num_identities,
        static_objects=list(metadata.static_objects),
        lixit_keypoints=metadata.lixit_keypoints,
        has_cm_per_pixel=metadata.cm_per_pixel is not None,
        capabilities=sorted(c.value for c in metadata.capabilities),
        error=None,
    )


def _load_video_labels(annotations_path: Path, pose_est: "PoseEstimation") -> VideoLabels | None:
    """Load VideoLabels from a JSON file if present; else None.

    Workers are handed a path rather than the project's annotation store, so
    that this runs in a child process with no access to whatever backs that
    store. The job builder hydrates the document before dispatch.
    """
    document = read_document(annotations_path)
    if document is None:
        return None
    return VideoLabels.load(document, pose_est)


def _warn_on_frame_count_mismatch(
    video: str, video_frame_count: int, pose_frame_count: int
) -> None:
    """Log a warning when a video and its pose file disagree on frame count.

    The video/pose frame-count check is deferred from project load. Feature
    extraction and predictions are indexed to pose frames, so a mismatch does
    not corrupt them; this surfaces the discrepancy for headless/batch runs
    where no video is opened interactively.

    Args:
        video: Video filename, used in the log message.
        video_frame_count: Frame count read from the video file.
        pose_frame_count: Frame count from the pose file.
    """
    if video_frame_count != pose_frame_count:
        logger.warning(
            "%s: video frame count (%d) does not match pose file frame count (%d); "
            "features and predictions are pose-indexed and unaffected",
            video,
            video_frame_count,
            pose_frame_count,
        )


def _open_pose_and_labels(
    job: _BaseFeatureLoadJobSpec,
) -> "tuple[PoseEstimation, VideoLabels | None]":
    """Open pose + label resources for a job."""
    pose_est = open_pose_file(job["pose_path"], job["cache_dir"])
    _, video_frame_count = get_fps_and_nframes(str(job["video_path"]))
    _warn_on_frame_count_mismatch(job["video"], video_frame_count, pose_est.num_frames)
    labels_obj = _load_video_labels(job["annotations_path"], pose_est)
    return pose_est, labels_obj


def _extract_identity_features(
    video: str,
    identity: int,
    pose_est: "PoseEstimation",
    feature_dir: Path,
    behavior_settings: dict,
    cache_format: str,
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

    For every identity in the video, label arrays are loaded from the requested
    behavior track. Frames where the identity does not exist are forced to
    ``TrackLabels.Label.NONE``, and only ``NONE`` frames are dropped before
    feature extraction - both ``BEHAVIOR`` and ``NOT_BEHAVIOR`` frames for the
    requested behavior are retained as training rows. Identities with no
    labeled frames after that filter are skipped entirely.

    This function is stateless and picklable so it can be dispatched to
    process-pool workers from :class:`~jabs.project.Project`.

    Args:
        job: Video and feature-extraction settings.

    Returns:
        Collected per-frame/window features, label arrays, and per-identity
        group keys for the video.
    """
    pose_est, labels_obj = _open_pose_and_labels(job)
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
        labels = labels_obj.get_track_labels(
            str(identity), behavior_name
        ).labels_masked_to_identity(pose_est.identity_mask(identity))

        if (labels != TrackLabels.Label.NONE).sum() == 0:
            continue

        per_frame_df, window_df = _extract_identity_features(
            video, identity, pose_est, feature_dir, behavior_settings, cache_format, labels
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
    pose_est, labels_obj = _open_pose_and_labels(job)
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
            behavior_labels = labels_obj.get_track_labels(
                str(identity), behavior_key
            ).labels_masked_to_identity(identity_mask)
            labels_by_behavior[behavior_key] = behavior_labels
            include_mask |= behavior_labels == TrackLabels.Label.BEHAVIOR

        # Include only frames with explicit BEHAVIOR labels in any class.
        labels = np.full(identity_mask.shape, TrackLabels.Label.NONE, dtype=np.int8)
        labels[include_mask] = TrackLabels.Label.BEHAVIOR

        if (labels != TrackLabels.Label.NONE).sum() == 0:
            continue

        per_frame_df, window_df = _extract_identity_features(
            video, identity, pose_est, feature_dir, behavior_settings, cache_format, labels
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
