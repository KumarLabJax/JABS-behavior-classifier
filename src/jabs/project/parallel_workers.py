import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

import jabs.feature_extraction as fe
from jabs.pose_estimation import get_pose_path, open_pose_file
from jabs.video_reader.utilities import get_fps

from .track_labels import TrackLabels
from .video_labels import VideoLabels


class JobSpec(TypedDict):
    """Specification of a single video feature extraction job."""

    video: str
    video_path: Path
    annotations_path: Path
    feature_dir: Path
    cache_dir: Path | None
    behavior_settings: dict[str, object]
    behavior_name: str | None


class CollectResult(TypedDict):
    """Result of collecting features from a single video."""

    per_frame: list[pd.DataFrame]
    window: list[pd.DataFrame]
    labels: list[np.ndarray]
    group_keys: list[tuple[str, int]]


def _load_video_labels(annotations_path: Path, pose_est) -> VideoLabels | None:
    """Load VideoLabels from a JSON file if present; else None."""
    ap = annotations_path
    if not ap.exists():
        return None
    with ap.open("r") as f:
        data = json.load(f)
    return VideoLabels.load(data, pose_est)


def collect_labeled_features(job: JobSpec) -> CollectResult:
    """Extracts labeled features for a single video.

    This function loads per-frame and window features for a given video. If features
    are not pre-computed then this will result in features being computed directly
    from pose. It is intended to be used in parallel in Project.get_labeled_features().

    Note: this function is a standalone function to facilitate pickling for parallel
    processing via ProcessPoolExecutor. It should not rely on any instance-specific
    state, and is passed all necessary data via the JobSpec argument. A Project instance
    maintains a pool of workers that call this function in parallel from
    Project.load_labeled_features() in order to speed up feature extraction across
    multiple videos.

    Args:
        job (JobSpec): Specification of the video and settings for feature extraction.

    Returns:
        CollectResult: Collected per-frame and window features, labels, and
            identity mapping for the video.
    """
    video: str = job["video"]
    video_path = job["video_path"]
    annotations_path = job["annotations_path"]
    feature_dir = job["feature_dir"]
    cache_dir = job["cache_dir"]
    behavior_settings: dict = job["behavior_settings"]
    behavior_name = job.get("behavior_name")

    # Pose + fps
    pose_est = open_pose_file(get_pose_path(video_path), cache_dir)
    fps = get_fps(str(video_path))

    # Labels (might be None)
    labels_obj = _load_video_labels(annotations_path, pose_est)
    if labels_obj is None:
        return {"per_frame": [], "window": [], "labels": [], "group_keys": []}

    per_frame_list: list[pd.DataFrame] = []
    window_list: list[pd.DataFrame] = []
    labels_list: list[np.ndarray] = []
    group_keys: list[tuple[str, int]] = []

    for identity in pose_est.identities:
        # Extract labels for this (video, identity)
        labels = labels_obj.get_track_labels(str(identity), behavior_name).get_labels()

        # Exclude frames where identity does not exist
        labels = labels.copy()
        labels[pose_est.identity_mask(identity) == 0] = TrackLabels.Label.NONE

        # Skip identities without any BEHAVIOR/NOT_BEHAVIOR labels
        if (
            (labels == TrackLabels.Label.BEHAVIOR) | (labels == TrackLabels.Label.NOT_BEHAVIOR)
        ).sum() == 0:
            continue

        # Feature extraction for this identity
        features = fe.IdentityFeatures(
            video,
            identity,
            feature_dir,
            pose_est,
            fps=fps,
            op_settings=behavior_settings,
        )

        # Per-frame features
        per_frame = features.get_per_frame(labels)
        per_frame = fe.IdentityFeatures.merge_per_frame_features(per_frame)

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
