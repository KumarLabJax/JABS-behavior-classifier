"""A no-window feature set (embedding_only) must not collapse the window frame to 0 rows.

``get_labeled_features`` requires labels/per_frame/window/groups to share a row count and
combines per_frame + window column-wise. Embedding-only emits no window features, so the
per-identity window frame must stay row-aligned to per_frame (N rows, 0 columns) instead of
the (0, 0) an empty dict produces.
"""

import shutil
from pathlib import Path

import numpy as np

import jabs.pose_estimation as pose_est
from jabs.project.parallel_workers import _extract_identity_features
from jabs.project.track_labels import TrackLabels

_SAMPLE = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"


def test_embedding_only_window_df_is_row_aligned_zero_columns(tmp_path, sidecar_writer):
    """embedding_only -> window frame has per_frame's row count and zero columns."""
    pose_path = tmp_path / "sample_pose_est_v6.h5"
    shutil.copy(_SAMPLE, pose_path)
    poses = pose_est.open_pose_file(pose_path)
    ident = poses.identities[0]
    n = poses.num_frames
    sidecar_writer(
        tmp_path / "sample.vjepa.h5",
        num_frames=n,
        embed_dim=4,
        identities={
            i: (np.zeros((n, 4), np.float32), np.ones(n, np.uint8)) for i in poses.identities
        },
    )
    labels = np.full(n, int(TrackLabels.Label.BEHAVIOR), dtype=np.int8)
    settings = {"window_size": 5, "embedding": True, "embedding_only": True}

    per_frame_df, window_df = _extract_identity_features(
        "sample.mp4",
        ident,
        poses,
        tmp_path / "features",
        settings,
        "hdf5",
        30.0,
        labels,
    )

    assert per_frame_df.shape[0] > 0
    assert window_df.shape == (per_frame_df.shape[0], 0)  # N rows, 0 cols (not 0 rows)
    assert all(c.startswith("embedding ") for c in per_frame_df.columns)


def test_pose_features_still_have_window_columns(tmp_path, sidecar_writer):
    """Sanity: a normal (pose) feature set still produces non-empty window columns."""
    pose_path = tmp_path / "sample_pose_est_v6.h5"
    shutil.copy(_SAMPLE, pose_path)
    poses = pose_est.open_pose_file(pose_path)
    ident = poses.identities[0]
    n = poses.num_frames
    labels = np.full(n, int(TrackLabels.Label.BEHAVIOR), dtype=np.int8)
    settings = {"window_size": 5}

    per_frame_df, window_df = _extract_identity_features(
        "sample.mp4", ident, poses, tmp_path / "features", settings, "hdf5", 30.0, labels
    )

    assert window_df.shape[0] == per_frame_df.shape[0]
    assert window_df.shape[1] > 0
