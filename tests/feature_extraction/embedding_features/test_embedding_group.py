import shutil
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation as pose_est
from jabs.feature_extraction.embedding_features import EmbeddingFeatureGroup
from jabs.feature_extraction.embedding_features.sidecar import EmbeddingSidecarError

_SAMPLE = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"


def _pose_with_sidecar(tmp_path, sidecar_writer, embed_dim, *, frame_delta=0):
    """Copy the sample pose into tmp_path and write a matching (or mismatched) sidecar."""
    pose_path = tmp_path / "sample_pose_est_v6.h5"
    shutil.copy(_SAMPLE, pose_path)
    poses = pose_est.open_pose_file(pose_path)
    n = poses.num_frames + frame_delta
    ident = poses.identities[0]
    emb = np.random.default_rng(0).normal(size=(n, embed_dim)).astype(np.float32)
    cov = np.ones(n, np.uint8)
    sidecar_writer(
        tmp_path / "sample.vjepa.h5",
        num_frames=n,
        embed_dim=embed_dim,
        identities={ident: (emb, cov)},
    )
    return poses, ident


def test_group_per_frame_returns_columns_len_num_frames(tmp_path, sidecar_writer):
    """The group yields one embedding module whose columns are length num_frames."""
    poses, ident = _pose_with_sidecar(tmp_path, sidecar_writer, embed_dim=5)
    group = EmbeddingFeatureGroup(poses, 1.0)
    out = group.per_frame(ident)  # {"embedding": {"emb_0000": arr, ...}}
    assert set(out.keys()) == {"embedding"}
    cols = out["embedding"]
    assert len(cols) == 5
    assert cols["emb_0000"].shape == (poses.num_frames,)


def test_group_window_is_empty(tmp_path, sidecar_writer):
    """The group emits no window features."""
    poses, ident = _pose_with_sidecar(tmp_path, sidecar_writer, embed_dim=3)
    group = EmbeddingFeatureGroup(poses, 1.0)
    pf = group.per_frame(ident)
    win = group.window(ident, 5, pf)
    assert win == {"embedding": {}}


def test_group_raises_on_frame_count_mismatch(tmp_path, sidecar_writer):
    """A sidecar whose frame count differs from the pose file raises loudly."""
    poses, ident = _pose_with_sidecar(tmp_path, sidecar_writer, embed_dim=2, frame_delta=5)
    group = EmbeddingFeatureGroup(poses, 1.0)
    with pytest.raises(EmbeddingSidecarError):
        group.per_frame(ident)
