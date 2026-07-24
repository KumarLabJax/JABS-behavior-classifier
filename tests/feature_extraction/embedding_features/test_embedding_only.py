"""The `embedding_only` op-setting: train on embeddings alone (pose groups filtered out).

This is the deployment-relevant condition (embeddings REPLACE pose). It is a read-time
filter, so pose features are still computed/cached but excluded from the returned feature
set, leaving only the `embedding` columns.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation as pose_est
from jabs.feature_extraction.features import IdentityFeatures

_SAMPLE = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"


def _setup(tmp_path, sidecar_writer, *, with_sidecar=True, embed_dim=4):
    pose_path = tmp_path / "sample_pose_est_v6.h5"
    shutil.copy(_SAMPLE, pose_path)
    poses = pose_est.open_pose_file(pose_path)
    if with_sidecar:
        n = poses.num_frames
        sidecar_writer(
            tmp_path / "sample.vjepa.h5",
            num_frames=n,
            embed_dim=embed_dim,
            identities={
                i: (np.zeros((n, embed_dim), np.float32), np.ones(n, np.uint8))
                for i in poses.identities
            },
        )
    return poses, poses.identities[0]


def _prefixes(feats: IdentityFeatures) -> set[str]:
    return {k.split(" ", 1)[0] for k in feats.get_per_frame_flat()}


def test_embedding_only_keeps_only_embedding_columns(tmp_path, sidecar_writer):
    """With embedding_only, every per-frame column belongs to the embedding module."""
    poses, ident = _setup(tmp_path, sidecar_writer)
    feats = IdentityFeatures(
        "sample.mp4",
        ident,
        None,
        poses,
        op_settings={"embedding": True, "embedding_only": True},
    )
    prefixes = _prefixes(feats)
    assert prefixes == {"embedding"}, prefixes


def test_hybrid_keeps_pose_and_embedding(tmp_path, sidecar_writer):
    """Without embedding_only, embeddings are additive: pose columns remain present."""
    poses, ident = _setup(tmp_path, sidecar_writer)
    feats = IdentityFeatures(
        "sample.mp4",
        ident,
        None,
        poses,
        op_settings={"embedding": True, "embedding_only": False},
    )
    prefixes = _prefixes(feats)
    assert "embedding" in prefixes
    assert "pairwise_distances" in prefixes


def test_embedding_only_yields_no_pose_window_features(tmp_path, sidecar_writer):
    """Embeddings emit no window features, and pose window features are filtered out."""
    poses, ident = _setup(tmp_path, sidecar_writer)
    feats = IdentityFeatures(
        "sample.mp4",
        ident,
        None,
        poses,
        op_settings={"embedding": True, "embedding_only": True},
    )
    window = feats.get_features(window_size=2)["window"]
    assert window == {}


def test_embedding_only_without_active_embedding_raises(tmp_path, sidecar_writer):
    """embedding_only with no sidecar (embedding inactive) fails loudly, not to empty features."""
    poses, ident = _setup(tmp_path, sidecar_writer, with_sidecar=False)
    with pytest.raises(ValueError):
        IdentityFeatures(
            "sample.mp4",
            ident,
            None,
            poses,
            op_settings={"embedding": True, "embedding_only": True},
        )
