"""The feature cache must reflect sidecar state.

These exercise the REAL feature cache (a directory), unlike the gating tests which
pass ``directory=None``. They pin the two silent-staleness directions: a sidecar
appearing after a cache exists must invalidate it, and an embedding run must not
leak embedding columns into a later pose-only run that shares the cache.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation as pose_est
from jabs.core.enums import CacheFormat
from jabs.feature_extraction.features import IdentityFeatures

_SAMPLE = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"
_EMBED_DIM = 4


def _open_pose(tmp_path):
    pose_path = tmp_path / "sample_pose_est_v6.h5"
    shutil.copy(_SAMPLE, pose_path)
    return pose_est.open_pose_file(pose_path)


def _write_sidecar(tmp_path, poses, sidecar_writer):
    n = poses.num_frames
    sidecar_writer(
        tmp_path / "sample.vjepa.h5",
        num_frames=n,
        embed_dim=_EMBED_DIM,
        identities={
            i: (np.zeros((n, _EMBED_DIM), np.float32), np.ones(n, np.uint8))
            for i in poses.identities
        },
    )


def _emb_cols(feats: IdentityFeatures) -> list[str]:
    return [k for k in feats.get_per_frame_flat() if k.startswith("embedding ")]


@pytest.mark.parametrize("fmt", [CacheFormat.HDF5, CacheFormat.PARQUET])
def test_sidecar_added_after_cache_invalidates(tmp_path, sidecar_writer, fmt):
    """A sidecar appearing after the cache was written must force a recompute."""
    poses = _open_pose(tmp_path)
    ident = poses.identities[0]
    feat_dir = tmp_path / "features"

    # 1. cache with NO sidecar present (embedding enabled but inert)
    f1 = IdentityFeatures(
        "sample.mp4", ident, feat_dir, poses, op_settings={"embedding": True}, cache_format=fmt
    )
    assert _emb_cols(f1) == []

    # 2. sidecar now exists; a fresh instance on the SAME cache dir must see embeddings
    _write_sidecar(tmp_path, poses, sidecar_writer)
    f2 = IdentityFeatures(
        "sample.mp4", ident, feat_dir, poses, op_settings={"embedding": True}, cache_format=fmt
    )
    assert len(_emb_cols(f2)) == _EMBED_DIM


@pytest.mark.parametrize("fmt", [CacheFormat.HDF5, CacheFormat.PARQUET])
def test_embedding_cache_does_not_leak_into_pose_only(tmp_path, sidecar_writer, fmt):
    """After an embedding run, a pose-only run on the same cache must not see emb columns."""
    poses = _open_pose(tmp_path)
    ident = poses.identities[0]
    feat_dir = tmp_path / "features"
    _write_sidecar(tmp_path, poses, sidecar_writer)

    f1 = IdentityFeatures(
        "sample.mp4", ident, feat_dir, poses, op_settings={"embedding": True}, cache_format=fmt
    )
    assert len(_emb_cols(f1)) == _EMBED_DIM

    f2 = IdentityFeatures(
        "sample.mp4", ident, feat_dir, poses, op_settings={"embedding": False}, cache_format=fmt
    )
    assert _emb_cols(f2) == []
