import shutil
from pathlib import Path

import numpy as np

import jabs.pose_estimation as pose_est
from jabs.feature_extraction.features import IdentityFeatures

_SAMPLE = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"


def _setup(tmp_path, sidecar_writer, *, with_sidecar: bool, embed_dim: int = 4):
    """Copy the sample pose into tmp_path, optionally writing a matching sidecar."""
    pose_path = tmp_path / "sample_pose_est_v6.h5"
    shutil.copy(_SAMPLE, pose_path)
    poses = pose_est.open_pose_file(pose_path)
    ident = poses.identities[0]
    if with_sidecar:
        n = poses.num_frames
        sidecar_writer(
            tmp_path / "sample.vjepa.h5",
            num_frames=n,
            embed_dim=embed_dim,
            identities={ident: (np.zeros((n, embed_dim), np.float32), np.ones(n, np.uint8))},
        )
    return poses, ident


def _has_embedding_columns(feats: IdentityFeatures) -> bool:
    flat = feats.get_per_frame_flat()
    return any(k.startswith("embedding ") for k in flat)


def test_enabled_and_sidecar_present_activates(tmp_path, sidecar_writer):
    """Embedding columns appear only when the setting is on and a sidecar exists."""
    poses, ident = _setup(tmp_path, sidecar_writer, with_sidecar=True)
    feats = IdentityFeatures("sample.mp4", ident, None, poses, op_settings={"embedding": True})
    assert _has_embedding_columns(feats)


def test_enabled_but_no_sidecar_is_inert(tmp_path, sidecar_writer):
    """With the setting on but no sidecar, the group contributes nothing."""
    poses, ident = _setup(tmp_path, sidecar_writer, with_sidecar=False)
    feats = IdentityFeatures("sample.mp4", ident, None, poses, op_settings={"embedding": True})
    assert not _has_embedding_columns(feats)


def test_sidecar_present_but_disabled_is_inert(tmp_path, sidecar_writer):
    """With a sidecar present but the setting off, the group contributes nothing."""
    poses, ident = _setup(tmp_path, sidecar_writer, with_sidecar=True)
    feats = IdentityFeatures("sample.mp4", ident, None, poses, op_settings={"embedding": False})
    assert not _has_embedding_columns(feats)


def test_default_settings_do_not_activate(tmp_path, sidecar_writer):
    """Default op_settings (None) never activate embeddings, even with a sidecar."""
    poses, ident = _setup(tmp_path, sidecar_writer, with_sidecar=True)
    feats = IdentityFeatures("sample.mp4", ident, None, poses, op_settings=None)
    assert not _has_embedding_columns(feats)
