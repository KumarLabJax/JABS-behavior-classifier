from pathlib import Path

import numpy as np
import pytest

from jabs.feature_extraction.embedding_features.sidecar import (
    EmbeddingInfo,
    EmbeddingSidecarError,
    sidecar_exists,
    sidecar_path_for_pose,
)


def test_sidecar_path_strips_pose_suffix():
    """Sidecar path drops the ``_pose_est_vN`` suffix and uses the ``.vjepa.h5`` name."""
    p = sidecar_path_for_pose(Path("/proj/video1_pose_est_v6.h5"))
    assert p == Path("/proj/video1.vjepa.h5")


def test_sidecar_exists(tmp_path, sidecar_factory):
    """``sidecar_exists`` reflects presence of the sibling sidecar file."""
    pose = tmp_path / "video_pose_est_v6.h5"
    assert not sidecar_exists(pose)
    sidecar_factory(
        name="video.vjepa.h5",
        num_frames=3,
        embed_dim=2,
        identities={0: (np.zeros((3, 2), np.float32), np.ones(3, np.uint8))},
    )
    assert sidecar_exists(pose)


def test_embedding_info_reads_columns_and_marks_uncovered_nan(sidecar_factory):
    """Reader exposes D columns and NaN-fills rows where coverage is 0."""
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)  # 4 frames, D=3
    cov = np.array([1, 0, 1, 1], np.uint8)
    path = sidecar_factory(num_frames=4, embed_dim=3, identities={2: (emb, cov)})
    info = EmbeddingInfo(path, identity=2)
    assert info.embed_dim == 3
    assert info.num_frames == 4
    assert info.column_names == ["emb_0000", "emb_0001", "emb_0002"]
    assert info.frame_embeddings.shape == (4, 3)
    # uncovered frame 1 -> NaN row
    assert np.all(np.isnan(info.frame_embeddings[1]))
    # covered rows preserved
    assert np.allclose(info.frame_embeddings[0], emb[0])
    assert np.allclose(info.frame_embeddings[3], emb[3])


def test_embedding_info_missing_identity_raises(sidecar_factory):
    """A missing identity id raises ``EmbeddingSidecarError``."""
    path = sidecar_factory(
        num_frames=2,
        embed_dim=2,
        identities={0: (np.zeros((2, 2), np.float32), np.ones(2, np.uint8))},
    )
    with pytest.raises(EmbeddingSidecarError):
        EmbeddingInfo(path, identity=5)


def test_embedding_info_rejects_unsupported_format_version(sidecar_factory):
    """An unsupported ``format_version`` raises ``EmbeddingSidecarError``."""
    path = sidecar_factory(
        num_frames=1,
        embed_dim=1,
        identities={0: (np.zeros((1, 1), np.float32), np.ones(1, np.uint8))},
        format_version=999,
    )
    with pytest.raises(EmbeddingSidecarError):
        EmbeddingInfo(path, identity=0)
