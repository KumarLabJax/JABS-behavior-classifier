import numpy as np

from jabs.feature_extraction.embedding_features.embedding import EmbeddingFeature
from jabs.feature_extraction.embedding_features.sidecar import EmbeddingInfo


def test_per_frame_emits_one_column_per_dim(sidecar_factory):
    """per_frame returns one length-num_frames array per embedding dimension."""
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    cov = np.array([1, 1, 1, 1], np.uint8)
    path = sidecar_factory(num_frames=4, embed_dim=3, identities={0: (emb, cov)})
    info = EmbeddingInfo(path, identity=0)

    feat = EmbeddingFeature(poses=None, pixel_scale=1.0, embedding_info=info)
    out = feat.per_frame(0)

    assert set(out.keys()) == {"emb_0000", "emb_0001", "emb_0002"}
    assert out["emb_0000"].shape == (4,)
    assert np.allclose(out["emb_0000"], emb[:, 0])
    assert np.allclose(out["emb_0002"], emb[:, 2])


def test_window_returns_empty(sidecar_factory):
    """Embedding features emit no window features."""
    path = sidecar_factory(
        num_frames=2,
        embed_dim=2,
        identities={0: (np.zeros((2, 2), np.float32), np.ones(2, np.uint8))},
    )
    info = EmbeddingInfo(path, identity=0)
    feat = EmbeddingFeature(poses=None, pixel_scale=1.0, embedding_info=info)
    assert feat.window(0, 5, feat.per_frame(0)) == {}


def test_feature_names_match_columns(sidecar_factory):
    """feature_names reports the per-instance embedding column names."""
    path = sidecar_factory(
        num_frames=1,
        embed_dim=2,
        identities={0: (np.zeros((1, 2), np.float32), np.ones(1, np.uint8))},
    )
    info = EmbeddingInfo(path, identity=0)
    feat = EmbeddingFeature(poses=None, pixel_scale=1.0, embedding_info=info)
    assert feat.feature_names() == ["emb_0000", "emb_0001"]
