import numpy as np

from jabs.feature_extraction.embedding_features.embedding import EmbeddingFeature
from jabs.feature_extraction.embedding_features.sidecar import EmbeddingInfo
from jabs.feature_extraction.window_operations.window_stats import window_std_dev


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


def test_window_emits_std_dev_at_configured_radius(sidecar_factory):
    """A single configured radius yields one std_dev_w<r> op with a column per dim."""
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)  # num_frames=4, embed_dim=3
    path = sidecar_factory(num_frames=4, embed_dim=3, identities={0: (emb, np.ones(4, np.uint8))})
    info = EmbeddingInfo(path, identity=0)
    feat = EmbeddingFeature(poses=None, pixel_scale=1.0, embedding_info=info, window_sizes=(1,))

    out = feat.window(0, 5, feat.per_frame(0))

    assert set(out.keys()) == {"std_dev_w1"}
    assert set(out["std_dev_w1"].keys()) == {"emb_0000", "emb_0001", "emb_0002"}
    assert out["std_dev_w1"]["emb_0000"].shape == (4,)
    # emb_0000 == [0,3,6,9]; radius-1 interior window at frame 1 = std([0,3,6]) = sqrt(6)
    assert np.isclose(out["std_dev_w1"]["emb_0000"][1], np.sqrt(6.0))
    assert np.allclose(out["std_dev_w1"]["emb_0000"], window_std_dev(emb[:, 0], window=1))


def test_window_multi_radius_and_nan_coverage(sidecar_factory):
    """Multiple radii each produce an op key; uncovered (NaN) frames are excluded by nanstd."""
    emb = np.array([[0.0], [10.0], [20.0], [30.0]], np.float32)  # embed_dim=1
    cov = np.array([1, 1, 0, 1], np.uint8)  # frame 2 uncovered -> NaN in per-frame trace
    path = sidecar_factory(num_frames=4, embed_dim=1, identities={0: (emb, cov)})
    info = EmbeddingInfo(path, identity=0)
    feat = EmbeddingFeature(poses=None, pixel_scale=1.0, embedding_info=info, window_sizes=(1, 2))

    out = feat.window(0, 5, feat.per_frame(0))

    assert set(out.keys()) == {"std_dev_w1", "std_dev_w2"}
    # frame 1, radius 1: window frames [0,1,2] = [0,10,NaN] -> nanstd([0,10]) = 5.0
    assert np.isclose(out["std_dev_w1"]["emb_0000"][1], 5.0)
