from pathlib import Path

import numpy as np

from jabs.feature_extraction.embedding_features.sidecar import EmbeddingInfo

FIXTURE = Path(__file__).parent.parent.parent / "data" / "sidecar_format_v1.vjepa.h5"


def test_reads_frozen_format_v1_fixture():
    """The reader accepts the exact sidecar fixture produced by the pbs-vjepa generator."""
    info = EmbeddingInfo(FIXTURE, identity=0)
    assert info.embed_dim == 4
    assert info.num_frames == 3
    assert info.column_names == ["emb_0000", "emb_0001", "emb_0002", "emb_0003"]
    # coverage [1, 1, 0] -> frame 2 is NaN, frames 0-1 finite
    assert np.all(np.isnan(info.frame_embeddings[2]))
    assert np.all(np.isfinite(info.frame_embeddings[:2]))
