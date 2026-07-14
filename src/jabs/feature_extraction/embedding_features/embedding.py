"""Feature exposing precomputed V-JEPA embeddings as per-frame columns.

One column per embedding dimension. Window operations are disabled: each embedding
already integrates temporal context over the source clip, so JABS's rolling
statistics (mean/std/FFT) would be redundant and would explode the feature count.
"""

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature

from .sidecar import EmbeddingInfo


class EmbeddingFeature(Feature):
    """Per-frame V-JEPA embedding columns for one identity, from a sidecar."""

    _name = "embedding"
    _min_pose = 2

    def __init__(self, poses, pixel_scale: float, embedding_info: EmbeddingInfo) -> None:
        # Intentionally does not call Feature.__init__: the base only reads
        # poses.fps (for signal-band setup this feature never uses). This feature
        # overrides per_frame, window, and feature_names -- the only behavior the
        # group invokes -- so it needs nothing from the base initializer.
        self._embedding_info = embedding_info
        self._feature_names = list(embedding_info.column_names)

    def feature_names(self) -> list[str]:
        """Return the per-instance embedding column names.

        Overrides the base ``Feature.feature_names`` classmethod: embedding column
        count is only known per-instance from the sidecar, so a class attribute
        cannot express it. The group only ever calls this on instances.
        """
        return list(self._feature_names)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """Return one length-``num_frames`` array per embedding dimension."""
        emb = self._embedding_info.frame_embeddings  # (num_frames, D), NaN where uncovered
        return {name: emb[:, j] for j, name in enumerate(self._embedding_info.column_names)}

    def window(self, identity: int, window_size: int, per_frame_features: dict) -> dict:
        """Embeddings emit no window features (temporal context already baked in)."""
        return {}
