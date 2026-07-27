"""Feature exposing precomputed V-JEPA embeddings as per-frame columns.

One column per embedding dimension. Window operations are opt-in and, when enabled,
emit only a rolling ``std_dev`` per dimension at configurable radii (no FFT/other
moments): each embedding already integrates temporal context over its source clip,
so the FFT/full-moment suite would be redundant and would explode the feature count,
while a coarse rolling std at radii beyond the clip receptive field exposes
frame-to-frame dynamics the per-frame classifier cannot otherwise see.
"""

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.feature_extraction.window_operations.window_stats import window_std_dev

from .sidecar import EmbeddingInfo


class EmbeddingFeature(Feature):
    """Per-frame V-JEPA embedding columns for one identity, from a sidecar."""

    _name = "embedding"
    _min_pose = 2

    def __init__(
        self,
        poses,
        pixel_scale: float,
        embedding_info: EmbeddingInfo,
        window_sizes: tuple[int, ...] = (),
    ) -> None:
        # Intentionally does not call Feature.__init__: the base only reads
        # poses.fps (for signal-band setup this feature never uses). This feature
        # overrides per_frame, window, and feature_names -- the only behavior the
        # group invokes -- so it needs nothing from the base initializer.
        self._embedding_info = embedding_info
        self._feature_names = list(embedding_info.column_names)
        # Rolling-std radii, decoupled from the behavior window_size. Empty = off.
        self._window_sizes = tuple(int(w) for w in window_sizes)

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
        """Emit rolling std_dev per embedding dim at each configured radius.

        The ``window_size`` argument (the behavior's pose window) is intentionally
        ignored: embedding dynamics are probed at their own radii, which must exceed
        the ~16-frame clip receptive field to add information (see the design spec).
        Only ``std_dev`` is emitted (no FFT/other moments); ``window_std_dev`` uses a
        masked ``nanstd`` so NaN-filled uncovered frames are excluded. Returns ``{}``
        when no radii are configured, preserving the prior behavior.
        """
        if not self._window_sizes:
            return {}
        return {
            f"std_dev_w{r}": self._compute_window_feature(
                per_frame_features, None, r, window_std_dev
            )
            for r in self._window_sizes
        }
