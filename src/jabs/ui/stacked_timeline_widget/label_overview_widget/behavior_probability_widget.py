"""Widget that renders a single behavior's probability as a color heatmap."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .predicted_label_widget import PredictedLabelWidget


class BehaviorProbabilityWidget(PredictedLabelWidget):
    """Renders a single behavior's probability as a color heatmap.

    Unlike :class:`PredictedLabelWidget`, which uses a 3-entry LUT to paint a
    "not this class" color for low-probability frames, this widget always uses
    the behavior color (LUT index 2) and controls visibility with alpha alone.
    The result is a white background with the behavior color overlaid at
    ``alpha = probability`` -- nearly transparent for low-confidence frames,
    fully opaque for high-confidence frames.  Frames where the animal is absent
    (LUT index 0) are rendered transparently so the white background shows
    through unchanged.
    """

    def _build_frame_colors(
        self,
        color_indices: npt.NDArray[np.int16],
        probs_slice: npt.NDArray[np.floating] | None,
    ) -> npt.NDArray[np.uint8]:
        """Build per-frame RGBA using behavior color at probability-based alpha.

        Args:
            color_indices: LUT index per in-bounds frame, shape ``(n,)``.
                Index 0 means animal absent; any other index means animal present.
            probs_slice: Clipped probabilities ``[0, 1]`` for the same frames, or ``None``.

        Returns:
            RGBA array of shape ``(n, 4)`` with dtype ``uint8``.
        """
        n = len(color_indices)
        colors = np.zeros((n, 4), dtype=np.uint8)
        present = color_indices != 0
        if not np.any(present):
            return colors

        behavior_rgb = self._color_lut[2, :3]
        colors[present, :3] = behavior_rgb

        if probs_slice is not None:
            alphas = (probs_slice * 255).astype(np.uint8)
            # Keep a minimum alpha for present frames with zero probability so
            # post-processed/interpolated frames remain faintly visible.
            zero_prob_present = present & (alphas == 0)
            alphas[zero_prob_present] = 125
            colors[present, 3] = alphas[present]
        else:
            colors[present, 3] = self._color_lut[2, 3]

        return colors
