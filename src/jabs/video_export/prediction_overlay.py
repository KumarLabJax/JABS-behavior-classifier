"""What the exporter needs to draw the JABS prediction overlay on a video.

The player builds the same three things from project state when "View > Label Overlay
> Predictions" is on: a per-identity array of label values, an optional color table
for multi-class projects, and the meaning of the colors. An exported video is watched
away from JABS, so the last of those is burned in as a caption and legend instead of
living in the timeline's legend strip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PySide6 import QtGui

from jabs.overlay_drawing import BACKGROUND_COLOR, BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR

# Shown for the gray marker drawn when an identity has no prediction on a frame,
# which happens wherever the pose is missing.
_NO_PREDICTION_LABEL = "no prediction"


@dataclass(frozen=True)
class PredictionOverlay:
    """Per-identity predictions to draw, with the caption that explains them.

    Build one with :meth:`for_binary` or :meth:`for_multiclass` rather than by hand:
    the caption and legend have to agree with how the label values are colored, and
    those constructors are what keeps them in step.

    Attributes:
        labels: One array of label values per identity, indexed by identity and then
            by frame. Identities and frames the arrays do not cover are drawn without
            a marker.
        color_lut: RGBA table of shape ``(N, 4)`` for multi-class projects, where each
            label value is an index into the table. ``None`` in binary projects, where
            the fixed behavior/not-behavior colors are used instead.
        caption: One line naming what is being shown, drawn in the frame's top-left
            corner.
        legend: ``(name, color)`` pairs drawn as swatches under the caption.
    """

    labels: Sequence[npt.NDArray[np.integer]]
    color_lut: npt.NDArray[np.uint8] | None = None
    caption: str = ""
    legend: Sequence[tuple[str, QtGui.QColor]] = field(default_factory=tuple)

    @classmethod
    def for_binary(
        cls,
        labels: Sequence[npt.NDArray[np.integer]],
        *,
        behavior: str,
        postprocessed: bool,
    ) -> PredictionOverlay:
        """Build the overlay for a binary project's predictions.

        Args:
            labels: Per-identity arrays of ``TrackLabels.Label`` values.
            behavior: Name of the behavior the predictions are for.
            postprocessed: Whether these are post-processed rather than raw
                predictions. Named in the caption either way, since the two can
                differ substantially and the exported video no longer has the menu
                that says which is on screen.

        Returns:
            An overlay carrying the binary color scheme.
        """
        kind = "post-processed" if postprocessed else "raw"
        return cls(
            labels=labels,
            color_lut=None,
            caption=f"{behavior} predictions ({kind})",
            legend=(
                ("behavior", BEHAVIOR_COLOR),
                ("not behavior", NOT_BEHAVIOR_COLOR),
                (_NO_PREDICTION_LABEL, BACKGROUND_COLOR),
            ),
        )

    @classmethod
    def for_multiclass(
        cls,
        labels: Sequence[npt.NDArray[np.integer]],
        *,
        color_lut: npt.NDArray[np.uint8],
        class_names: Sequence[str],
        postprocessed: bool = False,
    ) -> PredictionOverlay:
        """Build the overlay for a multi-class project's predictions.

        Args:
            labels: Per-identity arrays of indices into ``color_lut``.
            color_lut: The project's multi-class RGBA table. Index 0 is the
                no-prediction color; indices 1 onward line up with ``class_names``.
            class_names: Class names in table order, starting at table index 1. This
                is the project's behavior list with the reserved "None" class first,
                the same order the timeline's legend uses.
            postprocessed: Whether these are post-processed rather than raw
                predictions.

        Returns:
            An overlay carrying the project's multi-class colors.

        Raises:
            ValueError: If ``class_names`` does not cover table indices 1 onward,
                which would label a class with another class's color.
        """
        if len(class_names) != len(color_lut) - 1:
            raise ValueError(
                f"class_names must name every color after the no-prediction entry: "
                f"got {len(class_names)} names for a table of {len(color_lut)} colors"
            )

        def color(index: int) -> QtGui.QColor:
            r, g, b, a = color_lut[index]
            return QtGui.QColor(int(r), int(g), int(b), int(a))

        kind = "post-processed" if postprocessed else "raw"
        legend = [(name, color(i + 1)) for i, name in enumerate(class_names)]
        legend.append((_NO_PREDICTION_LABEL, color(0)))
        return cls(
            labels=labels,
            color_lut=color_lut,
            caption=f"Multi-class predictions ({kind})",
            legend=tuple(legend),
        )

    def label_value(self, identity: int, frame_index: int) -> int | None:
        """Return one identity's label value on one frame.

        Args:
            identity: Identity index to look up.
            frame_index: Frame to look up.

        Returns:
            The label value, or ``None`` when this identity or frame is outside the
            label arrays. Predictions can be shorter than the video (or absent for an
            identity), and a frame without a value gets no marker rather than a
            wrong one.
        """
        if identity < 0 or identity >= len(self.labels):
            return None
        values = self.labels[identity]
        if frame_index < 0 or frame_index >= len(values):
            return None
        return int(values[frame_index])
