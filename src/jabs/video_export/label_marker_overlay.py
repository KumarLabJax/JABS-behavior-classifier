"""What the exporter needs to draw the JABS label markers on a video.

The player builds the same things from project state when "View > Label Overlay" is
on: a per-identity array of label values for the manual labels, another for the
predictions, an optional color table for multi-class projects, and the meaning of the
colors. An exported video is watched away from JABS, so the last of those is burned in
as a caption and legend instead of living in the timeline's legend strip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PySide6 import QtGui

from jabs.overlay_drawing import BACKGROUND_COLOR, BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR

# Shown for the gray marker drawn where an identity has no value on a frame, which
# happens wherever a frame is unlabeled or the pose is missing. Named for whichever
# markers are being drawn, since gray means "no label" and "no prediction" alike.
_NO_LABEL_LABEL = "no label"
_NO_PREDICTION_LABEL = "no prediction"
_NO_VALUE_LABEL = "no label or prediction"


@dataclass(frozen=True)
class LabelMarkerOverlay:
    """Per-identity labels and predictions to draw, with the caption explaining them.

    Build one with :meth:`for_binary` or :meth:`for_multiclass` rather than by hand:
    the caption and legend have to agree with which markers are drawn and how their
    label values are colored, and those constructors are what keeps them in step.

    A marker is drawn for each source that is set, in the order they are declared here:
    the manual label first and the prediction second, the same order the player draws
    them in.

    Attributes:
        manual_labels: One array of manual label values per identity, indexed by
            identity and then by frame, or ``None`` to draw no manual label marker.
            Identities and frames the arrays do not cover are drawn without a marker.
        predicted_labels: The same, for the predictions, or ``None`` to draw no
            prediction marker.
        color_lut: RGBA table of shape ``(N, 4)`` for multi-class projects, where each
            label value is an index into the table. ``None`` in binary projects, where
            the fixed behavior/not-behavior colors are used instead. Both sources are
            colored from it, since both carry the same kind of value.
        caption: One line naming what is being shown, drawn in the frame's top-left
            corner.
        legend: ``(name, color)`` pairs drawn as swatches under the caption.
    """

    manual_labels: Sequence[npt.NDArray[np.integer]] | None = None
    predicted_labels: Sequence[npt.NDArray[np.integer]] | None = None
    color_lut: npt.NDArray[np.uint8] | None = None
    caption: str = ""
    legend: Sequence[tuple[str, QtGui.QColor]] = field(default_factory=tuple)

    @classmethod
    def for_binary(
        cls,
        *,
        behavior: str,
        manual_labels: Sequence[npt.NDArray[np.integer]] | None = None,
        predicted_labels: Sequence[npt.NDArray[np.integer]] | None = None,
        postprocessed: bool = False,
    ) -> LabelMarkerOverlay:
        """Build the overlay for a binary project's labels, predictions, or both.

        Args:
            behavior: Name of the behavior the markers are for.
            manual_labels: Per-identity arrays of ``TrackLabels.Label`` values, or
                ``None`` to draw no manual label marker.
            predicted_labels: Per-identity arrays of ``TrackLabels.Label`` values, or
                ``None`` to draw no prediction marker.
            postprocessed: Whether the predictions are post-processed rather than raw.
                Named in the caption either way, since the two can differ substantially
                and the exported video no longer has the menu that says which is on
                screen. Ignored when no predictions are drawn.

        Returns:
            An overlay carrying the binary color scheme.

        Raises:
            ValueError: If neither source was given, which would burn in a caption for
                markers that are never drawn.
        """
        return cls(
            manual_labels=manual_labels,
            predicted_labels=predicted_labels,
            color_lut=None,
            caption=_caption(
                subject=behavior,
                labels=_draws_a_marker(manual_labels),
                predictions=_draws_a_marker(predicted_labels),
                postprocessed=postprocessed,
            ),
            legend=(
                ("behavior", BEHAVIOR_COLOR),
                ("not behavior", NOT_BEHAVIOR_COLOR),
                (
                    _no_value_label(
                        labels=_draws_a_marker(manual_labels),
                        predictions=_draws_a_marker(predicted_labels),
                    ),
                    BACKGROUND_COLOR,
                ),
            ),
        )

    @classmethod
    def for_multiclass(
        cls,
        *,
        color_lut: npt.NDArray[np.uint8],
        class_names: Sequence[str],
        manual_labels: Sequence[npt.NDArray[np.integer]] | None = None,
        predicted_labels: Sequence[npt.NDArray[np.integer]] | None = None,
        postprocessed: bool = False,
    ) -> LabelMarkerOverlay:
        """Build the overlay for a multi-class project's labels, predictions, or both.

        Args:
            color_lut: The project's multi-class RGBA table. Index 0 is the
                no-value color; indices 1 onward line up with ``class_names``.
            class_names: Class names in table order, starting at table index 1. This
                is the project's behavior list with the reserved "None" class first,
                the same order the timeline's legend uses.
            manual_labels: Per-identity arrays of indices into ``color_lut``, or
                ``None`` to draw no manual label marker.
            predicted_labels: Per-identity arrays of indices into ``color_lut``, or
                ``None`` to draw no prediction marker.
            postprocessed: Whether the predictions are post-processed rather than raw.
                Ignored when no predictions are drawn.

        Returns:
            An overlay carrying the project's multi-class colors.

        Raises:
            ValueError: If neither source was given, or if ``class_names`` does not
                cover table indices 1 onward, which would label a class with another
                class's color.
        """
        if len(class_names) != len(color_lut) - 1:
            raise ValueError(
                f"class_names must name every color after the no-prediction entry: "
                f"got {len(class_names)} names for a table of {len(color_lut)} colors"
            )

        def color(index: int) -> QtGui.QColor:
            r, g, b, a = color_lut[index]
            return QtGui.QColor(int(r), int(g), int(b), int(a))

        legend = [(name, color(i + 1)) for i, name in enumerate(class_names)]
        legend.append(
            (
                _no_value_label(
                    labels=_draws_a_marker(manual_labels),
                    predictions=_draws_a_marker(predicted_labels),
                ),
                color(0),
            )
        )
        return cls(
            manual_labels=manual_labels,
            predicted_labels=predicted_labels,
            color_lut=color_lut,
            caption=_caption(
                subject="Multi-class",
                labels=_draws_a_marker(manual_labels),
                predictions=_draws_a_marker(predicted_labels),
                postprocessed=postprocessed,
            ),
            legend=tuple(legend),
        )

    def __post_init__(self) -> None:
        """Reject an overlay with no markers to draw.

        Raises:
            ValueError: If neither label source carries values, which would burn a
                caption and legend into every frame for markers that never appear.
        """
        if not self.sources:
            raise ValueError("a label marker overlay needs manual labels, predictions, or both")

    @property
    def sources(self) -> tuple[Sequence[npt.NDArray[np.integer]], ...]:
        """The label arrays to draw a marker for, in the order they are drawn."""
        return tuple(
            source
            for source in (self.manual_labels, self.predicted_labels)
            if _draws_a_marker(source)
        )

    @property
    def marker_count(self) -> int:
        """How many markers are drawn beside each identity: one per source, so 1 or 2."""
        return len(self.sources)

    def marker_values(self, identity: int, frame_index: int) -> tuple[int | None, ...]:
        """Return one identity's label value from each source on one frame.

        Args:
            identity: Identity index to look up.
            frame_index: Frame to look up.

        Returns:
            One entry per source, in drawing order, holding that source's label value
            or ``None`` when this identity or frame is outside its arrays. Labels and
            predictions can be shorter than the video (or absent for an identity), and
            a frame without a value gets no marker rather than a wrong one.
        """
        return tuple(_value(source, identity, frame_index) for source in self.sources)


def _draws_a_marker(source: Sequence[npt.NDArray[np.integer]] | None) -> bool:
    """Whether a label source will put a marker on the frame.

    A source with no identities in it draws nothing, so it must not count towards the
    caption or the legend either: naming a marker that never appears is what the
    constructors exist to prevent.
    """
    return source is not None and len(source) > 0


def _value(
    source: Sequence[npt.NDArray[np.integer]], identity: int, frame_index: int
) -> int | None:
    """Return one label value from one source, or None if it is out of range."""
    if identity < 0 or identity >= len(source):
        return None
    values = source[identity]
    if frame_index < 0 or frame_index >= len(values):
        return None
    return int(values[frame_index])


def _caption(*, subject: str, labels: bool, predictions: bool, postprocessed: bool) -> str:
    """Build the caption naming the markers being drawn.

    With both sources the caption also says which marker is which, since the only
    thing that distinguishes them on the frame is their order.
    """
    kind = "post-processed" if postprocessed else "raw"
    if labels and predictions:
        return f"{subject} labels (left) and predictions ({kind}, right)"
    if labels:
        return f"{subject} labels"
    return f"{subject} predictions ({kind})"


def _no_value_label(*, labels: bool, predictions: bool) -> str:
    """Name the legend entry for the color drawn where a source has no value."""
    if labels and predictions:
        return _NO_VALUE_LABEL
    return _NO_LABEL_LABEL if labels else _NO_PREDICTION_LABEL
