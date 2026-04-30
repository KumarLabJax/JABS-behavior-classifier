"""Overview widget for a single behavior's per-class prediction bar."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from .behavior_probability_detail_bar import BehaviorProbabilityDetailBar
from .prediction_track_widget import PredictionTrackWidget


class PerClassPredictionTrackWidget(PredictionTrackWidget):
    """Prediction track for a single behavior class.

    Identical to :class:`PredictionTrackWidget` except the detail bar uses
    :class:`BehaviorProbabilityDetailBar` instead of :class:`ClassPredictionDetailBar`.
    That means the bar renders the behavior color at ``alpha = probability``
    for every present frame, with no "not behavior" color for probabilities < 0.5
    """

    @classmethod
    def _label_widget_factory(
        cls, parent: QWidget, compact: bool = False
    ) -> BehaviorProbabilityDetailBar:
        """Return a BehaviorProbabilityDetailBar as the detail bar."""
        return BehaviorProbabilityDetailBar(parent, compact=compact)
