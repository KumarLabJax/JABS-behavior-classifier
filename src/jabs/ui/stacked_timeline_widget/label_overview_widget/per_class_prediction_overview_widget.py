"""Overview widget for a single behavior's per-class prediction bar."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from .behavior_probability_widget import BehaviorProbabilityWidget
from .prediction_overview_widget import PredictionOverviewWidget


class PerClassPredictionOverviewWidget(PredictionOverviewWidget):
    """Prediction overview for a single behavior class.

    Identical to :class:`PredictionOverviewWidget` except the detail bar uses
    :class:`BehaviorProbabilityWidget` instead of :class:`PredictedLabelWidget`.
    That means the bar renders the behavior color at ``alpha = probability``
    for every present frame, with no "not this class" gray.
    """

    @classmethod
    def _label_widget_factory(
        cls, parent: QWidget, compact: bool = False
    ) -> BehaviorProbabilityWidget:
        """Return a BehaviorProbabilityWidget as the detail bar."""
        return BehaviorProbabilityWidget(parent, compact=compact)
