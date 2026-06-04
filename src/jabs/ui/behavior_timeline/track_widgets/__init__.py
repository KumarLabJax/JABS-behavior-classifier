"""Widgets for individual identity tracks within BehaviorTimelineWidget.

This package includes:
- LabelTrackWidget: One identity's label track (labeling mode).
- PredictionTrackWidget: One identity's prediction track (read-only, confidence alpha).
"""

from .label_track_widget import LabelTrackWidget
from .prediction_track_widget import PredictionTrackWidget

__all__ = [
    "LabelTrackWidget",
    "PredictionTrackWidget",
]
