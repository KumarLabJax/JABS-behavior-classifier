import numpy as np

from jabs.project import TrackLabels

from .label_overview_widget import LabelOverviewWidget
from .predicted_label_widget import PredictedLabelWidget
from .timeline_prediction_widget import TimelinePredictionWidget


class PredictionOverviewWidget(LabelOverviewWidget):
    """Widget that displays an overview of predicted labels and global inference results for a video.

    This widget replaces the manual label and timeline widgets of LabelOverviewWidget with
    widgets specialized for visualizing model predictions. It provides methods to set
    prediction data and disables setting manual labels.
    """

    @classmethod
    def _timeline_widget_factory(cls, parent):
        return TimelinePredictionWidget(parent)

    @classmethod
    def _label_widget_factory(cls, parent):
        return PredictedLabelWidget(parent)

    def set_predictions(self, predictions, probabilities):
        """set prediction data to display"""
        self._label_widget.set_predictions(predictions, probabilities)
        self._timeline_widget.set_labels(predictions)
        self.update_labels()

    def set_labels(self, labels: TrackLabels, mask: np.ndarray | None = None):
        """this widget does not support setting labels from a TrackLabels object"""
        raise NotImplementedError

    def reset(self):
        """Reset the widget to its initial state."""
        self._timeline_widget.reset()
        self._label_widget.set_predictions(None, None)
        self._num_frames = 0
