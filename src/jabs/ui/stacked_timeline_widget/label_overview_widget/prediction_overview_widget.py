import numpy as np

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

    def set_labels(self, labels: np.ndarray, probabilities: np.ndarray):
        """set prediction data to display

        overrides the set_labels method of LabelOverviewWidget to set predictions instead of manual labels.

        Args:
            labels (np.ndarray): Array of predicted labels.
            probabilities (np.ndarray): Array of prediction probabilities corresponding to the labels.
        """
        self._label_widget.set_labels(labels, probabilities)
        self._timeline_widget.set_labels(labels)
        self.update_labels()

    def reset(self):
        """Reset the widget to its initial state."""
        self._timeline_widget.reset()
        self._label_widget.set_labels(None, None)
        self._num_frames = 0
