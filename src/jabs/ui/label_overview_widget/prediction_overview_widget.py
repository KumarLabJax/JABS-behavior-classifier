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

    def _timeline_widget_factory(self, parent):
        return TimelinePredictionWidget(parent)

    def _label_widget_factory(self, parent):
        return PredictedLabelWidget(parent)

    def set_predictions(self, predictions, probabilities):
        """set prediction data to display"""
        self._label_widget.set_predictions(predictions, probabilities)
        if predictions is not None:
            self._timeline_widget.set_labels(predictions)
        else:
            self._timeline_widget.set_labels(
                np.full(
                    self.num_frames,
                    TrackLabels.Label.NONE.value,
                    dtype=np.byte,
                )
            )
        self.update_labels()

    def set_labels(self, labels: TrackLabels, mask: np.ndarray | None = None):
        """this widget does not support setting labels from a TrackLabels object"""
        raise NotImplementedError
