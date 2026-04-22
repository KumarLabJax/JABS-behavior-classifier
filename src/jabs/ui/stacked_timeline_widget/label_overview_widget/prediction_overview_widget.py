from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .label_overview_widget import LabelOverviewWidget
from .predicted_label_widget import PredictedLabelWidget
from .timeline_label_widget import TimelineLabelWidget


class PredictionOverviewWidget(LabelOverviewWidget):
    """Widget that displays an overview of predicted labels and global inference results for a video.

    This widget replaces the manual label and timeline widgets of LabelOverviewWidget with
    widgets specialized for visualizing model predictions. It provides methods to set
    prediction data and disables setting manual labels.
    """

    @classmethod
    def _timeline_widget_factory(cls, parent):
        """Return a TimelineLabelWidget for the timeline slot."""
        return TimelineLabelWidget(parent)

    @classmethod
    def _label_widget_factory(cls, parent):
        """Return a PredictedLabelWidget for the detail bar slot."""
        return PredictedLabelWidget(parent)

    def set_labels(
        self,
        labels: npt.NDArray[np.int16],
        probabilities: npt.NDArray[np.float32],
    ) -> None:
        """Set prediction data to display.

        Overrides :meth:`LabelOverviewWidget.set_labels` to accept pre-normalized
        LUT-index arrays and per-frame probabilities instead of ``TrackLabels``.
        Callers must normalize raw binary predictions via
        :func:`.label_overview_util.binary_predictions_to_lut_indices` before
        calling; multi-class callers pass class-index arrays directly.

        Args:
            labels: Pre-normalized class-index array of shape ``(n_frames,)``.
            probabilities: Per-frame prediction confidence, shape ``(n_frames,)``.
        """
        self._label_widget.set_labels(labels, probabilities)
        self._timeline_widget.set_labels(labels)
        self.update_labels()

    def reset(self) -> None:
        """Reset the widget to its initial state."""
        self._timeline_widget.reset()
        self._label_widget.set_labels(None, None)
        self._num_frames = 0
