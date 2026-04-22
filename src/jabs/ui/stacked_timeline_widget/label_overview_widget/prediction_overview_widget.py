from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .label_overview_widget import LabelOverviewWidget
from .predicted_label_widget import PredictedLabelWidget
from .timeline_label_widget import TimelineLabelWidget


class PredictionOverviewWidget(LabelOverviewWidget):
    """Widget that displays an overview of predicted labels for a video.

    Subclass of ``LabelOverviewWidget`` that replaces the detail bar with a
    ``PredictedLabelWidget`` (confidence alpha-blending, read-only) while
    keeping ``TimelineLabelWidget`` for the overview bar.  Overrides
    ``set_labels`` to accept per-frame probabilities instead of an identity mask.
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
        probabilities: npt.NDArray[np.floating],
    ) -> None:
        """Set prediction data to display.

        Overrides :meth:`LabelOverviewWidget.set_labels` to accept per-frame
        probabilities as the second argument instead of an identity mask.
        ``labels`` must be a direct LUT-index array, same contract as the parent.

        Args:
            labels: Class-index array of shape ``(n_frames,)`` with dtype ``int16``.
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
