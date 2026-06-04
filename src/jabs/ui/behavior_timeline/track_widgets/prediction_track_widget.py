from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PySide6.QtWidgets import QWidget

from .class_prediction_detail_bar import ClassPredictionDetailBar
from .label_overview_bar import LabelOverviewBar
from .label_track_widget import LabelTrackWidget


class PredictionTrackWidget(LabelTrackWidget):
    """Widget that displays an overview of predicted labels for a video.

    Subclass of ``LabelTrackWidget`` that replaces the detail bar with a
    ``ClassPredictionDetailBar`` (confidence alpha-blending, read-only) while
    keeping ``LabelOverviewBar`` for the overview bar.  Overrides
    ``set_labels`` to accept per-frame probabilities instead of an identity mask.
    """

    @classmethod
    def _timeline_widget_factory(cls, parent):
        """Return a LabelOverviewBar for the timeline slot."""
        return LabelOverviewBar(parent)

    @classmethod
    def _label_widget_factory(
        cls, parent: QWidget, compact: bool = False
    ) -> ClassPredictionDetailBar:
        """Create a ClassPredictionDetailBar as the label widget for this overview."""
        return ClassPredictionDetailBar(parent, compact=compact)

    def set_labels(
        self,
        labels: npt.NDArray[np.int16],
        probabilities: npt.NDArray[np.floating],
    ) -> None:
        """Set prediction data to display.

        Overrides :meth:`LabelTrackWidget.set_labels` to accept per-frame
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
