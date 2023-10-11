import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap, QColor

from src.project import TrackLabels
from .timeline_label_widget import TimelineLabelWidget


class GlobalInferenceWidget(TimelineLabelWidget):
    """
    subclass of TimelineLabeWidget with one primary modification:
    self._labels will be a numpy array and not a TrackLabels object
    """

    def __init__(self):
        super().__init__()

    def _update_bar(self):
        """
        Updates the bar pixmap. Downsamples with the current size and updates
        self._pixmap
        """
        width = self.size().width()
        height = self.size().height()
        self._pixmap = QPixmap(width, height)
        self._pixmap.fill(Qt.transparent)

        if self._labels is None:
            return

        downsampled = TrackLabels.downsample(self._labels, width)

        # draw the bar, each pixel along the width corresponds to a value in the
        # down sampled label array
        qp = QPainter(self._pixmap)
        for x in range(width):
            if downsampled[x] == TrackLabels.Label.NONE.value:
                qp.setPen(QColor(212, 212, 212))
            elif downsampled[x] == TrackLabels.Label.BEHAVIOR.value:
                qp.setPen(self._BEHAVIOR_COLOR)
            elif downsampled[x] == TrackLabels.Label.NOT_BEHAVIOR.value:
                qp.setPen(self._NOT_BEHAVIOR_COLOR)
            elif downsampled[x] == TrackLabels.Label.MIX.value:
                # bin contains mix of behavior/not behavior labels
                qp.setPen(self._MIX_COLOR)
            else:
                continue

            # draw a vertical bar of pixels
            for y in range(self._bar_padding,
                           self._bar_padding + self._bar_height):
                qp.drawPoint(x, y)
        qp.end()

    def set_num_frames(self, num_frames):
        """
        sets the number of frames in the current video, and resets the display
        with a blank track
        """
        self._labels = np.full(num_frames, TrackLabels.Label.NONE.value,
                               dtype=np.byte)
        super().set_num_frames(num_frames)
