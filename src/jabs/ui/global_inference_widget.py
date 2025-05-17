import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap, QImage

from jabs.project import TrackLabels
from .timeline_label_widget import TimelineLabelWidget


class GlobalInferenceWidget(TimelineLabelWidget):
    """subclass of TimelineLabeWidget with one primary modification:
    self._labels will be a numpy array and not a TrackLabels object
    """

    def __init__(self):
        super().__init__()

    def _update_bar(self):
        """Updates the bar pixmap. Downsamples with the current size and updates
        self._pixmap
        """
        width = self.size().width()
        height = self.size().height()
        self._pixmap = QPixmap(width, height)
        self._pixmap.fill(Qt.transparent)

        if self._labels is None:
            return

        downsampled = TrackLabels.downsample(self._labels, width)

        colors = self.color_lut[downsampled + 1] # shape (width, 4)
        colors = np.repeat(colors[np.newaxis, :, :], self._bar_height, axis=0)  # shape (bar_height, width, 4)
        img = QImage(colors.data, colors.shape[1], colors.shape[0], QImage.Format_RGBA8888)
        painter = QPainter(self._pixmap)
        painter.drawImage(0, self._bar_padding, img)
        painter.end()

    def set_num_frames(self, num_frames):
        """sets the number of frames in the current video, and resets the display
        with a blank track
        """
        self._labels = np.full(num_frames, TrackLabels.Label.NONE.value,
                               dtype=np.byte)
        super().set_num_frames(num_frames)
