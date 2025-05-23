import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap

from jabs.project import TrackLabels

from .timeline_label_widget import TimelineLabelWidget


class TimelinePredictionWidget(TimelineLabelWidget):
    """TimelinePredictionWidget

    subclass of TimelineLabelWidget with some modifications:
    self._labels will be a numpy array and not a TrackLabels object
    also uses opacity to indicate the confidence of the label

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_bar(self):
        """Updates the bar pixmap.

        Downsamples with the current size and updates self._pixmap
        """
        if self._labels is None:
            return

        width = self.size().width()
        height = self.size().height()
        self._pixmap = QPixmap(width, height)
        self._pixmap.fill(Qt.GlobalColor.transparent)

        downsampled = TrackLabels.downsample(self._labels, width)

        # use downsampled labels to generate RGBA colors
        # labels are -1, 0, 1, 2 so add 1 to the downsampled labels to convert to indices in color_lut
        colors = self.COLOR_LUT[downsampled + 1]  # shape (width, 4)
        color_bar = np.repeat(
            colors[np.newaxis, :, :], self._bar_height, axis=0
        )  # shape (bar_height, width, 4)

        img = QImage(
            color_bar.data,
            color_bar.shape[1],
            color_bar.shape[0],
            QImage.Format.Format_RGBA8888,
        )
        painter = QPainter(self._pixmap)
        painter.drawImage(0, self._bar_padding, img)
        painter.end()

    def set_num_frames(self, num_frames):
        """sets the number of frames in the current video, and resets the display with a blank track"""
        self._labels = np.full(num_frames, TrackLabels.Label.NONE.value, dtype=np.byte)
        super().set_num_frames(num_frames)
