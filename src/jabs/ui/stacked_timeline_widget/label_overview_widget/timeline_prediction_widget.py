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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _update_bar(self) -> None:
        """Update the timeline bar pixmap with downsampled label colors.

        Overrides _update_bar() from parent class to use a np.ndarray as input instead of a TrackLabels object.
        Downsamples the label array to match the current pixmap width, maps labels to RGBA colors, and renders
        the color bar as a QPixmap for display.
        """
        if self._labels is None:
            return

        width = self.size().width()
        height = self.size().height()

        # create a pixmap with a width that evenly divides the total number of
        # frames so that each pixel along the width represents a bin of frames
        # (_update_scale() has done this, we can use pixmap_offset to figure
        # out how many pixels of padding will be on each side of the final
        # pixmap)
        pixmap_width = width - 2 * self._pixmap_offset

        self._pixmap = QPixmap(pixmap_width, height)
        self._pixmap.fill(Qt.GlobalColor.transparent)

        downsampled = TrackLabels.downsample(self._labels, pixmap_width)

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
