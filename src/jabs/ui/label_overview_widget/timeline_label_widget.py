import math

import numpy as np
from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..colors import (
    BACKGROUND_COLOR,
    BEHAVIOR_COLOR,
    NOT_BEHAVIOR_COLOR,
    POSITION_MARKER_COLOR,
)


class TimelineLabelWidget(QWidget):
    """Widget that shows a "zoomed out" overview of labels for the entire video.

    Because each pixel along the width ends up representing multiple frames,
    you can't see fine detail, but you can see where manual labels have been
    applied. This can be useful for seeking through the video to a location of
    labeling.

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.
    """

    # Define color LUT (RGBA)
    COLOR_LUT = np.array(
        [
            BACKGROUND_COLOR,
            NOT_BEHAVIOR_COLOR,
            BEHAVIOR_COLOR,
            [144, 102, 132, 255],  # MIX
            [0, 0, 0, 0],  # PAD
        ],
        dtype=np.uint8,
    )
    _RANGE_COLOR = QColor(*POSITION_MARKER_COLOR)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._bar_height = 8
        self._bar_padding = 3
        self._height = self._bar_height + 2 * self._bar_padding
        self._window_size = 100
        self._frames_in_view = 2 * self._window_size + 1

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # TrackLabels object containing labels for current behavior & identity
        self._labels = None

        # In order to indicate where the current frame is on the bar,
        # we need to know out which element it corresponds to in the downsampled
        # array. That maps to a pixel location in the bar. To calculate that
        # we will need to know the bin size. This is updated at every resize
        # event
        self._bin_size = 0

        self._pixmap = None
        self._pixmap_offset = 0

        self._current_frame = 0
        self._num_frames = 0

    def sizeHint(self):
        """Override QWidget.sizeHint to give an initial starting size.

        Width hint is not so important because we allow the widget to resize
        horizontally to fill the available container. The height is fixed,
        so the value used here sets the height of the widget.
        """
        return QSize(400, self._height)

    def resizeEvent(self, event):
        """Handle resize event.

        Recalculates scaling factors and calls update_bar() to downsample current label array
        and rerender the bar
        """
        # if no video is loaded, there is nothing to display and nothing to
        # resize
        if self._num_frames == 0:
            return

        self._update_scale()
        self._update_bar()

    def paintEvent(self, event):
        """override QWidget paintEvent"""
        # make sure we have something to draw
        if self._pixmap is None or self._bin_size == 0:
            return

        qp = QPainter(self)

        # get the current position
        mapped_position = self._current_frame // self._bin_size
        start = (
            mapped_position
            - (self._window_size // self._bin_size)
            + self._pixmap_offset
        )

        # highlight the current position
        qp.setPen(QPen(self._RANGE_COLOR, 1, Qt.PenStyle.SolidLine))
        qp.setBrush(QBrush(self._RANGE_COLOR, Qt.BrushStyle.Dense4Pattern))
        qp.drawRect(
            start, 0, self._frames_in_view // self._bin_size, self.size().height() - 1
        )

        # draw the actual bar
        qp.drawPixmap(0 + self._pixmap_offset, 0, self._pixmap)

    def set_labels(self, labels):
        """load label track to display"""
        self._labels = labels
        self.update_labels()

    @Slot(int)
    def set_current_frame(self, current_frame):
        """called to reposition the view"""
        self._current_frame = current_frame
        self.update()

    def set_num_frames(self, num_frames):
        """sets the number of frames in the current video"""
        self._num_frames = num_frames
        self._update_scale()
        self._update_bar()

    def update_labels(self):
        """Update the rendered labels and redraw."""
        self._update_bar()
        self.update()

    def reset(self):
        """Resets the widget state."""
        self._labels = None
        self._update_scale()
        self._update_bar()

    def _update_bar(self):
        """Updates the bar pixmap.

        Downsamples label array with the current size and updates self._pixmap
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

        downsampled = self._labels.downsample(self._labels.get_labels(), pixmap_width)

        # use downsampled labels to generate RGBA colors
        # labels are -1, 0, 1, 2 so add 1 to the downsampled labels to convert to indices in color_lut
        colors = self.COLOR_LUT[downsampled + 1]  # shape (width, 4)

        # resize colors to bar height: shape = (height, width, 4)
        color_bar = np.repeat(colors[np.newaxis, :, :], self._bar_height, axis=0)

        # convert bar to QImage and draw it to the pixmap
        img = QImage(
            color_bar.data,
            color_bar.shape[1],
            color_bar.shape[0],
            QImage.Format.Format_RGBA8888,
        )
        painter = QPainter(self._pixmap)
        painter.drawImage(0, self._bar_padding, img)
        painter.end()

    def _update_scale(self):
        """update scale factor and bin size"""
        width = self.size().width()

        if width and self._num_frames:
            # calculate the bin size based on the number of frames and the
            # width of the widget
            pad_size = (
                math.ceil(float(self._num_frames) / width) * width - self._num_frames
            )
            self._bin_size = int(self._num_frames + pad_size) // width

            padding = (self._bin_size * width - self._num_frames) // self._bin_size
            self._pixmap_offset = padding // 2
