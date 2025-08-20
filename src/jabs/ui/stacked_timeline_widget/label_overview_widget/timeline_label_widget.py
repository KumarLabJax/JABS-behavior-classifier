import math

import numpy as np
from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import (
    QBrush,
    QImage,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from jabs.behavior_search import SearchHit
from jabs.project import TrackLabels

from ...colors import (
    BACKGROUND_COLOR,
    BEHAVIOR_COLOR,
    NOT_BEHAVIOR_COLOR,
    POSITION_MARKER_COLOR,
)
from .label_overview_util import diamond_at


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
            BACKGROUND_COLOR.getRgb(),
            NOT_BEHAVIOR_COLOR.getRgb(),
            BEHAVIOR_COLOR.getRgb(),
            (144, 102, 132, 255),  # MIX
            (0, 0, 0, 0),  # PAD
        ],
        dtype=np.uint8,
    )
    _BAR_HEIGHT = 8  # height of the bar in pixels
    _BAR_PADDING = 3  # padding around the bar in pixels
    _WINDOW_SIZE = 100  # number of frames to show on either side of the current frame

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._bar_height = self._BAR_HEIGHT
        self._bar_padding = self._BAR_PADDING
        self._height = self._bar_height + 2 * self._bar_padding
        self._window_size = self._WINDOW_SIZE
        self._frames_in_view = 2 * self._window_size + 1

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # TrackLabels object containing labels for current behavior & identity
        self._labels = None

        # search results to render in the bar
        self._search_results: list[SearchHit] = []

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

    def sizeHint(self) -> QSize:
        """Return the recommended size for the widget.

        Returns:
            QSize: The preferred size for the timeline label widget.
        """
        return QSize(400, self._height)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resize events.

        Recalculates scaling and updates the bar to match the new size.

        Args:
            event (QResizeEvent): The resize event.
        """
        # if no video is loaded, there is nothing to display and nothing to
        # resize
        if self._num_frames == 0:
            return

        self._update_scale()
        self._update_bar()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Render the timeline label bar and highlight the current frame.

        Args:
            event (QPaintEvent): The paint event.
        """
        # make sure we have something to draw
        if self._pixmap is None or self._bin_size == 0:
            return

        qp = QPainter(self)

        # get the current position
        mapped_position = self._current_frame // self._bin_size
        start = mapped_position - (self._window_size // self._bin_size) + self._pixmap_offset

        # highlight the current position
        qp.setPen(QPen(POSITION_MARKER_COLOR, 1, Qt.PenStyle.SolidLine))
        qp.setBrush(QBrush(POSITION_MARKER_COLOR, Qt.BrushStyle.Dense4Pattern))
        qp.drawRect(start, 0, self._frames_in_view // self._bin_size, self.size().height() - 1)

        # draw the actual bar
        qp.drawPixmap(0 + self._pixmap_offset, 0, self._pixmap)

        qp.setPen(QPen(Qt.GlobalColor.green, 1, Qt.PenStyle.SolidLine))
        qp.setBrush(QBrush(Qt.GlobalColor.green, Qt.BrushStyle.SolidPattern))
        center_y = self.size().height() // 2
        diamond_w = self.size().height() // 8
        diamond_h = self.size().height() // 8
        for hit in self._search_results:
            start_pos = hit.start_frame // self._bin_size + self._pixmap_offset
            end_pos = (hit.end_frame + 1) // self._bin_size + self._pixmap_offset
            qp.drawLine(start_pos, center_y, end_pos, center_y)
            qp.drawPolygon(diamond_at(start_pos, center_y, diamond_w, diamond_h))
            qp.drawPolygon(diamond_at(end_pos, center_y, diamond_w, diamond_h))

    def set_labels(self, labels: TrackLabels) -> None:
        """Load and display a new label track.

        Args:
            labels (TrackLabels): The label track to display.
        """
        self._labels = labels
        self.update_labels()

    def set_search_results(self, search_results: list[SearchHit]) -> None:
        """Set the search results for the widget.

        Args:
            search_results (list[SearchHit]): List of SearchHit objects to display.
        """
        self._search_results = search_results

    @Slot(int)
    def set_current_frame(self, current_frame: int) -> None:
        """Set the current frame to highlight in the timeline.

        Args:
            current_frame (int): The index of the current frame.
        """
        self._current_frame = current_frame
        self.update()

    def set_num_frames(self, num_frames: int) -> None:
        """Set the total number of frames in the video.

        Args:
            num_frames (int): The number of frames.
        """
        self._num_frames = num_frames
        self._update_scale()
        self._update_bar()

    def update_labels(self) -> None:
        """Update and redraw the timeline bar to reflect the current label data.

        Regenerates the color representation of the labels and updates the widget display.
        Should be called whenever the label data changes to keep the visualization in sync.
        """
        self._update_bar()
        self.update()

    def reset(self) -> None:
        """Reset the widget to its initial state.

        Clears any loaded labels and recalculates the scale, preparing the widget for new data.
        """
        self._labels = None
        self._update_scale()

    def _update_bar(self) -> None:
        """Downsample the label array and update the bar pixmap for display.

        Converts the current labels into a color bar, downsampling as needed to fit the widget width,
        and updates the internal pixmap for efficient rendering. The internal pixmap is reused by paintEvent
        and only updated when the labels change or the widget is resized.
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

    def _update_scale(self) -> None:
        """Recalculate the bin size and pixmap offset for the timeline bar.

        Determines how many frames each horizontal pixel represents and computes the necessary
        padding to center the bar, based on the widget width and total frame count.
        """
        width = self.size().width()

        if width and self._num_frames:
            # calculate the bin size based on the number of frames and the
            # width of the widget
            pad_size = math.ceil(float(self._num_frames) / width) * width - self._num_frames
            self._bin_size = int(self._num_frames + pad_size) // width

            padding = (self._bin_size * width - self._num_frames) // self._bin_size
            self._pixmap_offset = padding // 2
