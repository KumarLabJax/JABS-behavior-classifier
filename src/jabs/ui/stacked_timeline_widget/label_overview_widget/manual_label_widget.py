import numpy as np
from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPaintEvent,
    QPen,
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
    SELECTION_COLOR,
)
from .label_overview_util import render_search_hits


class ManualLabelWidget(QWidget):
    """Widget for visualizing manual behavior labels in a video timeline.

    Displays a horizontal bar representing frame-wise labels, with color coding for behavior,
    non-behavior, and gaps in identity. Supports selection, current frame indication, and
    second ticks for time reference.

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.
    """

    _BORDER_COLOR = QColor(212, 212, 212)

    _BAR_HEIGHT = 30
    _DEFAULT_WIDTH = 400
    _DEFAULT_WINDOW_SIZE = 100
    _TICK_HEIGHT = 4

    COLOR_LUT: np.ndarray = np.array(
        [
            BACKGROUND_COLOR.getRgb(),
            NOT_BEHAVIOR_COLOR.getRgb(),
            BEHAVIOR_COLOR.getRgb(),
        ],
        dtype=np.uint8,
    )
    GAP_ALPHA = 128  # semi-transparent for gaps

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # number of frames on each side of current frame to include in
        # sliding window
        self._window_size = self._DEFAULT_WINDOW_SIZE
        self._window_frames_total = self._window_size * 2 + 1

        # current position
        self._current_frame = 0
        self._selection_start = None
        self._selection_end = None

        # information about the video needed to properly render widget
        self._num_frames = 0
        self._framerate = 0

        # TrackLabels object containing labels for current behavior & identity
        self._labels: np.ndarray | None = None
        self._identity_mask: np.ndarray | None = None

        # search results to render in the bar
        self._search_results: list[SearchHit] = []

        self._bar_height = self._BAR_HEIGHT

        # size each frame takes up in the bar in pixels
        self._frame_width = self.size().width() // self._window_frames_total
        self._adjusted_width = self._window_frames_total * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) // 2

        # initialize some brushes and pens once rather than every paintEvent
        self._position_marker_pen = QPen(POSITION_MARKER_COLOR, 1, Qt.PenStyle.SolidLine)
        self._selection_brush = QBrush(SELECTION_COLOR, Qt.BrushStyle.DiagCrossPattern)
        self._padding_brush = QBrush(BACKGROUND_COLOR, Qt.BrushStyle.Dense6Pattern)

    def sizeHint(self) -> QSize:
        """Return the recommended initial size for the widget.

        The width is flexible to allow horizontal expansion, while the height is fixed
        to set the vertical size of the label bar.

        Returns:
            QSize: The preferred size of the widget.
        """
        return QSize(self._DEFAULT_WIDTH, self._bar_height)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resize events.

        Updates internal frame width, adjusted width, and offset values based on the new widget size.
        This ensures that the label bar and its elements are correctly scaled and centered after resizing.

        Args:
            event: QResizeEvent containing the new and old size of the widget.
        """
        self._frame_width = self.size().width() // self._window_frames_total
        self._adjusted_width = self._window_frames_total * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) // 2

    def paintEvent(self, event: QPaintEvent) -> None:
        """Handle widget paint events.

        Draws the label bar, including frame-wise color-coded labels, selection overlay, current frame marker,
        bounding box, and second ticks. Handles out-of-bounds frames with a padding pattern and uses a color
        lookup table to map label values to colors.

        Args:
            event: QPaintEvent containing the region to be redrawn.
        """
        # starting and ending frames of the current view
        # since the current frame is centered start might be negative and end might be > num_frames
        # out of bounds frames will be padded with a pattern
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        # slice size for grabbing label blocks
        slice_start = max(start, 0)
        slice_end = min(end, self._num_frames - 1)

        qp = QPainter(self)
        qp.setPen(Qt.PenStyle.NoPen)

        # Calculate padding in pixels
        start_padding = max(0, -start) * self._frame_width

        # Use QPainter to fill drawing area with the padding pattern
        qp.setBrush(self._padding_brush)
        qp.drawRect(self._offset, 0, self._adjusted_width, self._bar_height)

        # Draw the main bar image
        if self._labels is not None:
            labels = self._labels.get_labels()[slice_start : slice_end + 1]

            # turn labels into indices into color LUT, labels are -1, 0, 1 for no label, not behavior, behavior
            # add 1 to the labels to convert to indexes in color_lut
            color_indices = labels + 1

            # Map indices to RGBA colors
            colors = self.COLOR_LUT[color_indices]

            # set alpha for frames with dropped identity to make them semi-transparent
            gap_mask = self._identity_mask[slice_start : slice_end + 1] == 0
            colors[gap_mask, 3] = self.GAP_ALPHA

            # expand color array to bar height
            # shape (bar_height, frames in view, 4)
            colors_bar = np.repeat(colors[np.newaxis, :, :], self._bar_height, axis=0)

            # Repeat each column (frame) by self._frame_width pixels
            # shape: (bar_height, frames in view * frame_width, 4)
            colors_bar = np.repeat(colors_bar, self._frame_width, axis=1)

            # Draw the main bar image accounting for start padding
            img = QImage(
                colors_bar.data,
                colors_bar.shape[1],
                colors_bar.shape[0],
                QImage.Format.Format_RGBA8888,
            )
            qp.drawImage(self._offset + start_padding, 0, img)

        # Draw selection overlay if in select mode
        if self._selection_start is not None:
            self._draw_selection_overlay(qp)

        render_search_hits(
            qp,
            self._search_results,
            self._offset,
            start,
            self._frame_width,
            self._bar_height,
            self._window_frames_total,
        )

        self._draw_position_marker(qp)
        self._draw_bounding_box(qp)
        self._draw_second_ticks(qp, start, end)

        # done drawing
        qp.end()

    def _draw_position_marker(self, painter: QPainter) -> None:
        """Draw the marker indicating the current frame position.

        Renders a vertical line at the center of the label bar to visually indicate
        the current frame within the windowed view.

        Args:
            painter: The active QPainter used for drawing.
        """
        painter.setPen(self._position_marker_pen)
        position_offset = self._offset + self._adjusted_width // 2
        painter.drawLine(position_offset, 0, position_offset, self._bar_height - 1)

    def _draw_bounding_box(self, painter: QPainter) -> None:
        """Draw the bounding box around the label bar.

        Renders a rectangular border to visually frame the label bar area.

        Args:
            painter: The active QPainter used for drawing.
        """
        painter.setPen(self._BORDER_COLOR)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._offset, 0, self._adjusted_width - 1, self._bar_height - 1)

    def _draw_selection_overlay(self, painter: QPainter) -> None:
        """Draw the selection overlay on the label bar.

        Renders a patterned rectangle over the selected range of frames, visually indicating
        the current selection between the selection start and the current frame.

        Args:
            painter: The active QPainter used for drawing.
        """
        # starting and ending frames of the current view
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._selection_brush)

        # figure out the start and width of the selection rectangle
        if self._selection_end is not None:
            # we have an explicit end frame for the selection
            selection_start = max(self._selection_start - start, 0)
            selection_width = (
                min(end, self._selection_end) - max(start, self._selection_start) + 1
            ) * self._frame_width
        elif self._selection_start < self._current_frame:
            # normal selection, start is lower than current frame
            selection_start = max(self._selection_start - start, 0)
            selection_width = (
                self._current_frame - max(start, self._selection_start) + 1
            ) * self._frame_width
        elif self._selection_start > self._current_frame:
            # user started selecting and then scanned backwards, start is greater than current frame
            selection_start = self._current_frame - start
            selection_width = (
                min(end, self._selection_start) - self._current_frame + 1
            ) * self._frame_width
        else:
            # single frame selected
            selection_start = self._current_frame - start
            selection_width = self._frame_width

        # draw the selection overlay rectangle
        painter.drawRect(
            self._offset + (selection_start * self._frame_width),
            0,
            selection_width,
            self._bar_height,
        )

    def _draw_second_ticks(self, painter: QPainter, start: int, end: int) -> None:
        """Draw vertical tick marks at one-second intervals along the label bar.

        Uses the current frame rate to determine tick positions, providing a visual reference
        for elapsed time within the windowed view.

        Args:
            painter: The active QPainter used for drawing.
            start: The starting frame number of the current view.
            end: The ending frame number of the current view.
        """
        # can't draw if we don't know the frame rate yet
        if self._framerate == 0:
            return

        painter.setBrush(self._BORDER_COLOR)
        for i in range(start, end + 1):
            # we could add i > 0 and i < num_frames to this if test to avoid
            # drawing ticks in the 'padding' at the start and end of the video
            if i % self._framerate == 0:
                offset = (i - start) * self._frame_width + self._offset
                painter.drawRect(offset, 0, self._frame_width - 1, self._TICK_HEIGHT)

    def set_labels(self, labels: TrackLabels, mask: np.ndarray) -> None:
        """Load and display the label track and identity mask.

        Updates the widget with new frame-wise behavior labels and the corresponding identity mask,
        then triggers a repaint to reflect the changes.

        Args:
            labels: TrackLabels object behavior labels for each frame.
            mask: A numpy array indicating valid identity frames (1 for present, 0 for gap).
        """
        self._labels = labels
        self._identity_mask = mask
        self.update()

    def set_search_results(self, search_results: list[SearchHit]) -> None:
        """Set the search results for the widget.

        Args:
            search_results (list[SearchHit]): List of SearchHit objects to display.
        """
        self._search_results = search_results

    @Slot(int)
    def set_current_frame(self, current_frame: int) -> None:
        """
        Update the current frame and refresh the label bar view.

        Recenters the widget's window around the specified frame and triggers a repaint
        to reflect the new current frame position.

        Args:
            current_frame: The index of the frame to center the view on.
        """
        self._current_frame = current_frame
        self.update()

    def set_num_frames(self, num_frames: int) -> None:
        """Set the total number of frames in the current video.

        Updates the internal frame count, which is required for correct rendering
        of the label bar and its elements.

        Args:
            num_frames: The total number of frames in the loaded video.
        """
        self._num_frames = num_frames

    def set_framerate(self, fps: int) -> None:
        """Set the frame rate for the currently loaded video.

        Updates the internal frame rate value, which is used to determine the placement
        of second tick marks along the label bar.

        Args:
            fps: Frame rate in frames per second.
        """
        self._framerate = fps

    def start_selection(self, start_frame: int, end_frame: int | None = None) -> None:
        """Begin highlighting a selection range on the label bar.

        Sets the starting frame for the selection overlay, which will extend to the current frame
        unless an end frame is specified.

        Args:
            start_frame: The frame index where the selection begins.
            end_frame: Optional; the frame index where the selection ends. If not provided,
                the selection will extend to the current frame.
        """
        self._selection_start = start_frame
        self._selection_end = end_frame

    def clear_selection(self) -> None:
        """Stop highlighting the selection range.

        Clears the selection overlay from the label bar.
        """
        self._selection_start = None
        self._selection_end = None

    def reset(self) -> None:
        """Reset the widget to its initial state.

        Clears all internal attributes such as labels and identity mask and then triggers a repaint.
        """
        self._labels = None
        self._identity_mask = None
        self._selection_start = None
        self._selection_end = None
        self._num_frames = 0
        self.update()
