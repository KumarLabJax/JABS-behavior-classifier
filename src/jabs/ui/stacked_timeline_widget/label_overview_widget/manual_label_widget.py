from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
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
    _BAR_HEIGHT_COMPACT = 18
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
        # need to strip "compact" out of kwargs before calling super
        self._compact = compact = kwargs.pop("compact", False)
        super().__init__(*args, **kwargs)
        self._color_lut: npt.NDArray[np.uint8] = self.COLOR_LUT

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

        # Direct LUT-index array; binary labels are pre-offset (+1) on set_labels.
        self._labels: npt.NDArray[np.int16] | None = None
        self._identity_mask: npt.NDArray[np.int16] | None = None

        # search results to render in the bar
        self._search_results: list[SearchHit] = []

        self._bar_height = self._BAR_HEIGHT_COMPACT if compact else self._BAR_HEIGHT

        # float pixels per frame; set in resizeEvent
        self._frame_width: float = 0.0

        # initialize some brushes and pens once rather than every paintEvent
        self._position_marker_pen = QPen(POSITION_MARKER_COLOR, 1, Qt.PenStyle.SolidLine)
        self._selection_brush = QBrush(SELECTION_COLOR, Qt.BrushStyle.DiagCrossPattern)
        self._padding_brush = QBrush(BACKGROUND_COLOR, Qt.BrushStyle.Dense6Pattern)

    @property
    def compact(self) -> bool:
        """Whether the widget is in compact mode.

        Returns:
            bool: True if the widget is in compact mode, False otherwise.
        """
        return self._compact

    @compact.setter
    def compact(self, value: bool) -> None:
        """Set compact mode and trigger a layout and repaint update."""
        if self._compact != value:
            self._compact = value
            self._bar_height = self._BAR_HEIGHT_COMPACT if self._compact else self._BAR_HEIGHT
            self.updateGeometry()
            self.update()

    def set_color_lut(self, lut: npt.NDArray[np.uint8]) -> None:
        """Replace the color lookup table used to render label frames.

        In binary mode this is never called and the class-level ``COLOR_LUT``
        is used.  In multi-class mode ``StackedTimelineWidget`` calls this with
        the per-behavior palette produced by
        :func:`jabs.ui.colors.build_multiclass_color_lut`.

        Args:
            lut: RGBA array of shape ``(N, 4)`` mapping class indices to colors.

        Raises:
            ValueError: If ``lut`` is not a 2-D array with exactly 4 columns.
        """
        if lut.ndim != 2 or lut.shape[1] != 4:
            raise ValueError(f"lut must have shape (N, 4), got {lut.shape}")
        self._color_lut = lut
        self.update()

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

        Updates the floating-point pixels-per-frame value based on the new widget width.
        Content expands to fill the full width with no padding.

        Args:
            event: QResizeEvent containing the new and old size of the widget.
        """
        super().resizeEvent(event)
        w = self.size().width()
        self._frame_width = w / self._window_frames_total if w > 0 else 0.0

    def paintEvent(self, event: QPaintEvent) -> None:
        """Handle widget paint events.

        Draws the label bar, including frame-wise color-coded labels, selection overlay, current frame marker,
        bounding box, and second ticks. Handles out-of-bounds frames with a padding pattern and uses a color
        lookup table to map label values to colors.

        Args:
            event: QPaintEvent containing the region to be redrawn.
        """
        widget_width = self.size().width()
        if widget_width == 0 or self._window_frames_total == 0:
            return
        self._frame_width = widget_width / self._window_frames_total
        fw = self._frame_width

        # starting and ending frames of the current view
        # since the current frame is centered start might be negative and end might be > num_frames
        # out of bounds frames will be padded with a pattern
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        # slice size for grabbing label blocks
        slice_start = max(start, 0)
        slice_end = min(end, self._num_frames - 1)
        n_in_bounds = slice_end - slice_start + 1

        # Number of padding frames before the in-bounds content
        n_start_pad = max(0, -start)

        # Pixel edges of the in-bounds region using floating-point frame positions
        in_bounds_x0 = math.floor(n_start_pad * fw)
        in_bounds_x1 = math.floor((n_start_pad + n_in_bounds) * fw)
        in_bounds_px = in_bounds_x1 - in_bounds_x0

        qp = QPainter(self)
        qp.setPen(Qt.PenStyle.NoPen)

        # Fill entire bar with the padding pattern, then overdraw in-bounds content
        qp.setBrush(self._padding_brush)
        qp.drawRect(0, 0, widget_width, self._bar_height)

        # Draw the main bar image
        if self._labels is not None and n_in_bounds > 0 and in_bounds_px > 0:
            # _labels stores direct LUT indices (binary labels are pre-offset on set_labels)
            color_indices = self._labels[slice_start : slice_end + 1]

            # Map indices to RGBA colors
            colors = self._color_lut[color_indices]

            # set alpha for frames with dropped identity to make them semi-transparent
            gap_mask = self._identity_mask[slice_start : slice_end + 1] == 0
            colors[gap_mask, 3] = self.GAP_ALPHA

            # Per-frame pixel widths: floor((i+1)*fw) - floor(i*fw) for each frame
            win_indices = np.arange(n_start_pad, n_start_pad + n_in_bounds + 1, dtype=np.float64)
            frame_px_widths = np.diff(np.floor(win_indices * fw)).astype(int)

            # expand color array to bar height: shape (bar_height, n_in_bounds, 4)
            colors_bar = np.repeat(colors[np.newaxis, :, :], self._bar_height, axis=0)

            # Expand each frame to its pixel width: shape (bar_height, in_bounds_px, 4)
            colors_bar = np.repeat(colors_bar, frame_px_widths, axis=1)

            img = QImage(
                colors_bar.data,
                colors_bar.shape[1],
                colors_bar.shape[0],
                QImage.Format.Format_RGBA8888,
            )
            qp.drawImage(in_bounds_x0, 0, img)

        # Draw selection overlay if in select mode
        if self._selection_start is not None:
            self._draw_selection_overlay(qp)

        render_search_hits(
            qp,
            self._search_results,
            0,
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
        position_offset = self.size().width() // 2
        painter.drawLine(position_offset, 0, position_offset, self._bar_height - 1)

    def _draw_bounding_box(self, painter: QPainter) -> None:
        """Draw the bounding box around the label bar.

        Renders a rectangular border to visually frame the label bar area.

        Args:
            painter: The active QPainter used for drawing.
        """
        painter.setPen(self._BORDER_COLOR)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(0, 0, self.size().width() - 1, self._bar_height - 1)

    def _draw_selection_overlay(self, painter: QPainter) -> None:
        """Draw the selection overlay on the label bar.

        Renders a patterned rectangle over the selected range of frames, visually indicating
        the current selection between the selection start and the current frame.

        Args:
            painter: The active QPainter used for drawing.
        """
        fw = self._frame_width
        start = self._current_frame - self._window_size

        if self._selection_end is not None:
            true_start, true_end = self._selection_start, self._selection_end
        else:
            true_start = min(self._selection_start, self._current_frame)
            true_end = max(self._selection_start, self._current_frame)

        # Clamp to the visible window (in window-relative frame indices)
        vis_start = max(true_start - start, 0)
        vis_end = min(true_end - start + 1, self._window_frames_total)

        if vis_end <= vis_start:
            return

        x0 = math.floor(vis_start * fw)
        x1 = math.floor(vis_end * fw)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._selection_brush)
        painter.drawRect(x0, 0, max(x1 - x0, 1), self._bar_height)

    def _draw_second_ticks(self, painter: QPainter, start: int, end: int) -> None:
        """Draw vertical tick marks at one-second intervals along the label bar.

        Uses the current frame rate to determine tick positions, providing a visual reference
        for elapsed time within the windowed view.

        Args:
            painter: The active QPainter used for drawing.
            start: The starting frame number of the current view.
            end: The ending frame number of the current view.
        """
        if self._framerate == 0:
            return

        fw = self._frame_width
        painter.setBrush(self._BORDER_COLOR)
        for i in range(start, end + 1):
            if i % self._framerate == 0:
                win_i = i - start
                x0 = math.floor(win_i * fw)
                x1 = math.floor((win_i + 1) * fw)
                painter.drawRect(x0, 0, max(x1 - x0, 1), self._TICK_HEIGHT)

    def set_labels(self, labels: npt.NDArray[np.int16], mask: npt.NDArray[np.int16]) -> None:
        """Load and display the label track and identity mask.

        Updates the widget with new frame-wise behavior labels and the corresponding identity mask,
        then triggers a repaint to reflect the changes.

        ``labels`` must already be a direct LUT-index array: for binary classifiers, callers
        shift ``TrackLabels.get_labels()`` by +1 (NONE→0, NOT_BEHAVIOR→1, BEHAVIOR→2)
        before calling; multi-class callers can pass the array from
        ``VideoLabels.build_multiclass_label_array`` directly, since that generates the correct
        LUT indexing directly.

        Args:
            labels: Class-index array of shape ``(n_frames,)`` with dtype ``int16``.
            mask: Integer array indicating valid identity frames (1 = present, 0 = gap).
        """
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")
        if mask.ndim != 1:
            raise ValueError("mask must be a 1D array")
        if labels.shape[0] != mask.shape[0]:
            raise ValueError(
                f"labels and mask must have the same length: {labels.shape[0]} != {mask.shape[0]}"
            )
        if labels.shape[0] != self._num_frames:
            raise ValueError(
                f"labels length must match num_frames: {labels.shape[0]} != {self._num_frames}"
            )
        if np.any(labels < 0) or np.any(labels >= len(self._color_lut)):
            raise ValueError("labels contain indices outside the active color LUT range")

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
        self._color_lut = self.COLOR_LUT
        self.update()
