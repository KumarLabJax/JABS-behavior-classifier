import numpy as np

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QImage
from PySide6.QtWidgets import QWidget, QSizePolicy

from .colors import (
    BEHAVIOR_COLOR,
    NOT_BEHAVIOR_COLOR,
    BACKGROUND_COLOR,
    POSITION_MARKER_COLOR,
    SELECTION_COLOR,
)


class ManualLabelWidget(QWidget):
    """widget used to show labels for a range of frames around the current frame"""

    _BORDER_COLOR = QColor(212, 212, 212)
    _SELECTION_COLOR = QColor(*SELECTION_COLOR)
    _POSITION_MARKER_COLOR = QColor(*POSITION_MARKER_COLOR)
    _BACKGROUND_COLOR = QColor(*BACKGROUND_COLOR)
    _BEHAVIOR_COLOR = QColor(*BEHAVIOR_COLOR)
    _NOT_BEHAVIOR_COLOR = QColor(*NOT_BEHAVIOR_COLOR)

    COLOR_LUT = np.array(
        [
            BACKGROUND_COLOR,
            NOT_BEHAVIOR_COLOR,
            BEHAVIOR_COLOR,
            (0, 0, 0, 0),  # transparent color used for identity gaps
        ],
        dtype=np.uint8,
    )
    GAP_INDEX = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # number of frames on each side of current frame to include in
        # sliding window
        self._window_size = 100
        self._nframes = self._window_size * 2 + 1

        # current position
        self._current_frame = 0
        self._selection_start = None

        # information about the video needed to properly render widdget
        self._num_frames = 0
        self._framerate = 0

        # TrackLabels object containing labels for current behavior & identity
        self._labels = None
        self._identity_mask = None

        self._bar_height = 30

        # size each frame takes up in the bar in pixels
        self._frame_width = self.size().width() // self._nframes
        self._adjusted_width = self._nframes * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) // 2

        # initialize some brushes and pens once rather than every paintEvent
        self._position_marker_pen = QPen(self._POSITION_MARKER_COLOR, 1, Qt.SolidLine)
        self._selection_brush = QBrush(self._SELECTION_COLOR, Qt.DiagCrossPattern)
        self._padding_brush = QBrush(self._BACKGROUND_COLOR, Qt.Dense6Pattern)

    def sizeHint(self):
        """Override QWidget.sizeHint to give an initial starting size.
        Width hint is not so important because we allow the widget to resize
        horizontally to fill the available container. The height is fixed,
        so the value used here sets the height of the widget.
        """
        return QSize(400, self._bar_height)

    def resizeEvent(self, event):
        self._frame_width = self.size().width() // self._nframes
        self._adjusted_width = self._nframes * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) // 2

    def paintEvent(self, event):
        """override QWidget paintEvent

        This draws the widget.
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
        qp.setPen(Qt.NoPen)

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
            mask = self._identity_mask[slice_start : slice_end + 1]

            # set color index to gap index (transparent) for any gaps in the identity
            color_indices[mask == 0] = self.GAP_INDEX

            # Map indices to RGBA colors
            colors = self.COLOR_LUT[color_indices]

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
                QImage.Format_RGBA8888,
            )
            qp.drawImage(self._offset + start_padding, 0, img)

        # Draw selection overlay if in select mode
        if self._selection_start is not None:
            self._draw_selection_overlay(qp)

        self._draw_position_marker(qp)
        self._draw_bounding_box(qp)
        self._draw_second_ticks(qp, start, end)

        # done drawing
        qp.end()

    def _draw_position_marker(self, painter: QPainter):
        """draws the current position marker

        Args:
            painter: active QPainter
        """
        painter.setPen(self._position_marker_pen)
        position_offset = self._offset + self._adjusted_width // 2
        painter.drawLine(position_offset, 0, position_offset, self._bar_height - 1)

    def _draw_bounding_box(self, painter: QPainter):
        """draws the bounding box around the bar

        Args:
            painter: active QPainter
        """
        painter.setPen(self._BORDER_COLOR)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(
            self._offset, 0, self._adjusted_width - 1, self._bar_height - 1
        )

    def _draw_selection_overlay(self, painter: QPainter):
        """draws the selection overlay

        Args:
            painter: active QPainter
        """

        # starting and ending frames of the current view
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        painter.setPen(Qt.NoPen)
        painter.setBrush(self._selection_brush)

        # figure out the start and width of the selection rectangle
        if self._selection_start < self._current_frame:
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

    def _draw_second_ticks(self, painter: QPainter, start: int, end: int):
        """draw ticks at one second intervals

        Args:
            painter: active QPainter
            start: starting frame number
            end: ending frame number
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
                painter.drawRect(offset, 0, self._frame_width - 1, 4)

    def set_labels(self, labels, mask=None):
        """load label track to display"""
        self._labels = labels
        self._identity_mask = mask
        self.update()

    def set_current_frame(self, current_frame):
        """called to reposition the view around new current frame"""
        self._current_frame = current_frame
        # force redraw
        self.repaint()

    def set_num_frames(self, num_frames):
        """set number of frames in current video, needed to properly render"""
        self._num_frames = num_frames

    def set_framerate(self, fps):
        """set the frame rate for the currently loaded video, needed to draw the
        ticks at one second intervals

        Args:
            fps: frame rate in frames per second
        """
        self._framerate = fps

    def start_selection(self, start_frame):
        """start highlighting selection from start_frame to self._current_frame"""
        self._selection_start = start_frame

    def clear_selection(self):
        """stop highlighting selection"""
        self._selection_start = None
