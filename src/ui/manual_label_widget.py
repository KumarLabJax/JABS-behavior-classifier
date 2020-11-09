from itertools import groupby

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPalette
from PyQt5.QtWidgets import QWidget, QSizePolicy

from .colors import (BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR, BACKGROUND_COLOR,
                     POSITION_MARKER_COLOR, SELECTION_COLOR)


class ManualLabelWidget(QWidget):
    """
    widget used to show labels for a range of frames around the current frame
    """

    _BORDER_COLOR = QColor(212, 212, 212)
    _SELECTION_COLOR = QColor(*SELECTION_COLOR)
    _POSITION_MARKER_COLOR = QColor(*POSITION_MARKER_COLOR)
    _BACKGROUND_COLOR = QColor(*BACKGROUND_COLOR)
    _BEHAVIOR_COLOR = QColor(*BEHAVIOR_COLOR)
    _NOT_BEHAVIOR_COLOR = QColor(*NOT_BEHAVIOR_COLOR)

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

        self._bar_height = 40

        # size each frame takes up in the bar in pixels
        self._frame_width = self.size().width() // self._nframes
        self._adjusted_width = self._nframes * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) / 2

        # initialize some brushes and pens once rather than every paintEvent
        self._position_marker_pen = QPen(self._POSITION_MARKER_COLOR, 1,
                                         Qt.SolidLine)
        self._selection_brush = QBrush(self._SELECTION_COLOR,
                                       Qt.DiagCrossPattern)
        self._padding_brush = QBrush(self._BACKGROUND_COLOR, Qt.Dense6Pattern)

    def sizeHint(self):
        """
        Override QWidget.sizeHint to give an initial starting size.
        Width hint is not so important because we allow the widget to resize
        horizontally to fill the available container. The height is fixed,
        so the value used here sets the height of the widget.
        """
        return QSize(400, self._bar_height)

    def resizeEvent(self, event):
        self._frame_width = self.size().width() // self._nframes
        self._adjusted_width = self._nframes * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) / 2

    def paintEvent(self, event):
        """
        override QWidget paintEvent

        This draws the widget.
        TODO: this could could be broken up into a few logical steps
        """

        width = self._adjusted_width

        # starting and ending frames of the current view
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        # slice size for grabbing label blocks
        slice_start = max(start, 0)
        slice_end = min(end, self._num_frames - 1)

        if self._labels is not None:
            label_blocks = self._labels.get_slice_blocks(slice_start, slice_end)
        else:
            label_blocks = []

        qp = QPainter(self)
        qp.setPen(Qt.NoPen)

        # draw start padding if any
        start_padding_width = 0
        if start < 0:
            qp.setBrush(self._padding_brush)
            start_padding_width = abs(start) * self._frame_width
            qp.drawRect(self._offset, 0, start_padding_width, self._bar_height)

        # draw end padding if any
        end_padding_frames = 0
        if self._num_frames and end >= self._num_frames:
            qp.setBrush(self._padding_brush)
            end_padding_frames = end - (self._num_frames - 1)
            end_padding_width = end_padding_frames * self._frame_width
            qp.drawRect(
                self._offset + (self._nframes - end_padding_frames) * self._frame_width,
                0, end_padding_width, self._bar_height
            )

        # draw background color (will be color for no label)
        qp.setBrush(self._BACKGROUND_COLOR)
        qp.drawRect(self._offset + start_padding_width, 0,
                    width - start_padding_width - (end_padding_frames *
                                                   self._frame_width),
                    self._bar_height)

        # draw gap blocks
        for block in self._get_gap_blocks(slice_start, end):
            block_width = (block['end'] - block['start'] + 1) \
                          * self._frame_width
            offset_x = self._offset + start_padding_width + block['start'] \
                       * self._frame_width

            # clear background
            qp.setBrush(self.palette().color(QPalette.Background))
            qp.drawRect(offset_x, 0, block_width, self._bar_height)
            qp.setBrush(self._padding_brush)
            qp.drawRect(offset_x, 0, block_width, self._bar_height)

        # draw label blocks
        for block in label_blocks:
            if block['present']:
                qp.setBrush(self._BEHAVIOR_COLOR)
            else:
                qp.setBrush(self._NOT_BEHAVIOR_COLOR)

            block_width = (block['end'] - block['start'] + 1) \
                          * self._frame_width
            offset_x = self._offset + start_padding_width + block['start'] \
                       * self._frame_width

            qp.drawRect(offset_x, 0, block_width, self._bar_height)

        # highlight current selection
        if self._selection_start is not None:
            qp.setPen(Qt.NoPen)
            qp.setBrush(self._selection_brush)
            if self._selection_start < self._current_frame:
                # other end of selection is left of the current frame
                selection_start = max(self._selection_start - start, 0)
                selection_width = (
                    self._current_frame - max(start, self._selection_start) + 1
                ) * self._frame_width
            elif self._selection_start > self._current_frame:
                # other end of selection is to the right of the current frame
                selection_start = self._current_frame - start
                selection_width = (
                    min(end, self._selection_start) - self._current_frame + 1
                ) * self._frame_width
            else:
                # only the current frame is selected
                selection_start = self._current_frame - start
                selection_width = self._frame_width

            qp.drawRect(self._offset + (selection_start * self._frame_width), 0,
                        selection_width, self._bar_height)

        # draw current position indicator
        qp.setPen(self._position_marker_pen)
        position_offset = self._offset + (width / 2)  # midpoint of widget in pixels
        qp.drawLine(position_offset, 0, position_offset, self._bar_height - 1)

        # draw bounding box
        qp.setPen(self._BORDER_COLOR)
        qp.setBrush(Qt.NoBrush)
        # need to adjust the width and height to account for the pen
        qp.drawRect(self._offset, 0, width - 1, self._bar_height - 1)

        self._draw_second_ticks(qp, start, end)

        qp.end()

    def _draw_second_ticks(self, painter, start, end):
        """
        draw ticks at one second intervals
        :param painter: active QPainter
        :param start: starting frame number
        :param end: ending frame number
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
        """ load label track to display """
        self._labels = labels
        self._identity_mask = mask
        self.update()

    def set_current_frame(self, current_frame):
        """ called to reposition the view around new current frame """
        self._current_frame = current_frame
        # force redraw
        self.update()

    def set_num_frames(self, num_frames):
        """ set number of frames in current video, needed to properly render """
        self._num_frames = num_frames

    def set_framerate(self, fps):
        """
        set the frame rate for the currently loaded video, needed to draw the
        ticks at one second intervals
        :param fps: frame rate in frames per second
        """
        self._framerate = fps

    def start_selection(self, start_frame):
        """
        start highlighting selection from start_frame to self._current_frame
        """
        self._selection_start = start_frame

    def clear_selection(self):
        """ stop highlighting selection """
        self._selection_start = None

    def _get_gap_blocks(self, start, end):
        """ generate blocks for gaps in the current identity track """
        block_start = 0
        blocks = []

        if self._identity_mask is not None:
            for val, group in groupby(self._identity_mask[start:end+1]):
                count = len([*group])
                if val == 0:
                    blocks.append({
                        'start': block_start,
                        'end': block_start + count - 1,
                    })
                block_start += count
        return blocks
