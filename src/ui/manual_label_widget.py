import sys

from PyQt5.QtWidgets import QWidget, QApplication, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPalette
from PyQt5.QtCore import QSize, Qt, QPoint


class ManualLabelWidget(QWidget):
    """
    widget used to show labels for a range of frames arount the current frame
    """

    _OUTLINE_COLOR = QColor(212, 212, 212)
    _POSITION_MARKER_COLOR = QColor(255, 255, 0)
    _BACKGROUND_COLOR = QColor(128, 128, 128)
    _BEHAVIOR_COLOR = QColor(128, 0, 0)
    _NOT_BEHAVIOR_COLOR = QColor(0, 0, 128)

    def __init__(self):
        super(ManualLabelWidget, self).__init__()

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # number of frames on each side of current frame to include in
        # sliding window
        self._window_size = 100
        self._frames_to_draw = self._window_size * 2 + 1

        # current position
        self._current_frame = 0
        self._selection_start = None

        self._num_frames = 0

        # TrackLabels object containing labels for current behavior & identity
        self._labels = None

        self._bar_height = 40

    def sizeHint(self):
        """
        Override QWidget.sizeHint to give an initial starting size.
        Width hint is not so important because we allow the widget to resize
        horizontally to fill the available container. The height is fixed,
        so the value used here sets the height of the widget.
        """
        return QSize(400, self._bar_height)

    def paintEvent(self, event):
        """
        override QWidget paintEvent

        This draws the widget.
        TODO: this could could be broken up into a few logical steps
        """
        # width of entire widget
        width = self.size().width()

        # size each frame takes up in the bar in pixels
        pixels_per_frame = width / self._frames_to_draw

        # starting and ending frames of the current view
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        # slice size for grabbing label blocks
        slice_start = max(start, 0)
        slice_end = self._current_frame + self._window_size

        label_blocks = self._labels.get_slice_blocks(slice_start, slice_end)

        qp = QPainter(self)
        qp.setPen(Qt.NoPen)

        # draw start padding if any
        start_padding_width = 0
        if start < 0:
            start_padding_width = abs(start) * pixels_per_frame
            qp.setBrush(QBrush(self._BACKGROUND_COLOR, Qt.Dense6Pattern))
            qp.drawRect(0, 0, start_padding_width, self._bar_height)

        # draw end padding if any
        end_padding_frames = 0
        if self._num_frames and end >= self._num_frames:
            end_padding_frames = end - (self._num_frames - 1)
            end_padding_width = end_padding_frames * pixels_per_frame
            qp.setBrush(QBrush(self._BACKGROUND_COLOR, Qt.Dense6Pattern))
            qp.drawRect(
                (self._frames_to_draw - end_padding_frames) * pixels_per_frame,
                0, end_padding_width, self._bar_height
            )

        # draw background color (will be color for no label)
        qp.setBrush(self._BACKGROUND_COLOR)
        qp.drawRect(start_padding_width, 0,
                    width - start_padding_width - (end_padding_frames * pixels_per_frame),
                    self._bar_height)

        # draw label blocks
        for block in label_blocks:
            if block['present']:
                qp.setBrush(self._BEHAVIOR_COLOR)
            else:
                qp.setBrush(self._NOT_BEHAVIOR_COLOR)

            block_width = (block['end'] - block['start'] + 1) * pixels_per_frame
            offset_x = start_padding_width + block['start'] * pixels_per_frame

            qp.drawRect(offset_x, 0, block_width, self._bar_height)

        # highlight current selection
        if self._selection_start is not None:
            qp.setPen(Qt.NoPen)
            qp.setBrush(QBrush(self._POSITION_MARKER_COLOR,
                               Qt.DiagCrossPattern))
            if self._selection_start < self._current_frame:
                # other end of selection is left of the current frame
                selection_start = max(self._selection_start - start, 0)
                selection_width = (
                    self._current_frame - max(start, self._selection_start) + 1
                ) * pixels_per_frame
            elif self._selection_start > self._current_frame:
                # other end of selection is to the right of the current frame
                selection_start = self._current_frame - start
                selection_width = (
                    min(end, self._selection_start) - self._current_frame + 1
                ) * pixels_per_frame

            else:
                # only the current frame is selected
                selection_start = self._current_frame - start
                selection_width = pixels_per_frame

            qp.drawRect(selection_start * pixels_per_frame, 0,
                        selection_width, self._bar_height)

        # draw current position indicator
        pen = QPen(self._POSITION_MARKER_COLOR, 1, Qt.SolidLine)
        qp.setPen(pen)
        position_offset = width / 2 - 1  # midpoint of widget in pixels
        qp.drawLine(position_offset, 0, position_offset, self._bar_height - 1)

        # draw bounding box
        qp.setPen(self._OUTLINE_COLOR)
        qp.setBrush(Qt.NoBrush)
        # need to adjust the width and height to account for the pen
        qp.drawRect(0, 0, width - 1, self._bar_height - 1)

    def set_labels(self, labels):
        """ load label track to display """
        self._labels = labels

    def set_current_frame(self, current_frame):
        """ called to reposition the view """
        self._current_frame = current_frame
        # force redraw
        self.update()

    def start_selection(self, start_frame):
        """
        start highlighting selection from start_frame to self._current_frame
        """
        self._selection_start = start_frame

    def clear_selection(self):
        """ stop highlighting selection """
        self._selection_start = None

