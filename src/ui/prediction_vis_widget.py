from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
from PyQt5.QtWidgets import QWidget, QSizePolicy

from src.labeler.track_labels import TrackLabels
from .colors import (BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR, BACKGROUND_COLOR,
                     POSITION_MARKER_COLOR, SELECTION_COLOR)


class PredictionVisWidget(QWidget):
    """
    widget used to show predicted class for a range of frames around the
    current frame
    """

    _BORDER_COLOR = QColor(212, 212, 212)
    _SELECTION_COLOR = QColor(*SELECTION_COLOR)
    _POSITION_MARKER_COLOR = QColor(*POSITION_MARKER_COLOR)
    _BACKGROUND_COLOR = QColor(*BACKGROUND_COLOR)
    _BEHAVIOR_COLOR = QColor(*BEHAVIOR_COLOR)
    _NOT_BEHAVIOR_COLOR = QColor(*NOT_BEHAVIOR_COLOR)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._bar_height = 40

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # number of frames on each side of current frame to include in
        # sliding window
        self._window_size = 100
        self._nframes = self._window_size * 2 + 1

        # current position
        self._current_frame = 0

        # information about the video needed to properly render widget
        self._num_frames = 0

        self._predictions = None
        self._probabilities = None

        # size each frame takes up in the bar in pixels
        self._frame_width = self.size().width() // self._nframes
        self._adjusted_width = self._nframes * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) / 2

        # initialize some brushes and pens once rather than every paintEvent
        self._position_marker_pen = QPen(self._POSITION_MARKER_COLOR, 1,
                                         Qt.SolidLine)
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
        """

        # width of entire widget
        width = self._adjusted_width

        # starting and ending frames of the current view
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        # slice size for grabbing label blocks
        slice_start = max(start, 0)

        qp = QPainter(self)
        qp.setPen(Qt.NoPen)

        # draw start padding if any
        start_padding_width = 0
        if start < 0:
            start_padding_width = abs(start) * self._frame_width
            qp.setBrush(self._padding_brush)
            qp.drawRect(self._offset, 0, start_padding_width, self._bar_height)

        # draw end padding if any
        end_padding_frames = 0
        if self._num_frames and end >= self._num_frames:
            end_padding_frames = end - (self._num_frames - 1)
            end_padding_width = end_padding_frames * self._frame_width
            qp.setBrush(self._padding_brush)
            qp.drawRect(
                self._offset + (self._nframes - end_padding_frames) * self._frame_width,
                0, end_padding_width, self._bar_height
            )

        # draw background color (will be color for no label)
        qp.setBrush(QColor(255, 255, 255))
        qp.drawRect(self._offset + start_padding_width, 0,
                    width - start_padding_width - (end_padding_frames *
                                                   self._frame_width),
                    self._bar_height)

        if self._predictions is not None:
            slice_index = -1
            for label in self._predictions[slice_start:end+1]:
                slice_index += 1

                if label == TrackLabels.Label.BEHAVIOR:
                    color = self._BEHAVIOR_COLOR
                elif label == TrackLabels.Label.NOT_BEHAVIOR:
                    color = self._NOT_BEHAVIOR_COLOR
                else:
                    continue
                color.setAlphaF(
                    self._probabilities[slice_start:end + 1][slice_index])
                qp.setBrush(color)

                offset_x = self._offset + start_padding_width + slice_index * self._frame_width
                qp.drawRect(offset_x, 0, self._frame_width, self._bar_height)

        # draw current position indicator
        qp.setPen(self._position_marker_pen)
        position_offset = self._offset + self._adjusted_width / 2
        qp.drawLine(position_offset, 0, position_offset, self._bar_height - 1)

        # draw bounding box
        qp.setPen(self._BORDER_COLOR)
        qp.setBrush(Qt.NoBrush)
        # need to adjust the width and height to account for the pen
        qp.drawRect(self._offset, 0, width - 1, self._bar_height - 1)
        qp.end()

    def set_predictions(self, predictions, probabilities):
        """ set prediction data to display """
        self._predictions = predictions
        self._probabilities = probabilities
        self.update()

    def set_current_frame(self, current_frame):
        """ called to reposition the view around new current frame """
        self._current_frame = current_frame
        # force redraw
        self.update()

    def set_num_frames(self, num_frames):
        """ set number of frames in current video, needed to properly render """
        self._num_frames = num_frames
