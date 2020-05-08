from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics
from PyQt5.QtCore import QSize, Qt


class FrameLabelsWidget(QWidget):
    """
    draws ticks and frame labels, intended to be used under one or more
    ManualLabelsWidget
    """

    _COLOR = QColor(212, 212, 212)

    def __init__(self, *args, **kwargs):
        super(FrameLabelsWidget, self).__init__(*args, **kwargs)

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # number of frames on each side of current frame to include in
        # sliding window
        # this needs to match what is set in ManualLabelsWidget, so once
        # we make this configurable, it needs to get set in both locations
        self._window_size = 100

        # total number of frames being displayed in sliding window
        self._nframes = self._window_size * 2 + 1

        # number of frames between ticks/labels
        self._tick_interval = 50

        # current position
        self._current_frame = 0

        # information about the video needed to properly render widget
        self._num_frames = 0

        # size each frame takes up in the bar in pixels
        self._frame_width = self.size().width() / self._nframes

        self._font = QFont("Arial", 12)
        self._font_metrics = QFontMetrics(self._font)
        self._font_height = self._font_metrics.height()

    def sizeHint(self):
        """
        Override QWidget.sizeHint to give an initial starting size.
        Width hint is not so important because we allow the widget to resize
        horizontally to fill the available container. The height is fixed,
        so the value used here sets the height of the widget.
        """
        return QSize(400, self._font_height + 10)

    def resizeEvent(self, event):
        self._frame_width = self.size().width() / self._nframes

    def paintEvent(self, event):
        """
        override QWidget paintEvent

        This draws the widget.
        """

        # starting and ending frames of the current view
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        qp = QPainter(self)
        qp.setBrush(self._COLOR)
        qp.setFont(self._font)
        self._draw_ticks(qp, start, end)
        qp.end()

    def _draw_ticks(self, painter, start, end):
        """
        draw ticks draw ticks at the proper interval and draw the frame
        number under the tick
        :param painter: active QPainter
        :param start: starting frame number
        :param end: ending frame number
        """

        for i in range(start, end + 1):
            if (0 <= i <= self._num_frames) and i % self._tick_interval == 0:
                offset = ((i - start + .5) * self._frame_width) - 2
                painter.setPen(Qt.NoPen)
                painter.drawRect(offset, 0, 2, 8)

                label_text = f"{i}"
                label_width = self._font_metrics.width(label_text)
                painter.setPen(self._COLOR)
                painter.drawText(offset - label_width/2 + 1,
                                 self._font_height + 8, label_text)

    def set_current_frame(self, current_frame):
        """ called to reposition the view around new current frame """
        self._current_frame = current_frame
        # force redraw
        self.update()

    def set_num_frames(self, num_frames):
        """
        set number of frames in current video, needed to keep from drawing
        ticks past the end of the video (pace that is drawn as padding by
        ManualLabelsWidget
        """
        self._num_frames = num_frames
