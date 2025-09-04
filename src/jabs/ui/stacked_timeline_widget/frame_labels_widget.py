from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget


class FrameLabelsWidget(QWidget):
    """Widget for drawing frame ticks and labels below a LabelOverviewWidget.

    Displays tick marks and frame numbers for a sliding window of frames centered around the current frame.
    Intended to visually indicate frame positions and intervals in a video labeling interface.

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
        self._frame_width = self.size().width() // self._nframes
        self._adjusted_width = self._nframes * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) / 2

        self._font = QFont("Arial", 12)
        self._font_metrics = QFontMetrics(self._font)
        self._font_height = self._font_metrics.height()

    def sizeHint(self):
        """Give an initial starting size.

        Width hint is not so important because we allow the widget to resize
        horizontally to fill the available container. The height is fixed,
        so the value used here sets the height of the widget.
        """
        return QSize(400, self._font_height + 10)

    def resizeEvent(self, event):
        """handle resize events"""
        self._frame_width = self.size().width() // self._nframes
        self._adjusted_width = self._nframes * self._frame_width
        self._offset = (self.size().width() - self._adjusted_width) / 2

    def paintEvent(self, event):
        """override QWidget paintEvent

        This draws the widget.
        """
        if self._num_frames == 0:
            return

        # starting and ending frames of the current view
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        qp = QPainter(self)
        # make the ticks the same color as the text
        qp.setBrush(QApplication.palette().text().color())
        qp.setFont(self._font)
        self._draw_ticks(qp, start, end)
        qp.end()

    def _draw_ticks(self, painter, start, end):
        """draw ticks at the proper interval and draw the frame number under the tick

        Args:
            painter: active QPainter
            start: starting frame number
            end: ending frame number
        """
        for i in range(start, end + 1):
            if (0 <= i <= self._num_frames) and i % self._tick_interval == 0:
                offset = self._offset + ((i - start + 0.5) * self._frame_width) - 1
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(offset, 0, 2, 8)

                label_text = f"{i}"
                label_width = self._font_metrics.horizontalAdvance(label_text)
                painter.setPen(QApplication.palette().text().color())
                painter.drawText(offset - label_width / 2 + 1, self._font_height + 8, label_text)

    def set_current_frame(self, current_frame):
        """called to reposition the view around new current frame"""
        self._current_frame = current_frame
        self.update()

    def set_num_frames(self, num_frames):
        """set number of frames in current video

        this is used to keep from drawing ticks past the end of the video
        """
        self._num_frames = num_frames
