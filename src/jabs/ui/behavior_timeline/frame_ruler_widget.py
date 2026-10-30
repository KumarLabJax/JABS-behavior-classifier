from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget

from .track_widgets.timeline_util import DETAIL_WINDOW_SIZE, frame_edge_x, window_frame_count


class FrameRulerWidget(QWidget):
    """Widget for drawing frame ticks and labels below the identity track rows.

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

        # number of frames shown on either side of the current frame by the detail
        # bars; the ticks drawn here number the frames inside that window
        self._window_size: int = DETAIL_WINDOW_SIZE

        # total number of frames being displayed in sliding window
        self._nframes = window_frame_count(self._window_size)

        # number of frames between ticks/labels
        self._tick_interval = 50

        # current position
        self._current_frame = 0

        # information about the video needed to properly render widget
        self._num_frames = 0

        # detail bar whose frames the ticks line up with; see set_reference_bar()
        self._reference_bar: QWidget | None = None

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
                offset = self._frame_center_x(i, start) - 1
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(offset, 0, 2, 8)

                label_text = f"{i}"
                label_width = self._font_metrics.horizontalAdvance(label_text)
                painter.setPen(QApplication.palette().text().color())
                painter.drawText(offset - label_width / 2 + 1, self._font_height + 8, label_text)

    def set_reference_bar(self, bar: QWidget | None) -> None:
        """Line the ticks up with a detail bar.

        The detail bars sit inside identity frames and track widgets with their own
        margins, so they are narrower than the ruler and start further right. The
        ruler maps frames across the reference bar's span instead of its own width.
        Every detail bar has the same horizontal geometry, so any visible one works.

        Args:
            bar: Detail bar to follow, or ``None`` to span the ruler's full width.
        """
        self._reference_bar = bar
        self.update()

    def _bar_span(self) -> tuple[int, int]:
        """Return the x offset and width of the reference bar in ruler coordinates."""
        bar = self._reference_bar
        if bar is None or not bar.isVisible():
            return 0, self.size().width()
        x = self.mapFromGlobal(bar.mapToGlobal(QPoint(0, 0))).x()
        return x, bar.width()

    def _frame_center_x(self, frame: int, start: int) -> float:
        """Return the x coordinate of the center of a frame in the sliding window.

        Uses the detail bars' frame-to-pixel mapping across the reference bar's span,
        so the ticks line up with the frames drawn in the bars.

        Args:
            frame: Frame number to locate.
            start: First frame number in the sliding window.

        Returns:
            The x coordinate, in pixels, of the center of ``frame``.
        """
        x0, width = self._bar_span()
        frame_width = width / self._nframes
        window_index = frame - start
        left = frame_edge_x(window_index, frame_width)
        right = frame_edge_x(window_index + 1, frame_width)
        return x0 + (left + right) / 2

    def set_current_frame(self, current_frame):
        """called to reposition the view around new current frame"""
        self._current_frame = current_frame
        self.update()

    def set_num_frames(self, num_frames):
        """set number of frames in current video

        this is used to keep from drawing ticks past the end of the video
        """
        self._num_frames = num_frames
