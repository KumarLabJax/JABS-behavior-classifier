import sys

from PyQt5.QtWidgets import QWidget, QApplication, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPalette
from PyQt5.QtCore import QSize, Qt, QPoint


class ManualLabelWidget(QWidget):

    def __init__(self):
        super().__init__()

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # number of frames on each side of current frame to include in
        # sliding window
        self._window_size = 100

        # current position
        self._current_frame = 0

        self._num_frames = 0

        # TrackLabels object containing labels for current behavior & identity
        self._labels = None

        # positioning
        self._padding_left = 10
        self._padding_top = 15
        self._bar_height = 50

    def sizeHint(self):
        """ override QWidget.sizeHint to give an initial starting size """
        return QSize(400, 80)

    def paintEvent(self, event):
        qp = QPainter(self)
        self.draw_bar(qp)

    def draw_bar(self, qp):
        width = self.size().width()
        bar_width = width - (2 * self._padding_left)
        pixels_per_frame = bar_width / (self._window_size * 2 + 1)

        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        slice_start = max(start, 0)
        slice_end = self._current_frame + self._window_size + 1

        label_blocks = self._labels.get_slice_blocks(slice_start, slice_end)

        # draw padding if any
        start_padding_width = 0
        if start < 0:
            start_padding_width = abs(start) * pixels_per_frame
            qp.setBrush(QBrush(QColor(128, 128, 128), Qt.DiagCrossPattern))
            qp.drawRect(self._padding_left, self._padding_top,
                        start_padding_width,
                        self._bar_height)

        # draw end padding if any
        end_padding_frames = 0
        if self._num_frames and end >= self._num_frames:
            end_padding_frames = end - (self._num_frames - 1)
            end_padding_width = end_padding_frames * pixels_per_frame
            qp.setPen(QColor(212, 212, 212))
            qp.setBrush(QBrush(QColor(128, 128, 128), Qt.DiagCrossPattern))
            qp.drawRect(self._padding_left + (self._window_size * 2 + 1 - end_padding_frames) * pixels_per_frame,
                        self._padding_top, end_padding_width, self._bar_height)

        # draw background color (will be color for no label)
        qp.setBrush(QColor(128, 128, 128))
        qp.drawRect(self._padding_left + start_padding_width,
                    self._padding_top,
                    bar_width - start_padding_width - (end_padding_frames * pixels_per_frame),
                    self._bar_height)

        # draw label blocks
        qp.setPen(Qt.NoPen)
        for block in label_blocks:
            if block['present']:
                qp.setBrush(QColor(128, 0, 0))
            else:
                qp.setBrush(QColor(0, 0, 128))

            block_width = (block['end'] - block['start'] + 1) * pixels_per_frame
            offset_x = self._padding_left + start_padding_width + block[
                'start'] * pixels_per_frame

            qp.drawRect(offset_x, self._padding_top, block_width,
                        self._bar_height)

        # draw bounding box
        qp.setPen(QColor(212, 212, 212))
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(self._padding_left, self._padding_top, bar_width,
                    self._bar_height)

        # draw current position
        pen = QPen(QColor(255, 255, 0), 2, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(width // 2, self._padding_top + 2, width // 2,
                    self._padding_top + 49)

    def set_labels(self, labels):
        self._labels = labels

    def set_current_frame(self, current_frame):
        self._current_frame = current_frame
        self.update()
