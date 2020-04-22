from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import QSize, Qt

from .utilities import BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR, BACKGROUND_COLOR
from src.labeler.track_labels import TrackLabels


class TimelineLabelWidget(QWidget):
    """
    Widget that shows a "zoomed out" overview of labels for the entire video.
    Because each pixel along wthe width ends up representing multiple frames,
    you can't see fine detail, but you can see where manual labels have been
    applied. This can be useful for seeking through the video to a location of
    labeling.
    """

    _BEHAVIOR_COLOR = QColor(*BEHAVIOR_COLOR)
    _NOT_BEHAVIOR_COLOR = QColor(*NOT_BEHAVIOR_COLOR)
    _BACKGROUND_COLOR = QColor(*BACKGROUND_COLOR)
    _RANGE_COLOR = QColor(255, 255, 204)

    def __init__(self, *args, **kwargs):
        super(TimelineLabelWidget, self).__init__(*args, **kwargs)

        self._bar_height = 5
        self._bar_padding = 3
        self._height = self._bar_height + 2 * self._bar_padding
        self._window_size = 100
        self._frames_in_view = 2 * self._window_size + 1

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # TrackLabels object containing labels for current behavior & identity
        self._labels = None

        self._current_frame = 0
        self._num_frames = 0

    def sizeHint(self):
        """
        Override QWidget.sizeHint to give an initial starting size.
        Width hint is not so important because we allow the widget to resize
        horizontally to fill the available container. The height is fixed,
        so the value used here sets the height of the widget.
        """
        return QSize(400, self._height)

    def set_labels(self, labels):
        """ load label track to display """
        self._labels = labels

    def set_current_frame(self, current_frame):
        """ called to reposition the view """
        self._current_frame = current_frame
        # force redraw
        self.update()

    def set_num_frames(self, num_frames):
        """ set the number of frames in the current video """
        self._num_frames = num_frames

    def paintEvent(self, event):
        """ override QWidget paintEvent """

        # don't draw anything if we don't have a label array to draw
        if self._labels is None:
            return

        width = self.size().width()
        height = self.size().height()

        qp = QPainter(self)

        # downsample the label array to fit the width we have to draw
        downsampled = self._labels.downsample(width)
        scale_factor = (width / self._num_frames)

        # find the location of the current frame scaled to the width
        scaled_position = int(self._current_frame * scale_factor)

        # draw a box around what is currently being displayed in the
        # ManualLabelWidget
        start = scaled_position - (self._window_size * scale_factor)
        qp.setPen(QPen(self._RANGE_COLOR, 1, Qt.SolidLine))
        qp.drawRect(start, 0, self._frames_in_view * scale_factor, height - 1)

        # draw the bar, each pixel along the width corresponds to a value in the
        # down sampled label array
        for x in range(width):
            if downsampled[x] == TrackLabels.Label.NONE:
                qp.setPen(self._BACKGROUND_COLOR)
            elif downsampled[x] == TrackLabels.Label.BEHAVIOR:
                qp.setPen(self._BEHAVIOR_COLOR)
            elif downsampled[x] == TrackLabels.Label.NOT_BEHAVIOR:
                qp.setPen(self._NOT_BEHAVIOR_COLOR)
            else:
                # bin contains mix of behavior/not behavior labels, color these
                # as magenta
                qp.setPen(Qt.magenta)

            # draw a vertical bar of pixels
            for y in range(self._bar_padding,
                           self._bar_padding + self._bar_height):
                qp.drawPoint(x, y)
