from __future__ import annotations

import math
from typing import cast

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
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from jabs.behavior_search import SearchHit

from ...colors import (
    BACKGROUND_COLOR,
    BEHAVIOR_COLOR,
    NOT_BEHAVIOR_COLOR,
    POSITION_MARKER_COLOR,
)
from .label_overview_util import diamond_at


def _srgb_to_linear(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert sRGB-encoded values in [0, 1] to linear light values."""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert linear light values in [0, 1] to sRGB-encoded values."""
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1.0 / 2.4) - 0.055)


def _downsample_to_size(
    labels: npt.NDArray[np.int16],
    lut: npt.NDArray[np.uint8],
    size: int,
) -> npt.NDArray[np.uint8]:
    """Downsample a class-index array to ``size`` pixels via proportional color blending.

    Each output pixel covers an equal-width interval in frame space. Frame
    contributions are weighted by the fractional overlap between that pixel's
    interval and each source frame interval. This avoids introducing synthetic
    background at the right edge when the frame count is not evenly divisible by
    the widget width.

    When a bin contains any labeled (non-background) frames, the background
    frames in that bin are excluded from the color blend so they do not dilute
    the label color. The blend is then normalized by the total non-background
    weight rather than the full bin width. Bins that are entirely background
    retain the normal background color.

    Color blending is performed in linear light space (sRGB gamma-decoded before
    averaging, re-encoded afterward) so that proportional mixing corresponds to
    actual light output. Alpha is averaged linearly and is not gamma-corrected.

    Args:
        labels: Integer class-index array (0 = unlabeled/background, 1+ = class indices).
        lut: RGBA lookup table of shape ``(n_classes, 4)`` mapping class indices to colors.
        size: Desired output length, typically the widget width in pixels.

    Returns:
        RGBA array of shape ``(size, 4)`` with dtype ``uint8``.
    """
    if size <= 0 or labels.size == 0:
        return cast(npt.NDArray[np.uint8], np.zeros((0, 4), dtype=np.uint8))

    n_classes = len(lut)
    n_frames = labels.size
    edges = np.linspace(0.0, float(n_frames), num=size + 1, dtype=np.float64)
    counts = np.zeros((size, n_classes), dtype=np.float32)

    for i in range(size):
        start = edges[i]
        end = edges[i + 1]
        if end <= start:
            continue

        first = math.floor(start)
        last = math.ceil(end)

        for frame in range(first, last):
            frame_start = float(frame)
            frame_end = frame_start + 1.0
            overlap = min(end, frame_end) - max(start, frame_start)
            if overlap <= 0.0 or frame < 0 or frame >= n_frames:
                continue

            label = int(labels[frame])
            if 0 <= label < n_classes:
                counts[i, label] += overlap

    bin_widths = (edges[1:] - edges[:-1]).astype(np.float32)

    # When labeled frames are present, exclude background so it doesn't dilute the color.
    non_bg_weight = counts[:, 1:].sum(axis=1)
    has_labels = non_bg_weight > 0
    effective_counts = counts.copy()
    effective_counts[has_labels, 0] = 0.0
    normalizer = np.where(has_labels, non_bg_weight, bin_widths)

    # Convert LUT to linear light space for perceptually correct blending.
    lut_float: npt.NDArray[np.float32] = lut.astype(np.float32) / np.float32(255.0)
    lut_linear: npt.NDArray[np.float32] = lut_float.copy()
    lut_linear[:, :3] = _srgb_to_linear(lut_float[:, :3])

    # Blend in linear space, then re-encode RGB to sRGB; alpha stays linear.
    linear_colors = (effective_counts @ lut_linear) / normalizer[:, np.newaxis]
    result = linear_colors.copy()
    result[:, :3] = _linear_to_srgb(linear_colors[:, :3])

    return (result * 255.0).clip(0, 255).astype(np.uint8)


class TimelineLabelWidget(QWidget):
    """Widget that shows a "zoomed out" overview of labels for the entire video.

    Because each pixel along the width ends up representing multiple frames,
    you can't see fine detail, but you can see where manual labels have been
    applied. This can be useful for seeking through the video to a location of
    labeling.

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.
    """

    COLOR_LUT: np.ndarray = np.array(
        [
            BACKGROUND_COLOR.getRgb(),
            NOT_BEHAVIOR_COLOR.getRgb(),
            BEHAVIOR_COLOR.getRgb(),
        ],
        dtype=np.uint8,
    )
    _BAR_HEIGHT = 7  # height of the bar in pixels
    _BAR_PADDING = 2  # padding around the bar in pixels
    _WINDOW_SIZE = 100  # number of frames to show on either side of the current frame

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._bar_height = self._BAR_HEIGHT
        self._bar_padding = self._BAR_PADDING
        self._height = self._bar_height + 2 * self._bar_padding
        self._window_size = self._WINDOW_SIZE
        self._frames_in_view = 2 * self._window_size + 1

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._labels: npt.NDArray[np.int16] | None = None
        self._color_lut: npt.NDArray[np.uint8] = self.COLOR_LUT

        self._search_results: list[SearchHit] = []

        self._float_bin_size: float = 0.0
        self._pixmap: QPixmap | None = None

        self._current_frame = 0
        self._num_frames = 0

    def set_color_lut(self, lut: npt.NDArray[np.uint8]) -> None:
        """Replace the color lookup table used to render label frames.

        In binary mode this is never called and the class-level ``COLOR_LUT``
        is used.  In multi-class mode ``StackedTimelineWidget`` calls this with
        the per-behavior palette produced by
        :func:`jabs.ui.colors.build_multiclass_color_lut`.

        Args:
            lut: RGBA array of shape ``(N, 4)`` mapping class indices to colors.
        """
        self._color_lut = lut
        self._update_bar()
        self.update()

    def sizeHint(self) -> QSize:
        """Return the recommended size for the widget.

        Returns:
            QSize: The preferred size for the timeline label widget.
        """
        return QSize(400, self._height)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resize events.

        Recalculates scaling and updates the bar to match the new size.

        Args:
            event (QResizeEvent): The resize event.
        """
        if self._num_frames == 0:
            return

        self._update_scale()
        self._update_bar()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Render the timeline label bar and highlight the current frame.

        Args:
            event (QPaintEvent): The paint event.
        """
        widget_width = self.size().width()
        if widget_width <= 0 or self._num_frames == 0:
            return

        if self._pixmap is None or self._pixmap.width() != widget_width:
            self._float_bin_size = self._num_frames / widget_width
            self._update_bar()

        if self._pixmap is None:
            return

        fbs = self._num_frames / widget_width

        qp = QPainter(self)

        # Highlight the window around the current position
        mapped_position = int(self._current_frame / fbs)
        window_px = int(self._frames_in_view / fbs)
        window_half_px = int(self._window_size / fbs)
        highlight_start = mapped_position - window_half_px

        qp.setPen(QPen(POSITION_MARKER_COLOR, 1, Qt.PenStyle.SolidLine))
        qp.setBrush(QBrush(POSITION_MARKER_COLOR, Qt.BrushStyle.Dense4Pattern))
        qp.drawRect(highlight_start, 0, window_px, self.size().height() - 1)

        # Draw the label bar (fills the full widget width)
        qp.drawPixmap(0, 0, self._pixmap)

        qp.setPen(QPen(Qt.GlobalColor.green, 1, Qt.PenStyle.SolidLine))
        qp.setBrush(QBrush(Qt.GlobalColor.green, Qt.BrushStyle.SolidPattern))
        center_y = self.size().height() // 2
        diamond_w = self.size().height() // 8
        diamond_h = self.size().height() // 8
        for hit in self._search_results:
            start_pos = int(hit.start_frame / fbs)
            end_pos = int((hit.end_frame + 1) / fbs)
            qp.drawLine(start_pos, center_y, end_pos, center_y)
            qp.drawPolygon(diamond_at(start_pos, center_y, diamond_w, diamond_h))
            qp.drawPolygon(diamond_at(end_pos, center_y, diamond_w, diamond_h))

    def set_labels(self, labels: npt.NDArray[np.int16]) -> None:
        """Load and display a new label track.

        ``labels`` must already be a direct LUT-index array: binary callers
        use :func:`.label_overview_util.track_labels_to_lut_indices` to shift
        a ``TrackLabels`` before calling; multi-class callers pass the array
        from ``VideoLabels.build_multiclass_label_array`` directly.

        Args:
            labels: Class-index array of shape ``(n_frames,)`` with dtype ``int16``.
        """
        self._labels = labels
        self.update_labels()

    def set_search_results(self, search_results: list[SearchHit]) -> None:
        """Set the search results for the widget.

        Args:
            search_results (list[SearchHit]): List of SearchHit objects to display.
        """
        self._search_results = search_results

    @Slot(int)
    def set_current_frame(self, current_frame: int) -> None:
        """Set the current frame to highlight in the timeline.

        Args:
            current_frame (int): The index of the current frame.
        """
        self._current_frame = current_frame
        self.update()

    def set_num_frames(self, num_frames: int) -> None:
        """Set the total number of frames in the video.

        Args:
            num_frames (int): The number of frames.
        """
        self._num_frames = num_frames
        self._update_scale()
        self._update_bar()

    def update_labels(self) -> None:
        """Update and redraw the timeline bar to reflect the current label data.

        Regenerates the color representation of the labels and updates the widget display.
        Should be called whenever the label data changes to keep the visualization in sync.
        """
        self._update_bar()
        self.update()

    def reset(self) -> None:
        """Reset the widget to its initial state.

        Clears any loaded labels and recalculates the scale, preparing the widget for new data.
        """
        self._labels = None
        self._color_lut = self.COLOR_LUT
        self._pixmap = None
        self._update_scale()

    def _update_bar(self) -> None:
        """Downsample the label array and update the bar pixmap for display.

        Converts the current labels into a color bar using proportional color blending,
        then updates the internal pixmap for efficient rendering. The pixmap fills the
        full widget width with no padding. When no labels are loaded, draws a solid
        background bar so the widget is visible even before predictions are run.
        """
        width = self.size().width()
        height = self.size().height()

        if width <= 0 or height <= 0:
            return

        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)

        if self._labels is None:
            bg = self._color_lut[0]
            painter = QPainter(pixmap)
            painter.fillRect(
                0,
                self._bar_padding,
                width,
                self._bar_height,
                QColor(int(bg[0]), int(bg[1]), int(bg[2]), int(bg[3])),
            )
            painter.end()
            self._pixmap = pixmap
            return

        colors = _downsample_to_size(self._labels, self._color_lut, width)
        if colors.size == 0:
            return

        color_bar = np.repeat(colors[np.newaxis, :, :], self._bar_height, axis=0)

        img = QImage(
            color_bar.data,
            color_bar.shape[1],
            color_bar.shape[0],
            QImage.Format.Format_RGBA8888,
        )
        painter = QPainter(pixmap)
        painter.drawImage(0, self._bar_padding, img)
        painter.end()
        self._pixmap = pixmap

    def _update_scale(self) -> None:
        """Recalculate the floating-point bin size for the timeline bar.

        Determines how many frames each horizontal pixel represents based on the
        widget width and total frame count. Content fills the full widget width.
        Resets to 0.0 (disabling drawing) when either dimension is unavailable.
        """
        width = self.size().width()

        if width and self._num_frames:
            self._float_bin_size = self._num_frames / width
        else:
            self._float_bin_size = 0.0
            self._pixmap = None
