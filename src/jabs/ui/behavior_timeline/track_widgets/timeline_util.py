import math
from dataclasses import dataclass
from random import Random

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QImage, QPainter, QPen, QPolygonF

from jabs.project import TrackLabels

from ...colors import SEARCH_HIT_COLOR

#: Number of frames shown on each side of the current frame by the timeline's
#: sliding-window widgets, so a window spans ``2 * DETAIL_WINDOW_SIZE + 1``
#: frames.
#:
#: The detail bars, the overview bar's window highlight, and the frame ruler all
#: describe the same window, so they have to agree on its size: the overview bar
#: draws the box marking which slice of the video the detail bars are showing,
#: and the ruler numbers the frames inside it. Keeping one definition here means
#: making the window configurable later only has to change one place.
#:
#: Unrelated to ``jabs.feature_extraction.DEFAULT_WINDOW_SIZE``, which is the
#: number of frames used to compute window features.
DETAIL_WINDOW_SIZE = 100


@dataclass(frozen=True)
class WindowGeometry:
    """Frame and pixel extents of a detail bar's sliding window.

    A detail bar centers a fixed-size window on the current frame, so near the
    start or end of a video part of that window falls outside the video. The
    out-of-bounds frames are drawn with a padding pattern and the rest with the
    bar's content, which means every detail bar needs the same set of numbers:
    which frames are in bounds, how many padding frames precede them, and where
    the in-bounds region lands in widget pixels.

    Attributes:
        start: First frame of the window. Negative when the window extends past
            the beginning of the video.
        end: Last frame of the window, inclusive. May exceed the last frame of
            the video.
        slice_start: First in-bounds frame, inclusive.
        slice_end: Last in-bounds frame, inclusive. ``slice_start - 1`` when no
            frame of the window is in bounds.
        n_in_bounds: Number of in-bounds frames; ``0`` or less when none are.
        n_start_pad: Number of padding frames before the in-bounds frames.
        x0: Left pixel edge of the in-bounds region.
        width_px: Pixel width of the in-bounds region.
    """

    start: int
    end: int
    slice_start: int
    slice_end: int
    n_in_bounds: int
    n_start_pad: int
    x0: int
    width_px: int

    @property
    def has_in_bounds_content(self) -> bool:
        """Whether the window has in-bounds frames that occupy at least one pixel."""
        return self.n_in_bounds > 0 and self.width_px > 0


def window_geometry(
    current_frame: int, window_size: int, num_frames: int, frame_width: float
) -> WindowGeometry:
    """Compute the extents of the window a detail bar is about to paint.

    Args:
        current_frame: Frame the window is centered on.
        window_size: Number of frames on each side of ``current_frame``, so the
            window spans ``2 * window_size + 1`` frames.
        num_frames: Number of frames in the video.
        frame_width: Pixels per frame; fractional, since the window is scaled to
            the widget width.

    Returns:
        The window's frame and pixel extents.
    """
    start = current_frame - window_size
    end = current_frame + window_size

    slice_start = max(start, 0)
    slice_end = min(end, num_frames - 1)
    n_start_pad = max(0, -start)

    n_in_bounds = slice_end - slice_start + 1
    x0 = math.floor(n_start_pad * frame_width)
    x1 = math.floor((n_start_pad + n_in_bounds) * frame_width)

    return WindowGeometry(
        start=start,
        end=end,
        slice_start=slice_start,
        slice_end=slice_end,
        n_in_bounds=n_in_bounds,
        n_start_pad=n_start_pad,
        x0=x0,
        width_px=x1 - x0,
    )


def frame_pixel_widths(geometry: WindowGeometry, frame_width: float) -> npt.NDArray[np.int_]:
    """Compute how many pixels each in-bounds frame occupies.

    Widths come from the difference between consecutive floored pixel edges,
    so they sum to :attr:`WindowGeometry.width_px` and no pixel is painted
    twice or skipped. Individual frames get a width of zero when the window
    holds more frames than the widget has pixels.

    Args:
        geometry: Window extents, as returned by :func:`window_geometry`. Must
            have in-bounds content.
        frame_width: Pixels per frame; the same value passed to
            :func:`window_geometry`.

    Returns:
        Array of shape ``(geometry.n_in_bounds,)`` holding each frame's width
        in pixels.
    """
    edges = np.arange(
        geometry.n_start_pad,
        geometry.n_start_pad + geometry.n_in_bounds + 1,
        dtype=np.float64,
    )
    return np.diff(np.floor(edges * frame_width)).astype(int)


def draw_frame_colors(
    painter: QPainter,
    colors: npt.NDArray[np.uint8],
    geometry: WindowGeometry,
    frame_width: float,
    bar_height: int,
) -> None:
    """Draw one RGBA color per in-bounds frame as the bar's content image.

    Each frame's color is expanded to the full bar height and to the frame's
    pixel width, and the result is blitted at the left edge of the in-bounds
    region.

    Args:
        painter: Active painter for the bar widget.
        colors: RGBA color per in-bounds frame, shape
            ``(geometry.n_in_bounds, 4)`` with dtype ``uint8``.
        geometry: Window extents, as returned by :func:`window_geometry`. Must
            have in-bounds content.
        frame_width: Pixels per frame; the same value passed to
            :func:`window_geometry`.
        bar_height: Height of the bar in pixels.
    """
    # shape (bar_height, n_in_bounds, 4), then (bar_height, geometry.width_px, 4)
    colors_bar = np.repeat(colors[np.newaxis, :, :], bar_height, axis=0)
    colors_bar = np.repeat(colors_bar, frame_pixel_widths(geometry, frame_width), axis=1)

    # colors_bar backs the QImage rather than being copied into it, so it has to
    # outlive the draw; it does, since drawImage renders before this returns.
    image = QImage(
        colors_bar.data,
        colors_bar.shape[1],
        colors_bar.shape[0],
        QImage.Format.Format_RGBA8888,
    )
    painter.drawImage(geometry.x0, 0, image)


def track_labels_to_lut_indices(labels: TrackLabels) -> npt.NDArray[np.int16]:
    """Convert a binary ``TrackLabels`` to a direct color-LUT index array.

    Shifts raw label values by +1 so they map to
    :data:`jabs.ui.colors.BINARY_LABEL_COLOR_LUT`, the palette the timeline
    widgets use in binary mode: NONE(-1)→0, NOT_BEHAVIOR(0)→1, BEHAVIOR(1)→2.

    Args:
        labels: Binary label track to convert.

    Returns:
        Array of shape ``(n_frames,)`` with dtype ``int16``.
    """
    return (labels.get_labels() + 1).astype(np.int16)


def binary_predictions_to_lut_indices(
    predictions: npt.NDArray[np.int8],
) -> npt.NDArray[np.int16]:
    """Convert a binary prediction array to a direct color-LUT index array.

    Shifts raw prediction values by +1 so they map to
    :data:`jabs.ui.colors.BINARY_LABEL_COLOR_LUT`: no-prediction(-1)→0,
    not-behavior(0)→1, behavior(1)→2.

    Args:
        predictions: Raw binary prediction array with values in ``{-1, 0, 1}``.

    Returns:
        Array of shape ``(n_frames,)`` with dtype ``int16``.
    """
    return (predictions + 1).astype(np.int16)


def diamond_at(x: float, y: float, w: float, h: float) -> QPolygonF:
    """Create a diamond shape polygon centered at (x, y) with width w and height h.

    Args:
        x (float): The x-coordinate of the center.
        y (float): The y-coordinate of the center.
        w (float): The width of the diamond.
        h (float): The height of the diamond.

    Returns:
        QPolygonF: A polygon representing the diamond shape.
    """
    return QPolygonF(
        [
            QPointF(x, y - h),  # top
            QPointF(x + w, y),  # right
            QPointF(x, y + h),  # bottom
            QPointF(x - w, y),  # left
        ]
    )


def render_search_hits(
    qp,
    search_results: list,
    offset: float,
    start: int,
    frame_width: float,
    bar_height: int,
    window_frames_total: int,
) -> None:
    """Render search hits on the given QPainter.

    Args:
        qp (QPainter): The QPainter to draw on.
        search_results (list): List of search hit results.
        offset (float): The x offset for drawing (0 when content fills the full widget).
        start (int): The starting frame index for the current view.
        frame_width (float): Pixels per frame (may be fractional).
        bar_height (int): The height of the bar.
        window_frames_total (int): Total number of frames in the window.
    """
    qp.setPen(QPen(SEARCH_HIT_COLOR, 1, Qt.PenStyle.SolidLine))
    qp.setBrush(QBrush(SEARCH_HIT_COLOR, Qt.BrushStyle.SolidPattern))
    diamond_w = bar_height / 8
    diamond_h = bar_height / 8

    rand = Random(str(search_results[0].start_frame) if search_results else "42")

    y_min_dist_edge = diamond_h
    y_min_neighbor_sep = diamond_h

    prev_y_pos = 0
    for i, hit in enumerate(search_results):
        rel_start_frame = hit.start_frame - start
        rel_end_frame = hit.end_frame - start + 1
        bounded_rel_start = max(0, rel_start_frame)
        bounded_rel_end = min(window_frames_total, rel_end_frame)

        # Randomly select a y position but take care not to have y overlap with the
        # previous hit. This will give an ability to distinguish between hits that
        # overlap in time.
        if i == 0:
            y_pos = rand.random() * (bar_height - y_min_dist_edge * 2) + y_min_dist_edge
        else:
            y_pos = (
                rand.random() * (bar_height - y_min_dist_edge * 2 - y_min_neighbor_sep * 2)
                + y_min_dist_edge
            )
            if y_pos > prev_y_pos - y_min_neighbor_sep:
                y_pos += y_min_neighbor_sep * 2

        prev_y_pos = y_pos

        if bounded_rel_start > rel_end_frame or bounded_rel_end < rel_start_frame:
            # skip search hits that are completely out of bounds
            continue

        start_pos = math.floor(offset + bounded_rel_start * frame_width)
        end_pos = math.floor(offset + bounded_rel_end * frame_width)
        # ensure at least 1px span so single-frame hits remain visible
        if end_pos <= start_pos:
            end_pos = start_pos + 1
        qp.drawLine(start_pos, y_pos, end_pos, y_pos)

        if bounded_rel_start == rel_start_frame:
            qp.drawPolygon(diamond_at(start_pos, y_pos, diamond_w, diamond_h))

        if bounded_rel_end == rel_end_frame:
            qp.drawPolygon(diamond_at(end_pos, y_pos, diamond_w, diamond_h))
