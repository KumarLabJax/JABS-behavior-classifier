from random import Random

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QPen, QPolygon

from ...colors import SEARCH_HIT_COLOR


def diamond_at(x: float, y: float, w: float, h: float) -> QPolygon:
    """Create a diamond shape polygon centered at (x, y) with width w and height h.

    Args:
        x (float): The x-coordinate of the center.
        y (float): The y-coordinate of the center.
        w (float): The width of the diamond.
        h (float): The height of the diamond.

    Returns:
        QPolygon: A polygon representing the diamond shape.
    """
    return QPolygon(
        [
            QPoint(x, y - h),  # top
            QPoint(x + w, y),  # right
            QPoint(x, y + h),  # bottom
            QPoint(x - w, y),  # left
        ]
    )


def render_search_hits(
    qp, search_results, offset, start, frame_width, bar_height, window_frames_total
):
    """Render search hits on the given QPainter.

    Args:
        qp (QPainter): The QPainter to draw on.
        search_results (list): List of search hit results.
        offset (int): The offset for drawing.
        start (int): The starting frame index for the current view.
        frame_width (int): The width of each frame.
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

        start_pos = offset + (bounded_rel_start * frame_width)
        end_pos = offset + (bounded_rel_end * frame_width)
        qp.drawLine(start_pos, y_pos, end_pos, y_pos)

        if bounded_rel_start == rel_start_frame:
            qp.drawPolygon(diamond_at(start_pos, y_pos, diamond_w, diamond_h))

        if bounded_rel_end == rel_end_frame:
            qp.drawPolygon(diamond_at(end_pos, y_pos, diamond_w, diamond_h))
