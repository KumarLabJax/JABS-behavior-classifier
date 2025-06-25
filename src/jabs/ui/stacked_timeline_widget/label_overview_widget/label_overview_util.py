from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QPen, QPolygon

from ...colors import SEARCH_HIT_COLOR


def diamond_at(x, y, w, h):
    """Create a diamond shape polygon centered at (x, y) with width w and height h.

    Args:
        x (int): The x-coordinate of the center.
        y (int): The y-coordinate of the center.
        w (int): The width of the diamond.
        h (int): The height of the diamond.

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
    center_y = bar_height // 2
    diamond_w = bar_height // 8
    diamond_h = bar_height // 8

    for hit in search_results:
        rel_start_frame = hit.start_frame - start
        rel_end_frame = hit.end_frame - start + 1
        bounded_rel_start = max(0, rel_start_frame)
        bounded_rel_end = min(window_frames_total, rel_end_frame)

        if bounded_rel_start > rel_end_frame or bounded_rel_end < rel_start_frame:
            # skip search hits that are completely out of bounds
            continue

        start_pos = offset + (bounded_rel_start * frame_width)
        end_pos = offset + (bounded_rel_end * frame_width)
        qp.drawLine(start_pos, center_y, end_pos, center_y)

        if bounded_rel_start == rel_start_frame:
            qp.drawPolygon(diamond_at(start_pos, center_y, diamond_w, diamond_h))

        if bounded_rel_end == rel_end_frame:
            qp.drawPolygon(diamond_at(end_pos, center_y, diamond_w, diamond_h))
