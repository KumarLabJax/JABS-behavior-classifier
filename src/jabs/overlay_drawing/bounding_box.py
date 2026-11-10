"""Qt drawing primitives for an identity's bounding box at native video resolution.

Drawn by the frame and video exports. The on-screen
:class:`~jabs.ui.player_widget.overlays.bounding_box.BoundingBoxOverlay` draws its own
boxes, since it also lays out clickable identity tabs with ear tag icons and colors
them from the application palette, neither of which an exported frame can use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui

from .scaling import native_overlay_scale

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

logger = logging.getLogger(__name__)

# Box and identity tab outline color.
BOUNDING_BOX_COLOR = QtGui.QColor(0, 160, 255)

# Fill behind the identity text in the tab, matching the on-screen overlay's tab.
_TAB_FILL_COLOR = QtGui.QColor(64, 64, 64, 150)
_TAB_TEXT_COLOR = QtGui.QColor(255, 255, 255)

# Base sizes, in pixels, calibrated for an 800x800 frame and scaled from there.
_BASE_LINE_WIDTH = 2
_BASE_FONT_SIZE = 14
_BASE_TAB_PADDING = 4

# The pose file's first version with bounding boxes.
_MIN_BBOX_POSE_VERSION = 8

# Latched so a full-length export logs the missing-application warning once.
_warned_without_application = False


def pose_has_bounding_boxes(pose: PoseEstimation) -> bool:
    """Return whether a pose file carries bounding boxes to draw.

    Args:
        pose: Pose estimation to check.

    Returns:
        True for a v8+ pose file that was generated with bounding boxes.
    """
    return pose.format_major_version >= _MIN_BBOX_POSE_VERSION and bool(
        getattr(pose, "has_bounding_boxes", False)
    )


def native_bounding_box_sizes(width: int, height: int) -> tuple[int, int, int]:
    """Return ``(line_width, font_size, tab_padding)`` for drawing at native resolution.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Sizes in pixels, scaled from the frame size by
        :func:`~jabs.overlay_drawing.scaling.native_overlay_scale`.
    """
    scale = native_overlay_scale(width, height)
    return (
        max(_BASE_LINE_WIDTH, round(_BASE_LINE_WIDTH * scale)),
        max(_BASE_FONT_SIZE, round(_BASE_FONT_SIZE * scale)),
        max(_BASE_TAB_PADDING, round(_BASE_TAB_PADDING * scale)),
    )


def draw_identity_bounding_box(
    painter: QtGui.QPainter,
    pose: PoseEstimation,
    frame_index: int,
    identity: int,
    *,
    width: int,
    height: int,
) -> None:
    """Draw one identity's bounding box, with its display ID in a tab above it.

    Coordinates are native image coordinates, so this is for a painter on a frame at
    the video's own resolution. A missing box (stored as ``-1`` corners) or a degenerate
    one is skipped. The ID tab needs Qt's font database, so it is skipped, with a
    warning logged once, when there is no ``QGuiApplication``; the box is still drawn.

    Args:
        painter: The painter to draw with.
        pose: Pose estimation for the video; boxes need a v8+ pose file.
        frame_index: Frame to draw the box for.
        identity: Identity whose box is drawn.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """
    if not pose_has_bounding_boxes(pose):
        return
    bboxes = pose.get_bounding_boxes(identity)  # type: ignore[attr-defined]
    if bboxes is None:
        return

    # bbox format is [upper_left_x, upper_left_y], [lower_right_x, lower_right_y]
    # A missing box is stored as [-1, -1], [-1, -1], which this degenerate check skips.
    bbox = np.asarray(bboxes[frame_index], dtype=float)
    if not np.all(np.isfinite(bbox)) or not (bbox[1, 0] > bbox[0, 0] and bbox[1, 1] > bbox[0, 1]):
        return
    frame_rect = QtCore.QRectF(0, 0, width, height)
    box = QtCore.QRectF(
        QtCore.QPointF(bbox[0, 0], bbox[0, 1]), QtCore.QPointF(bbox[1, 0], bbox[1, 1])
    ).intersected(frame_rect)
    if box.isEmpty():
        return

    line_width, font_size, tab_padding = native_bounding_box_sizes(width, height)
    painter.save()
    try:
        painter.setPen(QtGui.QPen(BOUNDING_BOX_COLOR, line_width))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(box)
        _draw_identity_tab(
            painter,
            pose.identity_index_to_display(identity),
            box,
            frame_rect,
            line_width=line_width,
            font_size=font_size,
            padding=tab_padding,
        )
    finally:
        painter.restore()


def _draw_identity_tab(
    painter: QtGui.QPainter,
    text: str,
    box: QtCore.QRectF,
    frame_rect: QtCore.QRectF,
    *,
    line_width: int,
    font_size: int,
    padding: int,
) -> None:
    """Draw the identity tab above the box's top-left corner, kept inside the frame."""
    if QtGui.QGuiApplication.instance() is None:
        global _warned_without_application
        if not _warned_without_application:
            logger.warning("No QGuiApplication: exporting bounding boxes without identity tabs")
            _warned_without_application = True
        return

    font = QtGui.QFont(painter.font())
    font.setPixelSize(font_size)
    metrics = QtGui.QFontMetrics(font)
    tab_w = metrics.horizontalAdvance(text) + 2 * padding
    tab_h = metrics.height() + 2 * padding
    if tab_w > frame_rect.width() or tab_h > frame_rect.height():
        return

    tab_x = max(frame_rect.left(), min(box.left(), frame_rect.right() - tab_w))
    # prefer above the box; otherwise sit just inside its top edge
    tab_y = box.top() - tab_h if box.top() - tab_h >= frame_rect.top() else box.top()
    tab_y = min(tab_y, frame_rect.bottom() - tab_h)
    tab = QtCore.QRectF(tab_x, tab_y, tab_w, tab_h)

    painter.setBrush(_TAB_FILL_COLOR)
    painter.setPen(QtGui.QPen(BOUNDING_BOX_COLOR, line_width))
    painter.drawRect(tab)
    painter.setFont(font)
    painter.setPen(_TAB_TEXT_COLOR)
    painter.drawText(tab, QtCore.Qt.AlignmentFlag.AlignCenter, text)
