"""Qt drawing primitives for a single identity's pose skeleton."""

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui

from jabs.core.utils.pose_util import gen_line_fragments
from jabs.pose_estimation import PoseEstimation

from .colors import KEYPOINT_COLOR_MAP

# Color for the pose line segments (white, semi-transparent).
LINE_SEGMENT_COLOR = QtGui.QColor(255, 255, 255, 128)

# Base keypoint circle radius at a 1:1 (zoom 1.0) display scale.
KEYPOINT_SIZE = 3

# Alpha applied to keypoints/line segments of non-active identities.
_INACTIVE_KEYPOINT_ALPHA = 96


# Frame size the base marker sizes below are calibrated against, taken from 800x800
# open-field footage where a 3px radius reads well.
_REFERENCE_FRAME = 800
_BASE_KEYPOINT_SIZE = 3
_BASE_LINE_WIDTH = 2


def native_pose_sizes(width: int, height: int) -> tuple[int, int]:
    """Return ``(keypoint_size, line_width)`` for drawing pose at native video resolution.

    Markers grow with the frame so the overlay stays legible across resolutions, but
    they grow *sub-linearly* - as the square root of the frame's larger dimension.

    Scaling linearly (the previous behavior) keeps a marker at a constant fraction of
    the frame, which sounds right but is not what the eye judges. Measured on real
    footage: a mouse in 1080p home-cage video is about four times longer in pixels
    than one in 800x800 open-field video, so linear scaling produced 16px dots that
    covered the animal, while the same formula gave a well-judged 6px dot on the
    smaller frame. What reads badly is a marker's absolute size, not its share of the
    frame.

    Sizing markers from the animal's own extent was considered and rejected: matching
    the 800x800 proportions (dot diameter around 6% of body length) would make the
    1080p dots *larger* still, at roughly 24px.

    ``keypoint_size`` is a radius, so a dot's diameter is twice this value.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Tuple of ``(keypoint_size, line_width)`` in pixels.
    """
    scale = (max(width, height, 1) / _REFERENCE_FRAME) ** 0.5
    keypoint_size = max(_BASE_KEYPOINT_SIZE, round(_BASE_KEYPOINT_SIZE * scale))
    line_width = max(_BASE_LINE_WIDTH, round(_BASE_LINE_WIDTH * scale))
    return keypoint_size, line_width


def draw_identity_pose(
    painter: QtGui.QPainter,
    pose: PoseEstimation,
    frame_index: int,
    identity: int,
    *,
    to_output: Callable[[float, float], tuple[int, int] | None],
    keypoint_size: int,
    line_width: int,
    active: bool,
) -> None:
    """Draw one identity's pose skeleton and keypoints with ``painter``.

    Args:
        painter: The painter to draw with.
        pose: Pose estimation data for the video.
        frame_index: Frame to draw the pose for.
        identity: Identity whose pose is drawn.
        to_output: Maps an image-space ``(x, y)`` to the painter's coordinate space, or
            returns ``None`` to skip a point (e.g. a point outside a display crop).
        keypoint_size: Radius (pixels) of each keypoint circle.
        line_width: Width (pixels) of the skeleton line segments.
        active: Whether this is the active identity. Inactive identities are drawn more
            translucent.
    """
    points, mask = pose.get_points(frame_index, identity)
    if points is None or mask is None:
        return

    line_color = QtGui.QColor(LINE_SEGMENT_COLOR)
    if not active:
        line_color.setAlpha(line_color.alpha() // 3)
    pen = QtGui.QPen(line_color)
    pen.setWidth(line_width)
    painter.setPen(pen)

    for seg in gen_line_fragments(pose.get_connected_segments(), np.flatnonzero(mask == 0)):
        segment_points = [to_output(p[0], p[1]) for p in points[seg]]
        segment_points = [pt for pt in segment_points if pt is not None]
        if len(segment_points) >= 2:
            for i in range(len(segment_points) - 1):
                painter.drawLine(
                    segment_points[i][0],
                    segment_points[i][1],
                    segment_points[i + 1][0],
                    segment_points[i + 1][1],
                )

    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    for keypoint in PoseEstimation.KeypointIndex:
        point_index = keypoint.value
        if not mask[point_index]:
            continue
        out = to_output(points[point_index][0], points[point_index][1])
        if out is None:
            continue
        color = KEYPOINT_COLOR_MAP[keypoint]
        if not active:
            color = QtGui.QColor(color)
            color.setAlpha(_INACTIVE_KEYPOINT_ALPHA)
        painter.setBrush(color)
        painter.drawEllipse(QtCore.QPoint(out[0], out[1]), keypoint_size, keypoint_size)
