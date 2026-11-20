"""Qt drawing primitives for the arena landmarks a pose file can carry.

Landmarks are the static objects a pose file records for the arena itself: its
corners, one or more lixit, and a food hopper. They do not move, so they are drawn
from the pose file's static objects rather than per frame. Drawn by the on-screen
:class:`~jabs.ui.player_widget.overlays.landmark_overlay.LandmarkOverlay` and by the
frame and video exports in :mod:`jabs.video_export`.

Replaces the cv2 drawing that used to bake landmarks into the frame as it was decoded.
The colors are the ones that drawing used, converted from its BGR tuples.
"""

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui

from jabs.pose_estimation import PoseEstimation

from .mapping import visible_runs
from .scaling import native_overlay_scale

LANDMARK_CORNER_COLOR = QtGui.QColor(252, 135, 0)
LANDMARK_LIXIT_COLOR = QtGui.QColor(0, 222, 215)
LANDMARK_HOPPER_COLOR = QtGui.QColor(0, 255, 0)

# Base sizes at the player's display scale; callers scale these with the zoom.
LANDMARK_POINT_RADIUS = 2
LANDMARK_LINE_WIDTH = 1

# A lixit is recorded either as one keypoint or as three.
_LIXIT_KEYPOINTS_3 = 3

# The static objects draw_landmarks knows how to draw.
_DRAWN_STATIC_OBJECTS = ("corners", "lixit", "food_hopper")


def pose_has_landmarks(pose: PoseEstimation) -> bool:
    """Return whether a pose file carries any arena landmarks to draw.

    Args:
        pose: Pose estimation to check.

    Returns:
        True if the pose file records at least one point for the arena corners, a
        lixit, or the food hopper. An empty dataset does not count: it would enable
        an overlay that draws nothing.
    """
    return any(_has_points(pose.static_objects.get(name)) for name in _DRAWN_STATIC_OBJECTS)


def _has_points(static_object: object) -> bool:
    """Return whether a static object is an array holding at least one point."""
    return isinstance(static_object, np.ndarray) and static_object.size > 0


def native_landmark_sizes(width: int, height: int) -> tuple[int, int]:
    """Return ``(point_radius, line_width)`` for drawing landmarks at native resolution.

    Sizes come from :func:`~jabs.overlay_drawing.scaling.native_overlay_scale`, which
    explains why they grow sub-linearly with the frame.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Tuple of ``(point_radius, line_width)`` in pixels.
    """
    scale = native_overlay_scale(width, height)
    return (
        max(LANDMARK_POINT_RADIUS, round(LANDMARK_POINT_RADIUS * scale)),
        max(LANDMARK_LINE_WIDTH, round(LANDMARK_LINE_WIDTH * scale)),
    )


def draw_landmarks(
    painter: QtGui.QPainter,
    pose: PoseEstimation,
    *,
    to_output: Callable[[float, float], tuple[int, int] | None],
    point_radius: int,
    line_width: int,
) -> None:
    """Draw whichever landmarks the pose file carries, with ``painter``.

    A pose file that records none of them draws nothing: the static objects are
    optional.

    Args:
        painter: The painter to draw with.
        pose: Pose estimation data for the video.
        to_output: Maps an image-space ``(x, y)`` to the painter's coordinate space, or
            returns ``None`` to skip a point (e.g. a point outside a display crop).
        point_radius: Radius of the corner and lixit markers, in pixels.
        line_width: Width of the food hopper outline, in pixels.
    """
    static_objects = pose.static_objects

    corners = static_objects.get("corners")
    if corners is not None:
        _draw_points(
            painter,
            [(corners[i, 0], corners[i, 1]) for i in range(corners.shape[0])],
            to_output,
            LANDMARK_CORNER_COLOR,
            point_radius,
        )

    lixit = static_objects.get("lixit")
    if lixit is not None:
        if pose.num_lixit_keypoints == _LIXIT_KEYPOINTS_3:
            points = [
                (lixit[i, j, 0], lixit[i, j, 1])
                for i in range(lixit.shape[0])
                for j in range(_LIXIT_KEYPOINTS_3)
            ]
        else:
            # Older pose files record a lixit as a single keypoint.
            points = [(lixit[i, 0], lixit[i, 1]) for i in range(lixit.shape[0])]
        _draw_points(painter, points, to_output, LANDMARK_LIXIT_COLOR, point_radius)

    hopper = static_objects.get("food_hopper")
    if hopper is not None:
        pen = QtGui.QPen(LANDMARK_HOPPER_COLOR)
        pen.setWidth(line_width)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        mapped = [to_output(float(x), float(y)) for x, y in hopper]
        for run, complete in visible_runs(mapped, closed=True):
            if len(run) < 2:
                # A lone visible corner has no edge to draw: the edges reaching it
                # both cross the crop boundary.
                continue
            polygon = QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in run])
            if complete:
                # The hopper is a closed shape, so the last corner joins the first.
                # Only safe to close a run that is the whole outline: closing a run a
                # crop cut short would draw a chord across the cropped region.
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)


def _draw_points(
    painter: QtGui.QPainter,
    points: list[tuple[float, float]],
    to_output: Callable[[float, float], tuple[int, int] | None],
    color: QtGui.QColor,
    radius: int,
) -> None:
    """Draw a filled marker at each point that maps into view."""
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    for x, y in points:
        out = to_output(float(x), float(y))
        if out is None:
            continue
        painter.drawEllipse(QtCore.QPoint(out[0], out[1]), radius, radius)
