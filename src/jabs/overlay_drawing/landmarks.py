"""Qt drawing primitives for the arena landmarks a pose file can carry.

Landmarks are the static objects a pose file records for the arena itself: its
corners, one or more lixit, and a food hopper. They do not move, so they are drawn
from the pose file's static objects rather than per frame. Drawn by the on-screen
:class:`~jabs.ui.player_widget.overlays.landmark_overlay.LandmarkOverlay`.

Replaces the cv2 drawing that used to bake landmarks into the frame as it was decoded.
The colors are the ones that drawing used, converted from its BGR tuples.
"""

from collections.abc import Callable

from PySide6 import QtCore, QtGui

from jabs.pose_estimation import PoseEstimation

LANDMARK_CORNER_COLOR = QtGui.QColor(252, 135, 0)
LANDMARK_LIXIT_COLOR = QtGui.QColor(0, 222, 215)
LANDMARK_HOPPER_COLOR = QtGui.QColor(0, 255, 0)

# Base sizes at the player's display scale; callers scale these with the zoom.
LANDMARK_POINT_RADIUS = 2
LANDMARK_LINE_WIDTH = 1

# A lixit is recorded either as one keypoint or as three.
_LIXIT_KEYPOINTS_3 = 3


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
    optional, and older pose versions have no concept of them.

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
        mapped = [to_output(float(x), float(y)) for x, y in hopper]
        if all(point is not None for point in mapped):
            pen = QtGui.QPen(LANDMARK_HOPPER_COLOR)
            pen.setWidth(line_width)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in mapped]))


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
