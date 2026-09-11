"""Qt drawing primitives for an identity's segmentation contours.

Drawn by the on-screen
:class:`~jabs.ui.player_widget.overlays.segmentation_overlay.SegmentationOverlay` at the
scaled and cropped display resolution, and by the frame and video exports at the video's
native resolution.

This replaces the cv2 drawing that used to bake contours into the frame as it was
decoded. Painting them as an overlay instead means toggling the contours redraws the
frame already on screen rather than needing it decoded again, and the player and the
exports now share one implementation, the way they already share the pose skeleton.
"""

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PySide6 import QtCore, QtGui

from .scaling import native_overlay_scale

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

# Contours of the active identity are red, every other identity's are blue. These are
# the colors the cv2 drawing used, converted from its BGR tuples.
SEGMENTATION_ACTIVE_COLOR = QtGui.QColor(255, 0, 0)
SEGMENTATION_INACTIVE_COLOR = QtGui.QColor(0, 0, 255)

# Contour line width at the reference frame size, matching the single pixel cv2 drew.
_BASE_LINE_WIDTH = 1

# Padding value for unused contour points and unused contour slots in a pose file's
# segmentation data.
_PADDING = -1

# The pose file's first segmentation version. Earlier files carry no contours at all.
_MIN_SEGMENTATION_POSE_VERSION = 6


def native_segmentation_line_width(width: int, height: int) -> int:
    """Return the contour line width for drawing at native video resolution.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Line width in pixels, scaled from the frame size by
        :func:`~jabs.overlay_drawing.scaling.native_overlay_scale`.
    """
    return max(_BASE_LINE_WIDTH, round(_BASE_LINE_WIDTH * native_overlay_scale(width, height)))


def identity_contours(
    pose: "PoseEstimation", frame_index: int, identity: int
) -> list[npt.NDArray[np.int_]]:
    """Return one identity's segmentation contours for a frame, padding removed.

    A pose file stores a fixed number of contour slots per identity, each with a fixed
    number of points, padded with ``-1`` where an animal needs fewer. Unused slots and
    unused points are dropped here so callers only see real contours.

    Args:
        pose: Pose estimation for the video.
        frame_index: Frame to return contours for.
        identity: Identity to return contours for.

    Returns:
        A list of ``(n_points, 2)`` arrays of ``(x, y)`` image coordinates, empty when
        this identity has no contours on this frame, or when the pose file has no
        segmentation data at all.
    """
    if pose.format_major_version < _MIN_SEGMENTATION_POSE_VERSION:
        return []

    contours = pose.get_segmentation_data_per_frame(frame_index, identity)
    if contours is None:
        return []

    trimmed = []
    for contour in contours:
        points = contour[np.all(contour != _PADDING, axis=1), :]
        if len(points) > 0:
            trimmed.append(points.astype(int))
    return trimmed


def draw_identity_segmentation(
    painter: QtGui.QPainter,
    pose: "PoseEstimation",
    frame_index: int,
    identity: int,
    *,
    to_output: Callable[[float, float], tuple[int, int] | None],
    line_width: int,
    active: bool,
) -> None:
    """Draw one identity's segmentation contours with ``painter``.

    Args:
        painter: The painter to draw with.
        pose: Pose estimation data for the video.
        frame_index: Frame to draw the contours for.
        identity: Identity whose contours are drawn.
        to_output: Maps an image-space ``(x, y)`` to the painter's coordinate space, or
            returns ``None`` to skip a point (e.g. a point outside a display crop).
        line_width: Width of the contour outline in pixels.
        active: Whether this is the active identity, which is drawn in a different
            color. An export has no active identity and draws every identity the same.
    """
    pen = QtGui.QPen(SEGMENTATION_ACTIVE_COLOR if active else SEGMENTATION_INACTIVE_COLOR)
    pen.setWidth(line_width)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    for contour in identity_contours(pose, frame_index, identity):
        mapped = [to_output(float(x), float(y)) for x, y in contour]
        for run, complete in _visible_runs(mapped):
            if len(run) == 1:
                # A contour can be a single point, and cv2 drew that as a pixel.
                painter.drawPoint(QtCore.QPoint(*run[0]))
                continue

            polygon = QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in run])
            if complete:
                # A contour is a closed shape, so the last point joins the first. Only
                # safe to close a run that is the whole contour: closing a run that a
                # display crop cut short would draw a chord across the cropped region.
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)


def _visible_runs(
    mapped: list[tuple[int, int] | None],
) -> Iterator[tuple[list[tuple[int, int]], bool]]:
    """Split mapped contour points into runs of consecutive visible points.

    Yields ``(run, complete)`` for each non-empty run, where ``complete`` says the run
    is the entire contour, with nothing dropped by a crop. Single-point runs are
    included: a one-point contour is degenerate but a pose file can hold one, and cv2
    used to draw it as a pixel.
    """
    run: list[tuple[int, int]] = []
    dropped = False
    for point in mapped:
        if point is None:
            dropped = True
            if run:
                yield run, False
            run = []
            continue
        run.append(point)
    if run:
        yield run, not dropped
