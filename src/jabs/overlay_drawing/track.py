"""Qt drawing primitives for an identity's movement track.

The track shows where a keypoint - the nose by default - has been and where it is
going: a short trail of past positions and a short lead of future ones, each joined by
a line. Drawn by the on-screen
:class:`~jabs.ui.player_widget.overlays.track_overlay.TrackOverlay`.

Replaces the cv2 drawing that used to bake the track into the frame as it was decoded,
so toggling it repaints the frame already on screen instead of decoding it again. The
colors are the ones that drawing used, converted from its BGR tuples.
"""

from collections.abc import Callable

import numpy as np
from PySide6 import QtCore, QtGui

from jabs.pose_estimation import PoseEstimation

from .mapping import visible_runs

# Where the animal is going, and where it has been.
TRACK_FUTURE_COLOR = QtGui.QColor(255, 61, 61)
TRACK_PAST_COLOR = QtGui.QColor(255, 135, 135)

# How far the track reaches, in frames, either side of the current one.
DEFAULT_FUTURE_FRAMES = 10
DEFAULT_PAST_FRAMES = 5

# Base sizes at the player's display scale. Past positions are drawn larger than
# future ones, as the cv2 drawing did. Callers scale these with the zoom.
TRACK_FUTURE_RADIUS = 1
TRACK_PAST_RADIUS = 2
TRACK_LINE_WIDTH = 1


def draw_identity_track(
    painter: QtGui.QPainter,
    pose: PoseEstimation,
    frame_index: int,
    identity: int,
    *,
    to_output: Callable[[float, float], tuple[int, int] | None],
    future_radius: int,
    past_radius: int,
    line_width: int,
    future_frames: int = DEFAULT_FUTURE_FRAMES,
    past_frames: int = DEFAULT_PAST_FRAMES,
    point_index: int = PoseEstimation.KeypointIndex.NOSE,
) -> None:
    """Draw one identity's past and future track with ``painter``.

    Args:
        painter: The painter to draw with.
        pose: Pose estimation data for the video.
        frame_index: The current frame, where past meets future.
        identity: Identity whose track is drawn.
        to_output: Maps an image-space ``(x, y)`` to the painter's coordinate space, or
            returns ``None`` to skip a point (e.g. a point outside a display crop).
        future_radius: Radius of each future position marker, in pixels.
        past_radius: Radius of each past position marker, in pixels.
        line_width: Width of the line joining the positions, in pixels.
        future_frames: How many frames ahead of ``frame_index`` to draw.
        past_frames: How many frames behind ``frame_index`` to draw.
        point_index: Keypoint to track. Defaults to the nose.
    """
    points, mask = pose.get_identity_poses(identity)

    # The current frame belongs to the past track, so both halves meet at it.
    future = _visible_points(
        points[frame_index : frame_index + future_frames, point_index],
        mask[frame_index : frame_index + future_frames, point_index],
        to_output,
    )
    past_start = max(frame_index - past_frames, 0)
    past = _visible_points(
        points[past_start : frame_index + 1, point_index],
        mask[past_start : frame_index + 1, point_index],
        to_output,
    )

    _draw_half(painter, future, TRACK_FUTURE_COLOR, future_radius, line_width)
    _draw_half(painter, past, TRACK_PAST_COLOR, past_radius, line_width)


def _visible_points(
    points: np.ndarray,
    mask: np.ndarray,
    to_output: Callable[[float, float], tuple[int, int] | None],
) -> list[tuple[int, int] | None]:
    """Map the unmasked points, keeping a ``None`` wherever one was dropped.

    Masked-out frames are gaps in the track just as cropped-away points are, so both
    end up as ``None`` and break the line rather than being joined across.
    """
    mapped: list[tuple[int, int] | None] = []
    for point, visible in zip(points, mask, strict=True):
        mapped.append(to_output(float(point[0]), float(point[1])) if visible else None)
    return mapped


def _draw_half(
    painter: QtGui.QPainter,
    mapped: list[tuple[int, int] | None],
    color: QtGui.QColor,
    radius: int,
    line_width: int,
) -> None:
    """Draw one half of the track: a marker per position, joined by a line."""
    pen = QtGui.QPen(color)
    pen.setWidth(line_width)

    for run, _complete in visible_runs(mapped):
        if len(run) >= 2:
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in run]))

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        for x, y in run:
            painter.drawEllipse(QtCore.QPoint(x, y), radius, radius)
