"""Composite the JABS overlays onto a single video frame.

Shared by the GUI's "Export Frame", the GUI's "Export Video with Overlays", and
``jabs-cli export-video``, so an overlay looks the same whichever asked for it. They
do not ask for the same ones: only the video export passes a prediction overlay. Every
overlay is painted with the shared drawing in :mod:`jabs.overlay_drawing`, in the order
the player paints them: segmentation contours, then the pose keypoints and skeleton,
then the per-identity prediction markers.

Sharing the GUI's drawing is deliberate. A second reimplementation would inevitably
drift, and an exported video that does not match what the player shows is worse than no
export at all.

That means this module depends on Qt, which is fine: PySide6 is a hard dependency
of this package. Painting targets a ``QImage`` rather than a ``QPixmap``, and
``QImage`` needs no ``QGuiApplication``, so it works unchanged from a headless CLI
with no display and no offscreen platform plugin. The one exception is the caption
banner, which needs Qt's font database; see :mod:`jabs.video_export.caption`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PySide6 import QtGui

from jabs.overlay_drawing import (
    draw_identity_pose,
    draw_identity_segmentation,
    draw_label_marker,
    label_marker_color,
    native_label_marker_sizes,
    native_pose_sizes,
    native_segmentation_line_width,
)

from .caption import draw_overlay_caption

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

    from .prediction_overlay import PredictionOverlay


def render_overlay_frame(
    frame: npt.NDArray[np.uint8],
    pose_est: PoseEstimation,
    frame_index: int,
    *,
    draw_pose: bool = True,
    draw_segmentation: bool = True,
    prediction_overlay: PredictionOverlay | None = None,
) -> npt.NDArray[np.uint8]:
    """Draw the requested overlays onto one frame.

    Every identity is drawn the same way, at full opacity, with no
    active-identity emphasis - an export has no notion of a selected animal.

    Args:
        frame: Source frame in BGR order, shape ``(height, width, 3)``.
        pose_est: Pose estimation for the video the frame came from.
        frame_index: Index of this frame within the video.
        draw_pose: Whether to draw the pose keypoints and skeleton.
        draw_segmentation: Whether to draw the segmentation contours as well.
            Ignored when the pose file carries no segmentation data - it predates
            v6, or is v6+ but was generated without it.
        prediction_overlay: Predictions to mark next to each identity, with the
            caption explaining them, or ``None`` to draw no predictions.

    Returns:
        A new BGR frame with the overlays drawn. The input is not modified. With
        every overlay switched off this is simply a copy of the input.
    """
    img = frame.copy()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # `has_segmentation` rather than a version check: segmentation is optional even
    # in v6+ files, and this skips a no-op call per identity per frame.
    segmentation = draw_segmentation and getattr(pose_est, "has_segmentation", False)

    if not segmentation and not draw_pose and prediction_overlay is None:
        return img

    # QImage wraps this buffer, so painting below writes straight into `rgb`.
    rgb = np.ascontiguousarray(img[..., ::-1])
    height, width, channels = rgb.shape
    qimage = QtGui.QImage(
        rgb.data, width, height, channels * width, QtGui.QImage.Format.Format_RGB888
    )

    def to_native(x: float, y: float) -> tuple[int, int]:
        return round(float(x)), round(float(y))

    painter = QtGui.QPainter(qimage)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    try:
        if segmentation:
            for identity in pose_est.identities:
                draw_identity_segmentation(
                    painter,
                    pose_est,
                    frame_index,
                    identity,
                    to_output=to_native,
                    line_width=native_segmentation_line_width(width, height),
                    active=True,
                )

        if draw_pose:
            keypoint_size, line_width = native_pose_sizes(width, height)
            for identity in pose_est.identities:
                draw_identity_pose(
                    painter,
                    pose_est,
                    frame_index,
                    identity,
                    to_output=to_native,
                    keypoint_size=keypoint_size,
                    line_width=line_width,
                    active=True,
                )

        if prediction_overlay is not None:
            _draw_prediction_markers(
                painter, pose_est, frame_index, prediction_overlay, width, height
            )
            draw_overlay_caption(
                painter,
                width,
                height,
                prediction_overlay.caption,
                prediction_overlay.legend,
            )
    finally:
        painter.end()

    return np.ascontiguousarray(rgb[..., ::-1])


def _draw_prediction_markers(
    painter: QtGui.QPainter,
    pose_est: PoseEstimation,
    frame_index: int,
    prediction_overlay: PredictionOverlay,
    width: int,
    height: int,
) -> None:
    """Draw one prediction marker beside each identity's centroid.

    Placed to the left of the centroid, the way the player places it in every
    identity overlay mode but the floating one, so the marker sits beside the animal
    rather than on top of its pose.
    """
    marker_size, gap = native_label_marker_sizes(width, height)

    for identity in pose_est.identities:
        label = prediction_overlay.label_value(identity, frame_index)
        if label is None:
            continue

        shape = pose_est.get_identity_convex_hulls(identity)[frame_index]
        if shape is None:
            # No pose for this identity on this frame, so nothing to sit beside.
            continue

        center = shape.centroid
        x = round(float(center.x)) - marker_size - gap
        y = round(float(center.y)) - marker_size
        draw_label_marker(
            painter, x, y, marker_size, label_marker_color(label, prediction_overlay.color_lut)
        )
