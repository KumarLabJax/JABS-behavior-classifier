"""Composite the JABS pose overlay onto a single video frame.

Shared by the GUI's "Export Frame", the GUI's "Export Video with Pose Overlay",
and ``jabs-cli export-video``, so all three produce identical pixels. Segmentation
contours are baked into the BGR frame by the headless drawing in
:mod:`jabs.video_reader.frame_annotation`; the pose keypoints and skeleton are then
painted on top with :func:`~jabs.pose_drawing.draw_identity_pose`.

Sharing the GUI's skeleton drawing is deliberate. A second cv2 reimplementation
would inevitably drift, and an exported video that does not match what the player
shows is worse than no export at all.

That means this module depends on Qt, which is fine: PySide6 is a hard dependency
of this package. Painting targets a ``QImage`` rather than a ``QPixmap``, and
``QImage`` needs no ``QGuiApplication``, so it works unchanged from a headless CLI
with no display and no offscreen platform plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PySide6 import QtGui

from jabs.pose_drawing import draw_identity_pose, native_pose_sizes
from jabs.video_reader import overlay_segmentation

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation


def render_overlay_frame(
    frame: npt.NDArray[np.uint8],
    pose_est: PoseEstimation,
    frame_index: int,
    *,
    draw_segmentation: bool = True,
) -> npt.NDArray[np.uint8]:
    """Draw the pose overlay onto one frame.

    Every identity is drawn the same way, at full opacity, with no
    active-identity emphasis - an export has no notion of a selected animal.

    Args:
        frame: Source frame in BGR order, shape ``(height, width, 3)``.
        pose_est: Pose estimation for the video the frame came from.
        frame_index: Index of this frame within the video.
        draw_segmentation: Whether to bake segmentation contours in as well.
            Ignored when the pose file carries no segmentation data - it predates
            v6, or is v6+ but was generated without it.

    Returns:
        A new BGR frame with the overlay drawn. The input is not modified.
    """
    img = frame.copy()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # `has_segmentation` rather than a version check: segmentation is optional even
    # in v6+ files, and skipping the loop avoids a no-op call per identity per frame.
    if draw_segmentation and getattr(pose_est, "has_segmentation", False):
        for identity in pose_est.identities:
            overlay_segmentation(
                img, pose_est, identity=identity, frame_index=frame_index, active=True
            )

    # QImage wraps this buffer, so painting below writes straight into `rgb`.
    rgb = np.ascontiguousarray(img[..., ::-1])
    height, width, channels = rgb.shape
    qimage = QtGui.QImage(
        rgb.data, width, height, channels * width, QtGui.QImage.Format.Format_RGB888
    )

    keypoint_size, line_width = native_pose_sizes(width, height)

    def to_native(x: float, y: float) -> tuple[int, int]:
        return round(float(x)), round(float(y))

    painter = QtGui.QPainter(qimage)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    try:
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
    finally:
        painter.end()

    return np.ascontiguousarray(rgb[..., ::-1])
