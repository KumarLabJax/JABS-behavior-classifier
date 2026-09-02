"""Per-keypoint colors for the pose skeleton.

Kept apart from the drawing code so the palette can be inspected or reused without
pulling in the painter, and so a change here is visible as a change to the palette
rather than buried in a drawing change.
"""

import distinctipy
from PySide6 import QtGui

from jabs.pose_estimation import PoseEstimation

# Distinct per-keypoint colors, from a fixed seed so they are reproducible across
# runs and across the GUI overlay, frame export and video export.
_KEYPOINT_COLORS = distinctipy.get_colors(len(PoseEstimation.KeypointIndex), rng=42)

KEYPOINT_COLOR_MAP = {
    kp: QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))
    for kp, (r, g, b) in zip(PoseEstimation.KeypointIndex, _KEYPOINT_COLORS, strict=True)
}
