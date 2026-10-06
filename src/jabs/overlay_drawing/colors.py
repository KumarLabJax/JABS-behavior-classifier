"""Colors for the overlays drawn on a video frame.

Kept apart from the drawing code so the palettes can be inspected or reused without
pulling in the painter, and so a change here is visible as a change to the palette
rather than buried in a drawing change.

The behavior label colors also back the GUI's own palette: ``jabs.ui.colors``
re-exports them so the timeline, the label buttons and the frame overlays cannot
drift apart. They live here rather than in ``jabs.ui.colors`` because importing
anything under ``jabs.ui`` pulls in ``MainWindow``, which the exports must not do.
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

# Behavior label colors, used for the label/prediction marker in binary mode and as
# the fixed entries of the multi-class color LUT.
BACKGROUND_COLOR = QtGui.QColor(128, 128, 128, 255)
NOT_BEHAVIOR_COLOR = QtGui.QColor(0, 86, 229, 255)
BEHAVIOR_COLOR = QtGui.QColor(255, 165, 0, 255)
