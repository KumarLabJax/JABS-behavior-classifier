import distinctipy
from PySide6.QtGui import QColor

from jabs.pose_estimation import PoseEstimation

POSITION_MARKER_COLOR = QColor(231, 66, 126, 255)
SELECTION_COLOR = QColor(255, 255, 0, 255)

BACKGROUND_COLOR = QColor(128, 128, 128, 255)

NOT_BEHAVIOR_COLOR = QColor(0, 86, 229, 255)
NOT_BEHAVIOR_COLOR_BRIGHT = QColor(50, 119, 234)
NOT_BEHAVIOR_BUTTON_DISABLED_COLOR = QColor(0, 77, 206)

BEHAVIOR_COLOR = QColor(255, 165, 0, 255)
BEHAVIOR_BUTTON_COLOR_BRIGHT = QColor(255, 195, 77, 255)
BEHAVIOR_BUTTON_DISABLED_COLOR = QColor(229, 143, 0, 255)

INACTIVE_ID_COLOR = QColor(0, 222, 215, 255)
ACTIVE_ID_COLOR = QColor(255, 0, 0, 255)

SEARCH_HIT_COLOR = QColor(0, 255, 0, 255)


# Generate distinct colors for keypoints
# Use a fixed seed for reproducible colors
__KEYPOINT_COLORS = distinctipy.get_colors(len(PoseEstimation.KeypointIndex), rng=42)

KEYPOINT_COLOR_MAP = {
    kp: QColor(int(r * 255), int(g * 255), int(b * 255))
    for kp, (r, g, b) in zip(PoseEstimation.KeypointIndex, __KEYPOINT_COLORS, strict=True)
}
