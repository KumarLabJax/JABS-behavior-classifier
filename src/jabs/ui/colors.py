import colorsys

from PySide6.QtGui import QColor

from jabs.pose_estimation import PoseEstimation

POSITION_MARKER_COLOR = QColor(231, 66, 126, 255)
SELECTION_COLOR = QColor(255, 255, 0, 255)

BACKGROUND_COLOR = QColor(128, 128, 128, 255)
NOT_BEHAVIOR_COLOR = QColor(0, 86, 229, 255)
BEHAVIOR_COLOR = QColor(255, 165, 0, 255)

INACTIVE_ID_COLOR = QColor(0, 222, 215, 255)
ACTIVE_ID_COLOR = QColor(255, 0, 0, 255)

SEARCH_HIT_COLOR = QColor(0, 255, 0, 255)


def __rainbow_palette(n: int) -> list[tuple[int, ...]]:
    """Generate n visually distinct colors using HSV color space."""
    return [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 1, 1)) for i in range(n)]


__KEYPOINT_COLORS = __rainbow_palette(len(PoseEstimation.KeypointIndex))

KEYPOINT_COLOR_MAP = {
    keypoint: QColor(*__KEYPOINT_COLORS[i])  # type: ignore
    for i, keypoint in enumerate(PoseEstimation.KeypointIndex)
}
