import colorsys

from PySide6.QtGui import QColor

from jabs.pose_estimation import PoseEstimation

POSITION_MARKER_COLOR = (231, 66, 126)
SELECTION_COLOR = (255, 255, 0)

BACKGROUND_COLOR = (128, 128, 128, 255)
NOT_BEHAVIOR_COLOR = (0, 86, 229, 255)
BEHAVIOR_COLOR = (255, 159, 0, 255)

SEARCH_HIT_COLOR = (0, 255, 0, 255)


def __rainbow_palette(n: int) -> list[tuple[int, ...]]:
    """Generate n visually distinct colors using HSV color space."""
    return [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 1, 1)) for i in range(n)]


__KEYPOINT_COLORS = __rainbow_palette(len(PoseEstimation.KeypointIndex))

KEYPOINT_COLOR_MAP = {
    keypoint: QColor(*__KEYPOINT_COLORS[i])  # type: ignore
    for i, keypoint in enumerate(PoseEstimation.KeypointIndex)
}
