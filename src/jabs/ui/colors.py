import distinctipy
import numpy as np
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

# Colors excluded when generating per-behavior palettes so they don't collide with
# the fixed UI colors used in binary mode.
_EXCLUDED_BEHAVIOR_PALETTE_COLORS: list[tuple[float, float, float]] = [
    (BACKGROUND_COLOR.redF(), BACKGROUND_COLOR.greenF(), BACKGROUND_COLOR.blueF()),
    (NOT_BEHAVIOR_COLOR.redF(), NOT_BEHAVIOR_COLOR.greenF(), NOT_BEHAVIOR_COLOR.blueF()),
    (BEHAVIOR_COLOR.redF(), BEHAVIOR_COLOR.greenF(), BEHAVIOR_COLOR.blueF()),
]


def make_behavior_color_map(behavior_names: list[str]) -> dict[str, QColor]:
    """Generate a visually distinct QColor for each behavior in multi-class mode.

    Colors are produced by ``distinctipy``, excluding the fixed UI colors used
    in binary mode (background gray, not-behavior blue, behavior orange) so
    that the new palette does not clash with existing UI elements. The same
    input always produces the same output (fixed RNG seed).

    Args:
        behavior_names: Ordered list of project behavior names. Must not
            include the reserved ``"None"`` behavior.

    Returns:
        Dictionary mapping each behavior name to a ``QColor``.
    """
    if not behavior_names:
        return {}
    rgb_floats = distinctipy.get_colors(
        len(behavior_names),
        exclude_colors=_EXCLUDED_BEHAVIOR_PALETTE_COLORS,
        rng=0,
    )
    return {
        name: QColor(int(r * 255), int(g * 255), int(b * 255))
        for name, (r, g, b) in zip(behavior_names, rgb_floats, strict=True)
    }


def build_multiclass_color_lut(
    behavior_names: list[str], color_map: dict[str, QColor]
) -> np.ndarray:
    """Build an RGBA lookup table for multi-class label visualization.

    The table layout is:
    - Index 0: background / unlabeled (``BACKGROUND_COLOR`` gray)
    - Index 1: "None" explicit-negative label (``NOT_BEHAVIOR_COLOR`` blue)
    - Index 2..N+1: per-behavior colors in ``behavior_names`` order

    Combined class-index arrays produced by
    ``VideoLabels.build_multiclass_label_array`` map directly to these
    indices (no offset arithmetic needed).

    Args:
        behavior_names: Ordered list of project behavior names (not including
            the reserved ``"None"`` behavior).
        color_map: Mapping from behavior name to ``QColor``, as returned by
            :func:`make_behavior_color_map`.

    Returns:
        ``np.ndarray`` of shape ``(N+2, 4)`` and dtype ``np.uint8`` containing
        RGBA values.
    """
    entries: list[tuple[int, int, int, int]] = [
        BACKGROUND_COLOR.getRgb(),
        NOT_BEHAVIOR_COLOR.getRgb(),
        *[color_map[name].getRgb() for name in behavior_names],
    ]
    return np.array(entries, dtype=np.uint8)
