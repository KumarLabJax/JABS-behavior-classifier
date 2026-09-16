"""Qt drawing primitives for the per-identity behavior label/prediction marker.

The marker is a small filled square drawn next to an identity's centroid, colored by
that identity's label or prediction for the current frame. The player draws it at the
display's scale; the video export draws it at the video's native resolution. Both go
through this module so an exported video matches what the player shows.
"""

import numpy as np
import numpy.typing as npt
from PySide6 import QtGui

from .colors import BACKGROUND_COLOR, BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR
from .scaling import native_overlay_scale

# Base marker edge length, in pixels, at the player's display scale.
LABEL_MARKER_SIZE = 10

# Base gap between the marker and the identity's centroid.
LABEL_MARKER_GAP = 5

# Base gap between the two markers drawn when a label and a prediction are shown
# together. Smaller than the gap to the centroid so the pair reads as one group.
LABEL_MARKER_PAIR_GAP = 2

# White keeps the marker readable against both the animal and the arena floor.
LABEL_MARKER_OUTLINE_COLOR = QtGui.QColor(255, 255, 255)

# Binary label values, matching jabs.project.TrackLabels.Label. Repeated as plain
# integers because jabs.project pulls in the whole project subsystem, which the CLI
# export must not import just to color a square.
_LABEL_NOT_BEHAVIOR = 0
_LABEL_BEHAVIOR = 1


def native_label_marker_sizes(width: int, height: int) -> tuple[int, int, int]:
    """Return ``(marker_size, gap, pair_gap)`` for label markers at native resolution.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Tuple of ``(marker_size, gap, pair_gap)`` in pixels, all scaled from the frame
        size by :func:`~jabs.overlay_drawing.scaling.native_overlay_scale`. ``pair_gap``
        separates the two markers drawn when a label and a prediction are shown together.
    """
    scale = native_overlay_scale(width, height)
    marker_size = max(LABEL_MARKER_SIZE, round(LABEL_MARKER_SIZE * scale))
    gap = max(LABEL_MARKER_GAP, round(LABEL_MARKER_GAP * scale))
    pair_gap = max(LABEL_MARKER_PAIR_GAP, round(LABEL_MARKER_PAIR_GAP * scale))
    return marker_size, gap, pair_gap


def label_marker_color(
    label_value: int, color_lut: npt.NDArray[np.uint8] | None = None
) -> QtGui.QColor:
    """Return the marker color for one identity's label value on one frame.

    Args:
        label_value: The label or prediction for this identity and frame. With a
            ``color_lut`` this is an index into that table; without one it is a binary
            ``TrackLabels.Label`` value.
        color_lut: RGBA table of shape ``(N, 4)`` for multi-class coloring, or ``None``
            to use the binary behavior/not-behavior colors. Out-of-range indices are
            clamped, so a label array that outgrew its table still renders.

    Returns:
        The color to fill the marker with. Values that are neither behavior nor
        not-behavior (no prediction, or no pose on this frame) get
        :data:`~jabs.overlay_drawing.colors.BACKGROUND_COLOR`.
    """
    if color_lut is not None:
        index = max(0, min(int(label_value), len(color_lut) - 1))
        r, g, b, a = color_lut[index]
        return QtGui.QColor(int(r), int(g), int(b), int(a))

    value = int(label_value)
    if value == _LABEL_BEHAVIOR:
        return BEHAVIOR_COLOR
    if value == _LABEL_NOT_BEHAVIOR:
        return NOT_BEHAVIOR_COLOR
    return BACKGROUND_COLOR


def draw_label_marker(
    painter: QtGui.QPainter, x: int, y: int, size: int, color: QtGui.QColor
) -> None:
    """Draw one label marker with ``painter``.

    Antialiasing is turned off for the marker and restored afterwards: a square with
    antialiased edges reads as blurry at the sizes used here.

    Args:
        painter: The painter to draw with.
        x: Left edge of the marker, in the painter's coordinate space.
        y: Top edge of the marker, in the painter's coordinate space.
        size: Edge length of the marker in pixels.
        color: Fill color, from :func:`label_marker_color`.
    """
    antialiasing = painter.testRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
    painter.setBrush(color)
    painter.setPen(LABEL_MARKER_OUTLINE_COLOR)
    painter.drawRect(x, y, size, size)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, antialiasing)
