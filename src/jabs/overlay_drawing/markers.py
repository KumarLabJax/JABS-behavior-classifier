"""Qt drawing for the marker that calls out one animal on a frame.

Used to point out the animal nearest the subject, and the nearest one inside the
subject's field of view, when the user asks for them. Drawn by the on-screen
:class:`~jabs.ui.player_widget.overlays.closest_identity_overlay.ClosestIdentityOverlay`.

Replaces the cv2 drawing that used to bake these markers into the frame as it was
decoded. The colors are the ones that drawing used, converted from its BGR tuples.
"""

from PySide6 import QtCore, QtGui

# The nearest animal, and the nearest one the subject can actually see.
CLOSEST_MARKER_COLOR = QtGui.QColor(0, 0, 255)
CLOSEST_FOV_MARKER_COLOR = QtGui.QColor(0, 255, 0)

# Base radius at the player's display scale; callers scale it with the zoom.
CLOSEST_MARKER_RADIUS = 4


def draw_identity_marker(
    painter: QtGui.QPainter, x: int, y: int, radius: int, color: QtGui.QColor
) -> None:
    """Draw a filled marker at a point in the painter's coordinate space.

    Args:
        painter: The painter to draw with.
        x: Marker center, in the painter's coordinate space.
        y: Marker center, in the painter's coordinate space.
        radius: Marker radius in pixels.
        color: Fill color, saying which animal is being called out.
    """
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawEllipse(QtCore.QPoint(x, y), radius, radius)
