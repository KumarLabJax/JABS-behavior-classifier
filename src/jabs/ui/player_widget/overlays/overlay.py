from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui
from PySide6.QtCore import QEvent, QObject
from shapely import Point

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class Overlay(QObject):
    """Base class for interactive overlays in the frame widget."""

    _LIGHT_COLOR_THRESHOLD = 160  # Luminance threshold to determine if a color is "light"
    _MAX_PRIORITY = 1000  # Maximum priority for painting order

    def __init__(self, parent: "FrameWithOverlaysWidget"):
        super().__init__(parent)
        self._parent = parent  # Reference to the parent frame widget
        self._priority = 0  # Default priority for painting order
        self._enabled = True  # Flag to enable or disable the overlay

    @property
    def parent(self) -> "FrameWithOverlaysWidget":
        """Returns the parent frame widget."""
        return self._parent

    @property
    def priority(self) -> int:
        """Returns the priority of the overlay for painting order."""
        return self._priority

    @property
    def enabled(self) -> bool:
        """Returns whether the overlay is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Sets whether the overlay is enabled."""
        self._enabled = value

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints the overlay on the parent widget."""
        pass

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse press events on the overlay."""
        pass

    def handle_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse move events on the overlay."""
        pass

    def handle_mouse_release(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse release events on the overlay."""
        pass

    def event_filter(self, obj: QObject, event: QEvent) -> bool:
        """Filters events for the overlay."""
        return False

    def handle_leave(self, event: QEvent) -> None:
        """Handles leave events for the overlay."""
        pass

    def get_centroid(self, identity: int) -> Point | None:
        """Get the centroid of the given identity in the current frame.

        Args:
            identity (int): The identity index to get the centroid for.

        Returns:
            tuple[float, float]: The (x, y) coordinates of the centroid or
                None if there is no convex hull for the identity in the current frame.
        """
        convex_hull = self.parent.pose.get_identity_convex_hulls(identity)[
            self.parent.current_frame
        ]

        if convex_hull is None:
            return None

        return convex_hull.centroid

    def _is_color_light(self, color: QtGui.QColor) -> bool:
        """Determines if a color is considered light based on its luminance."""
        # Calculate luminance using the ITU-R BT.709 formula
        luminance = 0.2126 * color.red() + 0.7152 * color.green() + 0.0722 * color.blue()
        return luminance > self._LIGHT_COLOR_THRESHOLD
