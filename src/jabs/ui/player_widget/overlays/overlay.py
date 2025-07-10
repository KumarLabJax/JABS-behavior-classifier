from typing import TYPE_CHECKING

from PySide6 import QtGui
from PySide6.QtCore import QEvent, QObject

if TYPE_CHECKING:
    from ..frame_with_control_overlay import FrameWidgetWithInteractiveOverlays


class Overlay(QObject):
    """Base class for interactive overlays in the frame widget."""

    def __init__(self, parent: "FrameWidgetWithInteractiveOverlays"):
        super().__init__(parent)
        self.parent = parent

    def paint(self, painter: QtGui.QPainter) -> None:
        """Paints the overlay on the parent widget."""
        pass

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse press events on the overlay."""
        pass

    def handle_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse move events on the overlay."""
        pass

    def event_filter(self, obj: QObject, event: QEvent) -> bool:
        """Filters events for the overlay."""
        pass

    def handle_leave(self, event: QEvent) -> None:
        """Handles leave events for the overlay."""
        pass
