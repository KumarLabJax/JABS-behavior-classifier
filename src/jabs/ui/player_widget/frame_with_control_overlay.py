from intervaltree import IntervalTree
from PySide6 import QtCore, QtGui

from .frame_widget import FrameWidget
from .overlays.annotation_overlay import AnnotationOverlay
from .overlays.control_overlay import ControlOverlay


class FrameWidgetWithInteractiveOverlays(FrameWidget):
    """
    A `FrameWidget` subclass that adds an interactive overlays.

    This widget displays a number of interactive overlays on top of the video frame, incuding
     * a controls overlay, which is displayed when the mouse is over the frame pixmap area.
     * an overlay for displaying timeline annotations for the current frame.

    Currently, the controls overlay has one control: playback speed adjustment.
    This control consists of a badge that shows the current playback speed
    and allows the user to change the speed via a popup menu. The overlay
    and menu are only visible when the mouse is over the displayed pixmap area.

    Signals:
        playback_speed_changed (float): Emitted when the playback speed is changed by the user.
    """

    playback_speed_changed = QtCore.Signal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

        self._annotations: IntervalTree | None = None

        # initialize overlays
        self._control_overlay = ControlOverlay(self)
        self._control_overlay.playback_speed_changed.connect(self.playback_speed_changed)
        self.overlays = [self._control_overlay, AnnotationOverlay(self)]

    @property
    def playback_speed(self) -> float:
        """Returns the current playback speed set by the control overlay."""
        return self._control_overlay.playback_speed

    @property
    def annotations(self) -> IntervalTree | None:
        """Returns the interval annotations for the annotation overlay."""
        return self._annotations

    @annotations.setter
    def annotations(self, value: IntervalTree | None) -> None:
        """Sets the interval annotations for the annotation overlay."""
        self._annotations = value

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Handles the paint event for the widget and draws all overlays.

        Args:
            event (QtGui.QPaintEvent): The paint event containing region to be updated.
        """
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for overlay in self.overlays:
            overlay.paint(painter)
        painter.end()

    def mouseMoveEvent(self, event) -> None:
        """Handles mouse move events and delegates them to all overlays.

        Args:
            event (QtGui.QMouseEvent): The mouse move event.
        """
        super().mouseMoveEvent(event)
        for overlay in self.overlays:
            overlay.handle_mouse_move(event)

    def leaveEvent(self, event) -> None:
        """Handles leave events and delegates them to all overlays.

        Args:
            event (QtCore.QEvent): The leave event.
        """
        super().leaveEvent(event)
        for overlay in self.overlays:
            overlay.handle_leave(event)

    def mousePressEvent(self, event) -> None:
        """Handles mouse press events and delegates them to all overlays.

        Args:
            event (QtGui.QMouseEvent): The mouse press event.
        """
        handled = False
        for overlay in self.overlays:
            if overlay.handle_mouse_press(event):
                handled = True
                break
        if not handled:
            super().mousePressEvent(event)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Filters events before they reach the target object and delegates to overlays.

        Args:
            obj (QtCore.QObject): The object that is the target of the event.
            event (QtCore.QEvent): The event to be filtered.

        Returns:
            bool: True if the event should be filtered out, False otherwise.
        """
        for overlay in self.overlays:
            # allow any overlay to filter out events
            if hasattr(overlay, "event_filter") and overlay.event_filter(obj, event):
                return True
        return super().eventFilter(obj, event)
