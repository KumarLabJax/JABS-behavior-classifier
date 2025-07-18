import contextlib

from intervaltree import IntervalTree
from PySide6 import QtCore, QtGui

from jabs.pose_estimation import PoseEstimation

from .frame_widget import FrameWidget
from .overlays.annotation_overlay import AnnotationOverlay
from .overlays.control_overlay import ControlOverlay
from .overlays.floating_id_overlay import FloatingIdOverlay
from .overlays.overlay import Overlay


class FrameWidgetWithInteractiveOverlays(FrameWidget):
    """
    A `FrameWidget` subclass that adds an interactive overlays.

    This widget displays a number of interactive overlays on top of the video frame, including
     * a controls overlay, which is displayed when the mouse is over the frame pixmap area.
     * an overlay for displaying timeline annotations for the current frame.

    Signals:
        playback_speed_changed (float): Emitted when the playback speed is changed by the user.

    Implements some additional properties and methods so that overlays can access
    information from the frame widget.

    Todo:
        - Merge FrameWidget and FrameWidgetWithInteractiveOverlays into a single class, and
          implement the identity and pose overlays as Overlay subclasses.
    """

    playback_speed_changed = QtCore.Signal(float)
    id_label_clicked = QtCore.Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

        self._annotations: IntervalTree | None = None
        self._overlay_annotations_enabled = True
        self._floating_id_overlay_enabled = True

        # initialize overlays
        self._control_overlay = ControlOverlay(self)
        self._control_overlay.playback_speed_changed.connect(self.playback_speed_changed)
        self.overlays: list[Overlay] = [self._control_overlay]

        # add optional overlays
        if self._overlay_annotations_enabled:
            self.overlays.append(AnnotationOverlay(self))
        if self._floating_id_overlay_enabled:
            floating_overlay = FloatingIdOverlay(self)
            floating_overlay.id_label_clicked.connect(self.id_label_clicked)
            self.overlays.append(floating_overlay)

    @property
    def overlay_annotations_enabled(self) -> bool:
        """Get whether the annotation overlay is enabled."""
        return self._overlay_annotations_enabled

    @overlay_annotations_enabled.setter
    def overlay_annotations_enabled(self, enabled: bool) -> None:
        """Set whether the annotation overlay is enabled."""
        if self._overlay_annotations_enabled != enabled:
            self._overlay_annotations_enabled = enabled
            self.update()

    @property
    def floating_id_overlay_enabled(self) -> bool:
        """Get whether the floating ID overlay is enabled."""
        return self._floating_id_overlay_enabled

    @floating_id_overlay_enabled.setter
    def floating_id_overlay_enabled(self, enabled: bool) -> None:
        """Set whether the floating ID overlay is enabled."""
        if self._floating_id_overlay_enabled != enabled:
            self._floating_id_overlay_enabled = enabled
            if enabled:
                floating_overlay = FloatingIdOverlay(self)
                floating_overlay.id_label_clicked.connect(self.id_label_clicked)
                self.overlays.append(floating_overlay)
            else:
                for overlay in self.overlays:
                    if isinstance(overlay, FloatingIdOverlay):
                        with contextlib.suppress(TypeError, RuntimeError):
                            # disconnect the signal to avoid dangling connections
                            QtCore.QCoreApplication.processEvents()
                            overlay.id_label_clicked.disconnect(self.id_label_clicked)
                self.overlays = [
                    overlay
                    for overlay in self.overlays
                    if not isinstance(overlay, FloatingIdOverlay)
                ]
            self.update()

    @property
    def identity_overlay_mode(self) -> FrameWidget.IdentityOverlayMode:
        """Get the current identity overlay mode."""
        return super().identity_overlay_mode

    @identity_overlay_mode.setter
    def identity_overlay_mode(self, mode: FrameWidget.IdentityOverlayMode) -> None:
        """Set the identity overlay mode for the frame widget.

        Args:
            mode (IdentityOverlayMode): The mode to set for overlaying identities.
        """
        if self._id_overlay_mode != mode:
            self.floating_id_overlay_enabled = mode == FrameWidget.IdentityOverlayMode.FLOATING
            self._id_overlay_mode = mode
            self.update()

    @property
    def scaled_pix_x(self):
        """Get the scaled x-coordinate of the pixmap in the widget."""
        return self._scaled_pix_x

    @property
    def scaled_pix_y(self):
        """Get the scaled y-coordinate of the pixmap in the widget."""
        return self._scaled_pix_y

    @property
    def scaled_pix_height(self):
        """Get the scaled height of the pixmap in the widget."""
        return self._scaled_pix_height

    @property
    def scaled_pix_width(self):
        """Get the scaled width of the pixmap in the widget."""
        return self._scaled_pix_width

    @property
    def current_frame(self) -> int:
        """Get the current frame number."""
        return self._frame_number

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

    @property
    def pose(self) -> PoseEstimation:
        """Returns the pose estimation object associated with this widget."""
        return self._pose

    @property
    def active_identity(self) -> int | None:
        """Returns the currently active identity index, or None if no identity is active."""
        return self._active_identity

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
