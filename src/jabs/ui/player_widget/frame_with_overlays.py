import enum

import numpy as np
from intervaltree import IntervalTree
from PySide6 import QtCore, QtGui, QtWidgets

from jabs.pose_estimation import PoseEstimation

from .overlays.annotation_overlay import AnnotationOverlay
from .overlays.control_overlay import ControlOverlay
from .overlays.floating_id_overlay import FloatingIdOverlay
from .overlays.label_overlay import LabelOverlay
from .overlays.overlay import Overlay
from .overlays.pose_overlay import PoseOverlay


class FrameWithOverlaysWidget(QtWidgets.QLabel):
    """
    A Video Frame with interactive overlays.

    This widget displays a number of interactive overlays on top of the video frame, including
      * a controls overlay, which is displayed when the mouse is over the frame pixmap area.
      * an overlay for displaying timeline annotations for the current frame.
      * an overlay for displaying pose estimation keypoints and skeletons.
      * an overlay for displaying identity labels, which can be in different modes.
      * an overlay for displaying labels next to each mouse, which can be manual labels or predictions.

    Signals:
        playback_speed_changed (float): Emitted when the playback speed is changed by the user.
        id_label_clicked (int): Emitted when an identity label is clicked, passing the identity index.
        pixmap_clicked (dict): Emitted when the user clicks on the pixmap, passing the x and y coordinates of the click.

    Implements some additional properties and methods so that overlays can access
    information from the frame widget.
    """

    class PoseOverlayMode(enum.IntEnum):
        """Enum to define the mode for overlaying pose estimation on the frame."""

        ALL = enum.auto()
        ACTIVE_IDENTITY = enum.auto()
        NONE = enum.auto()

    class IdentityOverlayMode(enum.IntEnum):
        """Enum for identity overlay options."""

        MINIMAL = enum.auto()
        CENTROID = enum.auto()
        FLOATING = enum.auto()

    pixmap_clicked = QtCore.Signal(dict)
    playback_speed_changed = QtCore.Signal(float)
    id_label_clicked = QtCore.Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

        self._scaled_pix_x = 0
        self._scaled_pix_y = 0
        self._scaled_pix_width = 0
        self._scaled_pix_height = 0
        self._frame_number = 0
        self._active_identity = 0
        self._pose: PoseEstimation | None = None
        self._annotations: IntervalTree | None = None
        self._labels: list[np.ndarray] | None = None

        self._pose_overlay_mode = self.PoseOverlayMode.NONE
        self._id_overlay_mode = self.IdentityOverlayMode.FLOATING

        # initialize overlays
        self._control_overlay = ControlOverlay(self)
        self._control_overlay.playback_speed_changed.connect(self.playback_speed_changed)
        self._annotation_overlay = AnnotationOverlay(self)
        floating_id_overlay = FloatingIdOverlay(self)
        floating_id_overlay.id_label_clicked.connect(self.id_label_clicked)

        # overlays are listed in the order they should be painted
        # an overlay on top of another overlay will have priority when handling click events
        self.overlays: list[Overlay] = [
            PoseOverlay(self),
            LabelOverlay(self),
            self._annotation_overlay,
            floating_id_overlay,
            # the control overlay is painted last, so it is on top of all other overlays
            self._control_overlay,
        ]

    @property
    def overlay_annotations_enabled(self) -> bool:
        """Get whether the annotation overlay is enabled."""
        return self._annotation_overlay.enabled

    @overlay_annotations_enabled.setter
    def overlay_annotations_enabled(self, enabled: bool) -> None:
        """Set whether the annotation overlay is enabled."""
        if self._annotation_overlay.enabled != enabled:
            self._annotation_overlay.enabled = enabled
            self.update()

    @property
    def identity_overlay_mode(self) -> IdentityOverlayMode:
        """Get the current identity overlay mode."""
        return self._id_overlay_mode

    @identity_overlay_mode.setter
    def identity_overlay_mode(self, mode: IdentityOverlayMode) -> None:
        """Set the identity overlay mode for the frame widget.

        Args:
            mode (IdentityOverlayMode): The mode to set for overlaying identities.
        """
        if self._id_overlay_mode != mode:
            self._id_overlay_mode = mode
            self.update()

    @property
    def pose_overlay_mode(self) -> PoseOverlayMode:
        """Get the current pose overlay mode."""
        return self._pose_overlay_mode

    @pose_overlay_mode.setter
    def pose_overlay_mode(self, mode: PoseOverlayMode) -> None:
        """Set the pose overlay mode for the frame widget.

        Args:
            mode (PoseOverlayMode): The mode to set for overlaying pose estimation.
        """
        if self._pose_overlay_mode != mode:
            self._pose_overlay_mode = mode
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

    @property
    def labels(self) -> list[np.ndarray] | None:
        """Returns the label values for overlaying on the frame."""
        return self._labels

    def reset(self) -> None:
        """reset state of frame widget"""
        self._scaled_pix_x = 0
        self._scaled_pix_y = 0
        self._scaled_pix_width = 0
        self._scaled_pix_height = 0
        self._pose = None
        self._frame_number = 0
        self._active_identity = 0
        self.clear()

    def convert_identity_to_external(self, identity: int) -> int:
        """Convert an internal identity index to an external identity index.

        This is useful when the pose estimation uses external identities so that we can display
        the external identity instead of the internal jabs identity index.
        """
        if self._pose and self._pose.external_identities:
            try:
                return self._pose.external_identities[identity]
            except IndexError:
                # If the identity is not found in external identities, fall through to return the original identity
                pass
        return identity

    def set_pose(self, pose: PoseEstimation) -> None:
        """Set the pose estimation for the frame widget.

        Pose is used to annotate the frame, this method muse be called any time the
        a new video is loaded.
        """
        self._pose = pose

    def set_active_identity(self, identity: int) -> None:
        """Set the active identity for the frame widget.

        This identity will be highlighted in the frame.
        """
        self._active_identity = identity

    def set_label_overlay(self, labels: list[np.ndarray]) -> None:
        """set label values to use for overlaying on the frame.

        Labels are used to indicated behavior or not behavior label or prediction for each identity

        Args:
            labels (list[np.ndarray]): list of label arrays, one for each identity.
            Each array should contain the label for each frame in the video for the given identity.
        """
        self._labels = labels

    def update_frame(self, frame: QtGui.QImage, frame_number: int) -> None:
        """Update the frame displayed in the widget."""
        self._frame_number = frame_number
        self.setPixmap(QtGui.QPixmap.fromImage(frame))

    def _widget_to_image_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert the given x, y coordinates from FrameWidget coordinates to correct image coordinates.

        Ie which pixel did the user click on? We account for image scaling and translation
        """
        pixmap = self.pixmap()
        if (
            pixmap is not None
            and (
                self._scaled_pix_height != pixmap.height()
                or self._scaled_pix_width != pixmap.width()
                or self._scaled_pix_x != 0
                or self._scaled_pix_y != 0
            )
            and self._scaled_pix_width >= 1
            and self._scaled_pix_height >= 1
        ):
            # we've done all the checks and it's safe to transform the x, y point
            x -= self._scaled_pix_x
            y -= self._scaled_pix_y
            x *= pixmap.width() / self._scaled_pix_width
            y *= pixmap.height() / self._scaled_pix_height

        return x, y

    def image_to_widget_coords(self, pix_x: int, pix_y: int) -> tuple[int, int]:
        """Convert true image coordinates to FrameWidget (QLabel) coordinates

        Accounts for scaling and centering to fit image in FrameWidget.

        Args:
            pix_x (int): x coordinate in pixmap coordinates
            pix_y (int): y coordinate in pixmap coordinates

        Returns:
            tuple[int, int]: x, y coordinates in FrameWidget coordinates
        """
        pixmap = self.pixmap()
        if pixmap is not None and self._scaled_pix_width >= 1 and self._scaled_pix_height >= 1:
            # Scale pixmap coordinates to scaled pixmap size
            x = pix_x * self._scaled_pix_width / pixmap.width()
            y = pix_y * self._scaled_pix_height / pixmap.height()

            # Offset by the scaled pixmap's position in the widget
            x += self._scaled_pix_x
            y += self._scaled_pix_y
            return int(x), int(y)
        return pix_x, pix_y

    def sizeHint(self) -> QtCore.QSize:
        """Override QLabel.sizeHint to give an initial starting size."""
        return QtCore.QSize(1024, 1024)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Handles the paint event for the widget and draws all overlays.

        Args:
            event (QtGui.QPaintEvent): The paint event containing region to be updated.
        """
        if self.pixmap() is None or self.pixmap().isNull():
            return

        size = self.size()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        point = QtCore.QPoint(0, 0)

        # scale the image to the current size of the widget.
        pix = self.pixmap().scaled(
            size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        # because we are maintaining aspect ratio, the scaled frame might
        # not be the same dimensions as the area we are painting it.
        # adjust the start point to center the image in the widget
        point.setX((size.width() - pix.width()) // 2)
        point.setY((size.height() - pix.height()) // 2)

        painter.drawPixmap(point, pix)

        # save the scaled pixmap dimensions for use by the mousePressEvent and _overlay_identities methods
        self._scaled_pix_x = point.x()
        self._scaled_pix_y = point.y()
        self._scaled_pix_width = pix.width()
        self._scaled_pix_height = pix.height()

        # paint all overlays in the order they were added
        for overlay in self.overlays:
            overlay.paint(painter)

        painter.end()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse move events and delegates them to all overlays.

        Args:
            event (QtGui.QMouseEvent): The mouse move event.
        """
        super().mouseMoveEvent(event)
        for overlay in self.overlays:
            overlay.handle_mouse_move(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Handles leave events and delegates them to all overlays.

        Args:
            event (QtCore.QEvent): The leave event.
        """
        super().leaveEvent(event)
        for overlay in self.overlays:
            overlay.handle_leave(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse press events and delegates them to overlays.

        Args:
            event (QtGui.QMouseEvent): The mouse press event.

        Note:
            Overlays are checked in reverse order (top-most first) to allow overlays drawn on top
            to have priority in handling the event. If no overlay handles the event, emits the
            `pixmap_clicked` signal with the image coordinates of the click.
        """
        handled = False
        for overlay in reversed(self.overlays):
            if overlay.handle_mouse_press(event):
                handled = True
                break
        if not handled:
            pix_x, pix_y = self._widget_to_image_coords(event.x(), event.y())
            self.pixmap_clicked.emit({"x": pix_x, "y": pix_y})

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Filters events before they reach the target object and delegates to overlays.

        Args:
            obj (QtCore.QObject): The object that is the target of the event.
            event (QtCore.QEvent): The event to be filtered.

        Returns:
            bool: True if the event should be filtered out, False otherwise.

        Note:
            This method allows overlays to filter out events before they reach the target object.
            If any overlay returns True for event_filter, the event is considered handled.
            Overlays are checked in reverse order to allow the "top-most" overlay to have priority.
        """
        for overlay in reversed(self.overlays):
            # allow any overlay to filter out events
            if overlay.event_filter(obj, event):
                return True
        return super().eventFilter(obj, event)
