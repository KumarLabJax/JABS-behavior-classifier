import enum

import numpy as np
from intervaltree import IntervalTree
from PySide6 import QtCore, QtGui, QtWidgets

from jabs.pose_estimation import PoseEstimation

from .overlays.annotation_overlay import AnnotationOverlay
from .overlays.bounding_box import BoundingBoxOverlay
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
        BBOX = enum.auto()

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
        self._crop_p1: QtCore.QPoint | None = None
        self._crop_p2: QtCore.QPoint | None = None
        self._brightness = 1.0
        self._contrast = 1.0

        self._pose_overlay_mode = self.PoseOverlayMode.NONE
        self._id_overlay_mode = self.IdentityOverlayMode.BBOX

        self._control_overlay = ControlOverlay(self)
        self._control_overlay.playback_speed_changed.connect(self.playback_speed_changed)
        self._control_overlay.cropping_changed.connect(self._on_cropping_changed)
        self._control_overlay.brightness_changed.connect(self._on_brightness_changed)
        self._control_overlay.contrast_changed.connect(self._on_contrast_changed)

        self._annotation_overlay = AnnotationOverlay(self)
        floating_id_overlay = FloatingIdOverlay(self)
        floating_id_overlay.id_label_clicked.connect(self.id_label_clicked)
        bbox_overlay = BoundingBoxOverlay(self)
        bbox_overlay.id_label_clicked.connect(self.id_label_clicked)

        # overlays are listed in the order they should be painted
        # an overlay on top of another overlay will have priority when handling click events
        self.overlays: list[Overlay] = [
            PoseOverlay(self),
            LabelOverlay(self),
            bbox_overlay,
            self._annotation_overlay,
            floating_id_overlay,
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

    @property
    def is_cropped(self) -> bool:
        """Check if the frame is currently cropped."""
        return self._crop_p1 is not None and self._crop_p2 is not None

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

    def convert_identity_to_external(self, identity: int) -> str:
        """Convert an internal identity index to an external identity index.

        This is useful when the pose estimation uses external identities so that we can display
        the external identity instead of the internal jabs identity index.
        """
        if self._pose:
            return self._pose.identity_index_to_display(identity)
        return str(identity)

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

    def widget_to_image_coords(self, x: int, y: int) -> tuple[int, int]:
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

    def image_to_widget_coords_cropped(
        self, img_x: int, img_y: int, crop_rect: QtCore.QRect
    ) -> tuple[int, int] | None:
        """
        Convert image coordinates to widget coordinates within a cropped region.

        Args:
            img_x (int): X coordinate in image space.
            img_y (int): Y coordinate in image space.
            crop_rect (QtCore.QRect): Cropped rectangle region in image coordinates.

        Returns:
            tuple[int, int] | None: Widget coordinates if inside crop_rect, otherwise None.
        """
        # Only draw overlays if inside crop_rect
        if not crop_rect.contains(img_x, img_y):
            return None

        # Translate image coordinates to cropped region
        x = img_x - crop_rect.left()
        y = img_y - crop_rect.top()

        # Scale to widget coordinates
        pixmap_width = crop_rect.width()
        pixmap_height = crop_rect.height()
        if self._scaled_pix_width >= 1 and self._scaled_pix_height >= 1:
            x = x * self._scaled_pix_width / pixmap_width
            y = y * self._scaled_pix_height / pixmap_height
            x += self._scaled_pix_x
            y += self._scaled_pix_y
            return int(x), int(y)
        return None

    def sizeHint(self) -> QtCore.QSize:
        """Override QLabel.sizeHint to give an initial starting size."""
        return QtCore.QSize(1024, 1024)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Handles the paint event for the widget and draws all overlays.

        Will handle cropping and scaling of the pixmap before drawing it.

        Args:
            event (QtGui.QPaintEvent): The paint event containing region to be updated.
        """
        if self.pixmap() is None or self.pixmap().isNull():
            return

        size = self.size()
        pix = self.pixmap()

        pix = self._adjust_brightness_contrast(pix)

        # Step 1: Crop if crop points are set
        if self._crop_p1 and self._crop_p2:
            x1, y1 = self._crop_p1.x(), self._crop_p1.y()
            x2, y2 = self._crop_p2.x(), self._crop_p2.y()
            crop_rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            pix = pix.copy(crop_rect)
        else:
            crop_rect = QtCore.QRect(0, 0, pix.width(), pix.height())

        # Step 2: Scale cropped pixmap to widget size
        scaled_pix = pix.scaled(
            size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        point = QtCore.QPoint(
            (size.width() - scaled_pix.width()) // 2,
            (size.height() - scaled_pix.height()) // 2,
        )

        self._scaled_pix_x = point.x()
        self._scaled_pix_y = point.y()
        self._scaled_pix_width = scaled_pix.width()
        self._scaled_pix_height = scaled_pix.height()

        # Step 3: Draw the scaled pixmap in the center of the widget
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        painter.drawPixmap(point, scaled_pix)

        # Step 4: Draw overlays in widget coordinates, but only if inside crop_rect
        for overlay in self.overlays:
            overlay.paint(painter, crop_rect)

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
            pix_x, pix_y = self.widget_to_image_coords(event.x(), event.y())
            self.pixmap_clicked.emit({"x": pix_x, "y": pix_y})

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse release events and delegates them to overlays.

        Args:
            event (QtGui.QMouseEvent): The mouse release event.
        """
        super().mouseReleaseEvent(event)
        for overlay in reversed(self.overlays):
            overlay.handle_mouse_release(event)

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

    def _on_cropping_changed(self, p1: QtCore.QPoint, p2: QtCore.QPoint) -> None:
        """Handles cropping changes from the control overlay."""
        if p1 == p2 or p1 is None or p2 is None:
            self._crop_p1 = None
            self._crop_p2 = None
        else:
            self._crop_p1 = p1
            self._crop_p2 = p2
        self.update()

    def _on_brightness_changed(self, brightness: float) -> None:
        """Handles brightness changes from the control overlay."""
        if self._brightness != brightness:
            self._brightness = brightness
            self.update()

    def _on_contrast_changed(self, contrast: float) -> None:
        """Handles contrast changes from the control overlay."""
        if self._contrast != contrast:
            self._contrast = contrast
            self.update()

    def _adjust_brightness_contrast(self, pixmap: QtGui.QPixmap) -> QtGui.QPixmap:
        """Adjust the brightness of the given pixmap based on the current brightness setting."""
        # use a threshold to avoid unnecessary processing if brightness and contrast are very close to default
        if abs(self._brightness - 1.0) < 0.01 and abs(self._contrast - 1.0) < 0.01:
            return pixmap

        # convert Qt pixmap to numpy array for brightness and contrast manipulation
        img = pixmap.toImage()
        width, height = img.width(), img.height()
        bytes_per_pixel = img.depth() // 8
        arr = np.frombuffer(img.bits(), dtype=np.uint8, count=width * height * bytes_per_pixel)
        arr = arr.reshape((height, width, bytes_per_pixel))

        # adjust the RGB channels only, leave alpha channel unchanged
        arr[..., :3] = np.clip(
            (arr[..., :3] * self._brightness - 128) * self._contrast + 128, 0, 255
        )

        return QtGui.QPixmap.fromImage(
            QtGui.QImage(arr.data, width, height, img.bytesPerLine(), img.format())
        )
