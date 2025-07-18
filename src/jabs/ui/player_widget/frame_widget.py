import enum

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from jabs.pose_estimation import PoseEstimation
from jabs.project import TrackLabels
from jabs.ui.colors import (
    ACTIVE_ID_COLOR,
    BACKGROUND_COLOR,
    BEHAVIOR_COLOR,
    INACTIVE_ID_COLOR,
    KEYPOINT_COLOR_MAP,
    NOT_BEHAVIOR_COLOR,
)
from jabs.utils.pose_util import gen_line_fragments

_BEHAVIOR_LABEL_OUTLINE_COLOR = QtGui.QColor(255, 255, 255)
_FONT_SIZE = 16  # size of the font used for identity labels
_BEHAVIOR_LABEL_SIZE = 10  # size of the behavior label square
_GAP = 5  # gap between identity label and behavior label
_LINE_SEGMENT_COLOR = QtGui.QColor(255, 255, 255, 128)  # color for the pose line segments
_KEYPOINT_SIZE = 3  # size of the keypoint circles
_SELECTED_INDICATOR_SIZE = (
    2  # size of the circle drawn to indicate selected mouse when ID labels are hidden
)


class FrameWidget(QtWidgets.QLabel):
    """widget that implements a resizable pixmap label

    Used for displaying the current frame of the video. If necessary,
    the pixmap size is scaled down to fit the available area.
    """

    pixmap_clicked = QtCore.Signal(dict)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._font = QtGui.QFont()
        self._font.setPointSize(_FONT_SIZE)
        self._font.setBold(True)
        self._font_metrics = QtGui.QFontMetrics(self._font)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self._scaled_pix_x = 0
        self._scaled_pix_y = 0
        self._scaled_pix_width = 0
        self._scaled_pix_height = 0
        self._frame_number = 0
        self._active_identity = 0
        self._pose: PoseEstimation | None = None
        self._labels: list[np.ndarray] | None = None
        self._pose_overlay_mode = self.PoseOverlayMode.NONE
        self._id_overlay_mode = self.IdentityOverlayMode.FLOATING
        self._overlay_identity_enabled = True

        self.setMinimumSize(400, 400)

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
    def overlay_identity_enabled(self) -> bool:
        """Get whether the identity overlay is enabled."""
        return self._overlay_identity_enabled

    @overlay_identity_enabled.setter
    def overlay_identity_enabled(self, enabled: bool) -> None:
        """Set whether the identity overlay is enabled.

        Args:
            enabled (bool): True to enable identity overlay, False to disable.
        """
        if self._overlay_identity_enabled != enabled:
            self._overlay_identity_enabled = enabled
            self.update()

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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Process mousePressEvent to emit a signal with the clicked pixel coordinates."""
        pix_x, pix_y = self._widget_to_image_coords(event.x(), event.y())
        self.pixmap_clicked.emit({"x": pix_x, "y": pix_y})

        super().mousePressEvent(event)

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

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """override paintEvent handler to scale the image if the widget is resized.

        Don't enable resizing until after  the first frame has been drawn so
        that the widget will be expanded to fit the actual size of the frame.
        """
        # only draw if we have an image to show
        if self.pixmap() is not None and not self.pixmap().isNull():
            # current size of the widget
            size = self.size()

            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
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

            # draw the pixmap starting at the new calculated offset
            painter.drawPixmap(point, pix)

            # save the scaled pixmap dimensions for use by the mousePressEvent and _overlay_identities methods
            self._scaled_pix_x = point.x()
            self._scaled_pix_y = point.y()
            self._scaled_pix_width = pix.width()
            self._scaled_pix_height = pix.height()

            if self._pose_overlay_mode == self.PoseOverlayMode.ALL:
                self._overlay_pose(painter, all_identities=True)
            elif self._pose_overlay_mode == self.PoseOverlayMode.ACTIVE_IDENTITY:
                self._overlay_pose(painter, all_identities=False)

            if self._id_overlay_mode in (
                self.IdentityOverlayMode.CENTROID,
                self.IdentityOverlayMode.MINIMAL,
            ):
                self._overlay_identities(painter)

            if self._labels:
                self._overlay_labels(painter)

        else:
            # if we don't have a pixmap to display just call the original QLabel
            # paintEvent
            super().paintEvent(event)

    def update_frame(self, frame: QtGui.QImage, frame_number: int) -> None:
        """Update the frame displayed in the widget."""
        self._frame_number = frame_number
        self.setPixmap(QtGui.QPixmap.fromImage(frame))

    def _overlay_identities(self, painter: QtGui.QPainter) -> None:
        """Overlay identities on the current frame.

        This method draws the identity labels on the frame if pose estimation is available. The active identity
        label will be red, while other identities are drawn in a different color.
        """
        if self._pose is None:
            return

        identities = self._pose.identities
        painter.setFont(self._font)

        for identity in identities:
            shape = self._pose.get_identity_convex_hulls(identity)[self._frame_number]
            if shape is not None:
                center = shape.centroid

                color = ACTIVE_ID_COLOR if identity == self._active_identity else INACTIVE_ID_COLOR
                label_text = str(self.convert_identity_to_external(identity))

                # Convert image coordinates to widget coordinates and draw the label
                widget_x, widget_y = self.image_to_widget_coords(center.x, center.y)
                painter.setPen(color)

                if self._id_overlay_mode == self.IdentityOverlayMode.MINIMAL:
                    # draw a circle at the centroid of the identity
                    painter.setBrush(color)
                    painter.drawEllipse(
                        QtCore.QPoint(widget_x, widget_y),
                        _SELECTED_INDICATOR_SIZE,
                        _SELECTED_INDICATOR_SIZE,
                    )
                else:
                    painter.drawText(widget_x, widget_y, label_text)

    def _overlay_labels(self, painter: QtGui.QPainter) -> None:
        if self._pose is None:
            return

        identities = self._pose.identities

        for identity in identities:
            shape = self._pose.get_identity_convex_hulls(identity)[self._frame_number]
            if shape is None:
                continue

            center = shape.centroid
            widget_x, widget_y = self.image_to_widget_coords(center.x, center.y)

            # draw a square next to the centroid to indicate behavior label
            if self._id_overlay_mode == self.IdentityOverlayMode.FLOATING:
                # if the identity overlay is floating, we draw the behavior label to the right of the identity label
                # since that usually looks better due to the line connecting the label to the centroid
                behavior_x = widget_x + _GAP
            else:
                # if the identity overlay is not floating, we draw the behavior label to the left of the identity label
                # that leaves room for the identity label to be drawn
                behavior_x = widget_x - _BEHAVIOR_LABEL_SIZE - _GAP

            behavior_y = (
                widget_y
                - self._font_metrics.ascent()
                + self._font_metrics.height() // 2
                - _BEHAVIOR_LABEL_SIZE // 2
            )

            match self._labels[identity][self._frame_number]:
                case TrackLabels.Label.BEHAVIOR:
                    prediction_color = BEHAVIOR_COLOR
                case TrackLabels.Label.NOT_BEHAVIOR:
                    prediction_color = NOT_BEHAVIOR_COLOR
                case _:
                    prediction_color = BACKGROUND_COLOR

            painter.setBrush(prediction_color)
            painter.setPen(_BEHAVIOR_LABEL_OUTLINE_COLOR)
            painter.drawRect(behavior_x, behavior_y, _BEHAVIOR_LABEL_SIZE, _BEHAVIOR_LABEL_SIZE)

    def _overlay_pose(self, painter: QtGui.QPainter, all_identities: bool = False) -> None:
        """Overlay pose estimation on the current frame.

        This method draws the pose estimation skeletons on the frame. If `all_identities` is True,
        it will draw all identities; otherwise, it will only draw the active identity with a more prominent color.

        Args:
            painter (QtGui.QPainter): The painter used to draw on the frame.
            all_identities (bool): If True, draw all identities; if False, only draw the active identity.
        """
        if self._pose is None:
            return

        # draw the pose estimation skeletons
        for identity in self._pose.identities:
            if not all_identities and identity != self._active_identity:
                continue

            points, mask = self._pose.get_points(self._frame_number, identity)

            if points is None or mask is None:
                continue

            # Adjust alpha for non-active identities
            if identity != self._active_identity:
                line_color = QtGui.QColor(_LINE_SEGMENT_COLOR)
                line_color.setAlpha(line_color.alpha() // 3)  # More translucent
            else:
                line_color = _LINE_SEGMENT_COLOR

            pen = QtGui.QPen(line_color)
            pen.setWidth(3)
            painter.setPen(pen)

            for seg in gen_line_fragments(
                self._pose.get_connected_segments(), np.flatnonzero(mask == 0)
            ):
                segment_points = [self.image_to_widget_coords(p[0], p[1]) for p in points[seg]]

                # draw lines
                if len(segment_points) >= 2:
                    for i in range(len(segment_points) - 1):
                        painter.drawLine(
                            segment_points[i][0],
                            segment_points[i][1],
                            segment_points[i + 1][0],
                            segment_points[i + 1][1],
                        )

            # draw points at each keypoint of the pose (if it exists at this frame)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for keypoint in PoseEstimation.KeypointIndex:
                point_index = keypoint.value
                if mask[point_index]:
                    widget_x, widget_y = self.image_to_widget_coords(
                        points[point_index][0], points[point_index][1]
                    )

                    # Use the color map to get the color for the keypoint
                    # and make it translucent if it's not the active identity
                    color = KEYPOINT_COLOR_MAP[keypoint]
                    if identity != self._active_identity:
                        # Make keypoints translucent for non-active identities
                        translucent_color = QtGui.QColor(color)
                        translucent_color.setAlpha(96)
                        painter.setBrush(translucent_color)
                    else:
                        painter.setBrush(color)

                    painter.drawEllipse(
                        QtCore.QPoint(widget_x, widget_y), _KEYPOINT_SIZE, _KEYPOINT_SIZE
                    )
