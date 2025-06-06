import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from jabs.pose_estimation import PoseEstimation
from jabs.project import TrackLabels

from ..colors import BACKGROUND_COLOR, BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR

_ID_COLOR = QtGui.QColor(0, 222, 215)
_ACTIVE_COLOR = QtGui.QColor(255, 0, 0)
_BEHAVIOR_LABEL_OUTLINE_COLOR = QtGui.QColor(255, 255, 255)
_FONT_SIZE = 16  # size of the font used for identity labels
_BEHAVIOR_LABEL_SIZE = 10  # size of the behavior label square
_GAP = 5  # gap between identity label and behavior label


class FrameWidget(QtWidgets.QLabel):
    """widget that implements a resizable pixmap label

    Used for displaying the current frame of the video. If necessary,
    the pixmap size is scaled down to fit the available area.
    """

    pixmap_clicked = QtCore.Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._font = QtGui.QFont()
        self._font.setPointSize(_FONT_SIZE)
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
        self._pose = None
        self._labels = None

        self.setMinimumSize(400, 400)

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

    def set_label_overlay(self, labels: list[np.ndarray]):
        """set label values to use for overlaying on the frame.

        Labels are used to indicated behavior or not behavior label or predction for each identity

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

    def _image_to_widget_coords(self, pix_x: int, pix_y: int) -> tuple[int, int]:
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

            self._overlay_identities(painter)

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

                color = _ACTIVE_COLOR if identity == self._active_identity else _ID_COLOR
                label = (
                    str(identity)
                    if not self._pose.external_identities
                    else str(self._pose.external_identities[identity])
                )
                # Convert image coordinates to widget coordinates and draw the label
                widget_x, widget_y = self._image_to_widget_coords(center.x, center.y)
                painter.setPen(color)
                painter.drawText(widget_x, widget_y, label)

                # also add an overlay with a behavior label if available
                # (source of label can be manual label or model prediction)
                if self._labels:
                    # draw a square next to the identity label to indicate behavior label
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

                    painter.setBrush(QtGui.QColor(*prediction_color))
                    painter.setPen(_BEHAVIOR_LABEL_OUTLINE_COLOR)
                    painter.drawRect(
                        behavior_x, behavior_y, _BEHAVIOR_LABEL_SIZE, _BEHAVIOR_LABEL_SIZE
                    )
