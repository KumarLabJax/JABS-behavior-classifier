import dataclasses
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from jabs.ui.colors import ACTIVE_ID_COLOR, INACTIVE_ID_COLOR

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


@dataclasses.dataclass(frozen=True)
class LabelOverlayRect:
    """Data class to hold information about an annotation rectangle."""

    x: float
    y: float
    width: float
    height: float
    tag: str
    color: QtGui.QColor
    animal_id: int | None = None
    is_id_label: bool = False


class FloatingIdOverlay(Overlay):
    """Overlay for displaying interval-based annotations as tags in rounded rectangles.

    Args:
        parent (FrameWithOverlaysWidget): The parent widget that contains the frame and overlays.
    """

    _FONT_SIZE = 12  # size of the font used for floating identity labels
    _CENTROID_FONT_SIZE = 14  # size of the font used for centroid identity labels
    _HORIZONTAL_PADDING = 6  # Horizontal padding inside id label rectangles (pixels)
    _VERTICAL_PADDING = 2  # Vertical padding inside id label rectangles (pixels)
    _SPACING = 4  # Vertical space between stacked id label rectangles (pixels)
    _MARGIN_X = 10  # Left margin for non-animal id label rectangles (pixels)
    _MARGIN_Y = 10  # Top margin for non-animal id label rectangles (pixels)
    _CORNER_RADIUS = 4  # Corner radius for rounded id label rectangles (pixels)
    _BORDER_COLOR = QtGui.QColor(225, 225, 225, 255)  # Border color for id label rectangles
    _ACTIVE_IDENTITY_COLOR = QtGui.QColor(255, 0, 0, 255)  # Color for active identity
    _INACTIVE_IDENTITY_COLOR = QtGui.QColor(125, 125, 125, 255)  # Color for inactive identities
    _LABEL_OFFSET_VERTICAL = 20  # Vertical offset from centroid
    _LABEL_OFFSET_HORIZONTAL = 60  # Horizontal offset for identity rectangle
    _SELECTED_INDICATOR_SIZE = (
        2  # size of the circle drawn to indicate selected mouse when ID labels are hidden
    )

    id_label_clicked = QtCore.Signal(int)

    def __init__(self, parent: "FrameWithOverlaysWidget"):
        super().__init__(parent)

        self._font = QtGui.QFont()
        self._font.setBold(True)
        self._font.setPointSize(self._FONT_SIZE)
        self._font_metrics = QtGui.QFontMetrics(self._font)

        self._rects_with_data = []

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints floating id labels.

        Args:
            painter (QtGui.QPainter): The painter to draw on the frame.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if not self._enabled or self.parent.pixmap().isNull():
            return

        # get the current painter font so we can restore it later
        current_font = painter.font()
        painter.setFont(self._font)

        # keep track of drawn rectangles and associated data, used for mouse events so we can
        # determine which id was clicked
        self._rects_with_data.clear()

        if self.parent.identity_overlay_mode == self.parent.IdentityOverlayMode.FLOATING:
            self._font.setPointSize(self._FONT_SIZE)
            self._overlay_identities_floating(painter, crop_rect)
        else:
            self._font.setPointSize(self._CENTROID_FONT_SIZE)
            self._overlay_identities(painter, crop_rect)

        # restore the original font
        painter.setFont(current_font)

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> bool:
        """Handle mouse press events to check if a floating id label was clicked.

        Args:
            event (QtGui.QMouseEvent): The mouse event containing the position of the click.

        Returns:
            bool: True if an id label was clicked, False otherwise.
        """
        if not self._enabled:
            return False

        pos = event.position()
        for rect, data in self._rects_with_data:
            if rect.contains(pos):
                self.id_label_clicked.emit(data.animal_id)
                return True
        return False

    def _overlay_identities_floating(
        self, painter: QtGui.QPainter, crop_rect: QtCore.QRect
    ) -> None:
        """Overlay identities on the current frame using the floating style.

        Args:
            painter (QtGui.QPainter): The painter to draw on the frame.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        This method draws the identity labels on the frame if pose estimation is available. The active identity
        label will be red, while other identities are drawn in a different color.
        """
        if self.parent.pose is None:
            return

        identities = self.parent.pose.identities

        # Sort identities so active identity is last, this will ensure it is drawn on top
        active_id = self.parent.active_identity
        identities = [i for i in identities if i != active_id] + (
            [active_id] if active_id in identities else []
        )

        # Get the frame boundaries in widget coordinates
        frame_left = self.parent.scaled_pix_x
        frame_top = self.parent.scaled_pix_y
        frame_right = frame_left + self.parent.scaled_pix_width

        # Draw animal id labels
        for identity in identities:
            display_id = self.parent.convert_identity_to_external(identity)

            centroid = self.get_centroid(identity)
            if centroid is None:
                continue

            # Use cropped coordinate conversion
            widget_coords = self.parent.image_to_widget_coords_cropped(
                centroid.x, centroid.y, crop_rect
            )
            if widget_coords is None:
                continue

            widget_x, widget_y = widget_coords

            identity_text = f"{display_id}"
            identity_text_width = self._font_metrics.horizontalAdvance(identity_text)
            identity_text_height = self._font_metrics.height()
            identity_rect_width = identity_text_width + self._HORIZONTAL_PADDING * 2
            identity_rect_height = identity_text_height + self._VERTICAL_PADDING * 2
            identity_x = widget_x - identity_rect_width / 2 - self._LABEL_OFFSET_HORIZONTAL
            identity_y = widget_y - identity_rect_height - self._LABEL_OFFSET_VERTICAL

            # adjust position if needed to fit within the frame
            if identity_x < frame_left:
                identity_x = widget_x - identity_rect_width / 2 + self._LABEL_OFFSET_HORIZONTAL
            elif identity_x + identity_rect_width > frame_right:
                identity_x = frame_right - identity_rect_width

            if identity_y < frame_top:
                identity_y = widget_y + self._LABEL_OFFSET_VERTICAL

            identity_rect = LabelOverlayRect(
                x=identity_x,
                y=identity_y,
                width=identity_rect_width,
                height=identity_rect_height,
                tag=identity_text,
                color=self._ACTIVE_IDENTITY_COLOR
                if identity == active_id
                else self._INACTIVE_IDENTITY_COLOR,
                animal_id=identity,
                is_id_label=True,
            )

            # Draw line to connect floating id label to the centroid
            id_center_x = identity_rect.x + identity_rect.width / 2
            id_center_y = identity_rect.y + identity_rect.height / 2
            pen = QtGui.QPen(self._BORDER_COLOR)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(int(widget_x), int(widget_y), int(id_center_x), int(id_center_y))

            # Draw identity label rectangle
            q_rect = QtCore.QRectF(
                identity_rect.x, identity_rect.y, identity_rect.width, identity_rect.height
            )
            self._rects_with_data.append((q_rect, identity_rect))
            text_color = (
                QtGui.QColor(0, 0, 0)
                if self._is_color_light(identity_rect.color)
                else QtGui.QColor(255, 255, 255)
            )
            painter.setBrush(identity_rect.color)
            painter.setPen(self._BORDER_COLOR)
            painter.drawRoundedRect(q_rect, self._CORNER_RADIUS, self._CORNER_RADIUS)
            painter.setPen(text_color)
            painter.drawText(q_rect, QtCore.Qt.AlignmentFlag.AlignCenter, identity_rect.tag)

    def _overlay_identities(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Overlay identities on the current frame.

        Args:
            painter (QtGui.QPainter): The painter to draw on the frame.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        This method draws the identity labels on the frame if pose estimation is available. The active identity
        label will be red, while other identities are drawn in a different color.
        """
        if self.parent.pose is None:
            return

        identities = self.parent.pose.identities
        painter.setFont(self._font)

        for identity in identities:
            shape = self.parent.pose.get_identity_convex_hulls(identity)[self.parent.current_frame]
            if shape is not None:
                center = shape.centroid

                color = (
                    ACTIVE_ID_COLOR
                    if identity == self.parent.active_identity
                    else INACTIVE_ID_COLOR
                )
                label_text = str(self.parent.convert_identity_to_external(identity))

                # Convert image coordinates to widget coordinates and draw the label
                widget_coords = self.parent.image_to_widget_coords_cropped(
                    center.x, center.y, crop_rect
                )
                if widget_coords is None:
                    continue
                widget_x, widget_y = widget_coords

                painter.setPen(color)

                if self.parent.identity_overlay_mode == self.parent.IdentityOverlayMode.MINIMAL:
                    # draw a circle at the centroid of the identity
                    painter.setBrush(color)
                    painter.drawEllipse(
                        QtCore.QPoint(widget_x, widget_y),
                        self._SELECTED_INDICATOR_SIZE,
                        self._SELECTED_INDICATOR_SIZE,
                    )
                else:
                    painter.drawText(widget_x, widget_y, label_text)
