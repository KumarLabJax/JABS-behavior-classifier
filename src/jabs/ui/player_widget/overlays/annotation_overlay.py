from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_control_overlay import FrameWidgetWithInteractiveOverlays


class AnnotationOverlay(Overlay):
    """Overlay for displaying interval-based annotations as tags in rounded rectangles."""

    _HORIZONTAL_PADDING = 6
    _VERTICAL_PADDING = 2
    _LIGHT_COLOR_THRESHOLD = 160

    def __init__(self, parent: "FrameWidgetWithInteractiveOverlays"):
        super().__init__(parent)

        self._annotation_font = QtGui.QFont()
        self._annotation_font.setBold(True)
        self._annotation_font.setPointSize(12)
        self._font_metrics = QtGui.QFontMetrics(self._annotation_font)

        self._rects_with_data = []

    def paint(self, painter: QtGui.QPainter) -> None:
        """
        Paints annotation tags for intervals overlapping the current frame.

        Args:
            painter (QtGui.QPainter): The painter used for drawing.
        """
        self._rects_with_data.clear()

        if self.parent.pixmap() is None or self.parent.pixmap().isNull():
            return

        frame_number = self.parent.frame_number
        annotations = self.parent.annotations[frame_number] if self.parent.annotations else []

        if not annotations:
            return

        annotations = sorted(annotations, key=lambda a: a.data["tag"])

        # Layout constants
        margin_x = 10
        margin_y = 10
        spacing = 4
        rect_radius = 4

        # get the current painter font so we can restore it later
        current_font = painter.font()
        painter.setFont(self._annotation_font)

        # Anchor to the upper left of the pixmap area
        x0 = self.parent.scaled_pix_x
        y0 = self.parent.scaled_pix_y

        for i, annotation in enumerate(annotations):
            tag = annotation.data["tag"]
            color_str = annotation.data["color"]
            identity = annotation.data.get("animal_id", None)

            text = tag if identity is None else f"{identity}: {tag}"

            text_width = self._font_metrics.horizontalAdvance(text)
            text_height = self._font_metrics.height()

            rect_width = text_width + self._HORIZONTAL_PADDING * 2
            rect_height = text_height + self._VERTICAL_PADDING * 2

            rect = QtCore.QRectF(
                x0 + margin_x, y0 + margin_y + i * (rect_height + spacing), rect_width, rect_height
            )
            self._rects_with_data.append((rect, annotation.data))

            fill_color = QtGui.QColor(color_str)
            if not fill_color.isValid():
                fill_color = QtGui.QColor(220, 220, 220)  # fallback color
            fill_color.setAlpha(220)

            # use a contrasting text color based on the fill color
            text_color = (
                QtGui.QColor(0, 0, 0)
                if self.__is_color_light(fill_color)
                else QtGui.QColor(255, 255, 255)
            )

            # Draw rounded rectangle
            painter.setBrush(fill_color)
            painter.setPen(QtGui.QColor(225, 225, 225))
            painter.drawRoundedRect(rect, rect_radius, rect_radius)

            # Draw text
            painter.setPen(text_color)
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

        # restore the original font and antialiasing state
        painter.setFont(current_font)

    def __is_color_light(self, color: QtGui.QColor) -> bool:
        """Determines if a color is considered light based on its luminance."""
        # Calculate luminance using the ITU-R BT.709 formula
        luminance = 0.2126 * color.red() + 0.7152 * color.green() + 0.0722 * color.blue()
        return luminance > self._LIGHT_COLOR_THRESHOLD

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press events to check if an annotation rectangle was clicked."""
        pos = event.position() if hasattr(event, "position") else event.pos()
        for rect, annotation in self._rects_with_data:
            if rect.contains(pos):
                # TODO: display annotations details in a dialog or tooltip
                print(f"Clicked on annotation: {annotation['tag']}")
