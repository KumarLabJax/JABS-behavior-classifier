from collections import defaultdict
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from .annotation_info_dialog import AnnotationInfoDialog
from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class AnnotationOverlay(Overlay):
    """Overlay for displaying interval-based annotations as tags in rounded rectangles."""

    _HORIZONTAL_PADDING = 6  # Horizontal padding inside annotation rectangles (pixels)
    _VERTICAL_PADDING = 2  # Vertical padding inside annotation rectangles (pixels)
    _SPACING = 4  # Vertical space between stacked annotation rectangles (pixels)
    _MARGIN_X = 10  # Left margin for non-animal annotation rectangles (pixels)
    _MARGIN_Y = 10  # Top margin for non-animal annotation rectangles (pixels)
    _CORNER_RADIUS = 4  # Corner radius for rounded annotation rectangles (pixels)
    _BORDER_COLOR = QtGui.QColor(160, 160, 160, 255)  # Border color for annotation rectangles
    _ANNOTATION_OFFSET = 40  # Vertical offset from centroid (pixels)

    def __init__(self, parent: "FrameWithOverlaysWidget"):
        super().__init__(parent)

        self._annotation_font = QtGui.QFont()
        self._annotation_font.setBold(True)
        self._annotation_font.setPointSize(12)
        self._font_metrics = QtGui.QFontMetrics(self._annotation_font)

        self._rects_with_data = []

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints annotation tags for intervals overlapping the current frame.

        Args:
            painter (QtGui.QPainter): The painter used for drawing.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the video frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if not self._enabled or self.parent.pixmap().isNull():
            return

        frame_number = self.parent.current_frame
        annotations = self.parent.annotations[frame_number] if self.parent.annotations else []

        if not annotations:
            return

        # keep track of drawn rectangles and associated annotation data, used for mouse events so we can
        # determine which annotation was clicked
        self._rects_with_data.clear()

        # get the current painter font so we can restore it later
        current_font = painter.font()
        painter.setFont(self._annotation_font)

        # Group animal annotations by identity
        animal_annots = defaultdict(list)
        non_animal_annots = []
        for annotation in annotations:
            identity = annotation.data.get("identity", None)
            if identity is not None:
                animal_annots[identity].append(annotation)
            else:
                non_animal_annots.append(annotation)

        # Get the frame boundaries in widget coordinates
        frame_left = self.parent.scaled_pix_x
        frame_top = self.parent.scaled_pix_y
        frame_right = frame_left + self.parent.scaled_pix_width

        # Draw animal annotations (stacked per identity)
        for identity, animal_annotations in animal_annots.items():
            # get all annotations for this identity
            animal_annotations = sorted(
                animal_annotations, key=lambda a: a.data["tag"], reverse=True
            )

            # if there is a centroid for this identity, draw the annotations connected to the centroid in the video frame
            # otherwise, we will draw them in the upper left corner
            centroid = self.get_centroid(identity)
            if centroid is not None:
                coords = self.parent.image_to_widget_coords_cropped(
                    centroid.x, centroid.y, crop_rect
                )
                if coords is None:
                    continue  # Skip this annotation if mapping failed
                widget_x, widget_y = coords

                # generate the rectangles for each annotation
                rects = []
                n_annotations = len(animal_annotations)
                for idx, annotation in enumerate(animal_annotations):
                    text = annotation.data["tag"]
                    color_str = annotation.data["color"]
                    text_width = self._font_metrics.horizontalAdvance(text)
                    text_height = self._font_metrics.height()
                    rect_width = text_width + self._HORIZONTAL_PADDING * 2
                    rect_height = text_height + self._VERTICAL_PADDING * 2

                    # Default: center above centroid, stack vertically
                    x = widget_x - rect_width / 2
                    y = (
                        widget_y
                        - rect_height
                        - self._ANNOTATION_OFFSET
                        - idx * (rect_height + self._SPACING)
                    )

                    # Adjust horizontally if out of bounds
                    if x < frame_left:
                        x = frame_left
                    elif x + rect_width > frame_right:
                        x = frame_right - rect_width

                    # Adjust vertically if out of bounds (above frame)
                    # unlike when drawing above, the lower annotation index will be closer to the centroid
                    if y < frame_top:
                        y = (
                            widget_y
                            + self._ANNOTATION_OFFSET
                            + (n_annotations - 1 - idx) * (rect_height + self._SPACING)
                        )

                    rects.append((x, y, rect_width, rect_height, annotation, text, color_str))

                # Find the rect closest to the centroid (smallest |y - widget_y|)
                closest_rect = min(rects, key=lambda r: abs((r[1] + r[3] / 2) - widget_y))

                # Draw the line from this rect to the centroid
                # Note: if the annotations get split with some above and some below, this will only draw the line to one group
                line_x = int(closest_rect[0] + closest_rect[2] / 2)
                if closest_rect[1] + closest_rect[3] / 2 < widget_y:
                    # Above centroid: line from bottom center
                    line_y = int(closest_rect[1] + closest_rect[3])
                else:
                    # Below centroid: line from top center
                    line_y = int(closest_rect[1])
                pen = QtGui.QPen(self._BORDER_COLOR)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(line_x, line_y, int(widget_x), int(widget_y))

                # Draw all stacked rectangles
                for x, y, rect_width, rect_height, annotation, text, color_str in rects:
                    rect = QtCore.QRectF(x, y, rect_width, rect_height)
                    data = annotation.data.copy()
                    data["start"] = annotation.begin
                    data["end"] = annotation.end - 1  # make end inclusive
                    self._rects_with_data.append((rect, data))
                    fill_color = QtGui.QColor(color_str)
                    if not fill_color.isValid():
                        fill_color = QtGui.QColor(220, 220, 220)
                    text_color = (
                        QtGui.QColor(0, 0, 0)
                        if self._is_color_light(fill_color)
                        else QtGui.QColor(255, 255, 255)
                    )
                    painter.setBrush(fill_color)
                    painter.setPen(self._BORDER_COLOR)
                    painter.drawRoundedRect(rect, self._CORNER_RADIUS, self._CORNER_RADIUS)
                    painter.setPen(text_color)
                    painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)
            else:
                non_animal_annots += animal_annotations

        # Draw non-animal annotations stacked in upper left
        non_animal_annots = sorted(non_animal_annots, key=lambda a: a.data["tag"])
        for i, annotation in enumerate(non_animal_annots):
            identity = annotation.data.get("identity")

            if identity is not None:
                display_id = self.parent.convert_identity_to_external(identity)
                text = f"{display_id}: {annotation.data['tag']}"
            else:
                text = annotation.data["tag"]

            color_str = annotation.data["color"]
            text_width = self._font_metrics.horizontalAdvance(text)
            text_height = self._font_metrics.height()
            rect_width = text_width + self._HORIZONTAL_PADDING * 2
            rect_height = text_height + self._VERTICAL_PADDING * 2
            x = self.parent.scaled_pix_x + self._MARGIN_X
            y = self.parent.scaled_pix_y + self._MARGIN_Y + i * (rect_height + self._SPACING)

            rect = QtCore.QRectF(x, y, rect_width, rect_height)
            data = annotation.data.copy()
            data["start"] = annotation.begin
            data["end"] = annotation.end - 1  # make end inclusive
            self._rects_with_data.append((rect, data))
            fill_color = QtGui.QColor(color_str)
            if not fill_color.isValid():
                fill_color = QtGui.QColor(220, 220, 220)
            fill_color.setAlpha(220)
            text_color = (
                QtGui.QColor(0, 0, 0)
                if self._is_color_light(fill_color)
                else QtGui.QColor(255, 255, 255)
            )
            painter.setBrush(fill_color)
            painter.setPen(self._BORDER_COLOR)
            painter.drawRoundedRect(rect, self._CORNER_RADIUS, self._CORNER_RADIUS)
            painter.setPen(text_color)
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

        # restore the original font
        painter.setFont(current_font)

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> bool:
        """Handle mouse press events to check if an annotation rectangle was clicked.

        Args:
            event (QtGui.QMouseEvent): The mouse event containing the click position.

        Returns:
            bool: True if an annotation was clicked, False otherwise.
        """
        if not self._enabled:
            return False

        pos = event.position()
        for rect, annotation in self._rects_with_data:
            if rect.contains(pos):
                dialog = AnnotationInfoDialog(annotation, self.parent)
                dialog.exec()
                return True
        return False
