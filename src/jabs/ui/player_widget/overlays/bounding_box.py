from typing import TYPE_CHECKING, cast

from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import QApplication

from jabs.pose_estimation import PoseEstimationV8
from jabs.ui.ear_tag_icons import EarTagIconManager

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


TAB_PADDING = 4  # pixels of padding around text/icon in tab
TAB_FILL_COLOR = QtGui.QColor(64, 64, 64, 150)


class BoundingBoxOverlay(Overlay):
    """Overlay that draws bounding boxes from pose over the video frame.

    Bounding boxes are defined as two points: upper-left and lower-right corners.
    The identity is displayed in a "tab" above the upper-left corner of the box.
    Clicking on the tab emits the id_label_clicked signal with the identity, which
    will cause the parent widget to set that identity as active. The active identity's
    bounding box is drawn in the application palette accent color, which matches the
    color used to highlight the selected identity in the stacked timeline view, while
    inactive identities are drawn in the application palette highlight color. Uses an
    ear tag icon if available for the identity, otherwise displays the display ID as
    text.

    Requires PoseEst v8 or later.
    """

    id_label_clicked = QtCore.Signal(int)

    def __init__(self, parent: "FrameWithOverlaysWidget") -> None:
        super().__init__(parent)
        self._eartags = EarTagIconManager()
        self._label_font = QtGui.QFont()
        self._label_font.setPointSize(12)
        self._font_metrics = QtGui.QFontMetrics(self._label_font)
        self._line_width = 2
        # Track tab rectangles for hit-testing mouse clicks:
        self._tab_rects: list[tuple[int, QtCore.QRectF]] = []

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paint the bounding box overlay.

        Args:
            painter (QtGui.QPainter): The painter to draw on the frame.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if (
            not self._enabled
            or self.parent.pose.format_major_version < 8
            or self.parent.pixmap().isNull()
            or self.parent.identity_overlay_mode != self.parent.IdentityOverlayMode.BBOX
        ):
            return

        # we use methods available only in PoseEstimationV8 and later
        # cast the base PoseEstimation to PoseEstimationV8 to avoid IDE warnings about undefined methods
        pose = cast(PoseEstimationV8, self.parent.pose)
        if not pose.has_bounding_boxes:
            return

        # Reset tab rects each paint; they'll be repopulated for hit-testing
        self._tab_rects.clear()

        # Sort identities so active identity is last, this will ensure it is drawn on top
        identities = self.parent.pose.identities
        active_id = self.parent.active_identity
        identities = [i for i in identities if i != active_id] + (
            [active_id] if active_id in identities else []
        )

        # Get the frame boundaries in widget coordinates
        frame_left = self.parent.scaled_pix_x
        frame_top = self.parent.scaled_pix_y
        frame_right = frame_left + self.parent.scaled_pix_width
        frame_bottom = frame_top + self.parent.scaled_pix_height

        # QRectF for the visible video frame in widget coordinates.
        # In order to maintain aspect ratio, the frame image might not take up the entire widget space.
        # To avoid drawing overlay outside the frame image, we'll clip to this rectangle.
        frame_rect = QtCore.QRectF(
            frame_left, frame_top, self.parent.scaled_pix_width, self.parent.scaled_pix_height
        )

        # save the painter state so that we can restore later
        painter.save()
        painter.setClipRect(frame_rect)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setFont(self._label_font)

        identity_text_height = self._font_metrics.height()

        for identity in identities:
            # bbox format is [upper_left_x, upper_left_y], [lower_right_x, lower_right_y]
            bboxes = pose.get_bounding_boxes(identity)
            if bboxes is None:
                continue
            bbox = bboxes[self.parent.current_frame]

            top_left = self._convert_coordinates((bbox[0, 0], bbox[0, 1]), crop_rect)
            bottom_right = self._convert_coordinates((bbox[1, 0], bbox[1, 1]), crop_rect)

            x1, y1 = top_left.x(), top_left.y()
            x2, y2 = bottom_right.x(), bottom_right.y()

            # Skip degenerate or invalid boxes (zero/negative width/height)
            if not (x2 > x1 and y2 > y1):
                continue

            # Skip if the bbox is completely outside the frame
            # this also handles missing bounding boxes, which are set to [-1, -1], [-1, -1]
            if x2 < frame_left or x1 > frame_right or y2 < frame_top or y1 > frame_bottom:
                continue

            # set up the bounding box
            base_color = (
                self._get_highlight_color() if identity != active_id else self._get_accent_color()
            )
            pen = QtGui.QPen(base_color)
            pen.setWidth(self._line_width)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            bbox_rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

            # only draw the portion of the bbox that intersects the visible frame
            # note: this means if the bbox is partially offscreen due to cropping the video, we resize it
            # to fit. We could just draw the full box, which will be clipped by the painter's clip rect,
            # but this looks better.
            bbox_draw_rect = bbox_rect.intersected(frame_rect)
            if bbox_draw_rect.isEmpty():
                continue
            painter.drawRect(bbox_draw_rect)

            display_id = pose.identity_index_to_display(identity)

            # draw a "tab" above the upper-left corner of the bounding box to display the identity
            eartag_renderer = self._eartags.get_icon(display_id)
            if eartag_renderer and eartag_renderer.isValid():
                # make the icon height match the text height so the tab height is consistent
                default_size = eartag_renderer.defaultSize()
                if default_size.width() > 0 and default_size.height() > 0:
                    icon_h = identity_text_height
                    icon_w = int(icon_h * default_size.width() / default_size.height())
                else:
                    icon_w = icon_h = identity_text_height
                has_icon = True
            else:
                has_icon = False
                icon_w = 0
                icon_h = 0

            # Tab width depends on whether we draw an icon or text, height is consistent either way
            if has_icon and icon_w > 0:
                tab_w = icon_w + 2 * TAB_PADDING
            else:
                tab_w = self._font_metrics.horizontalAdvance(display_id) + 2 * TAB_PADDING
            tab_h = identity_text_height + 2 * TAB_PADDING

            # Position: prefer above the box; clamp to keep the tab fully inside frame_rect
            # Horizontal clamp
            tab_x = max(frame_left, min(x1, frame_right - tab_w))

            # Vertical: anchor to the visible top edge of the bbox, not the offscreen y1
            visible_top = max(y1, frame_top)
            desired_tab_y = visible_top - tab_h  # prefer above the visible top
            if desired_tab_y >= frame_top:
                tab_y = desired_tab_y
            else:
                # place flush with the visible top edge; clamp if needed
                tab_y = visible_top
                if tab_y + tab_h > frame_bottom:
                    tab_y = frame_bottom - tab_h

            # If the tab is larger than the frame itself, skip drawing it
            if tab_w > frame_rect.width() or tab_h > frame_rect.height():
                painter.restore()
                continue

            tab_rect = QtCore.QRectF(tab_x, tab_y, tab_w, tab_h)

            # Save for hit-testing in mouse handler
            self._tab_rects.append((identity, QtCore.QRectF(tab_rect)))

            painter.setBrush(QtGui.QBrush(TAB_FILL_COLOR))
            painter.setPen(QtGui.QPen(base_color, self._line_width))
            painter.drawRect(tab_rect)

            # Draw icon or text centered within the tab
            if has_icon and icon_w > 0 and icon_h > 0:
                icon_rect = QtCore.QRectF(
                    tab_rect.x() + (tab_rect.width() - icon_w) / 2.0,
                    tab_rect.y() + (tab_rect.height() - icon_h) / 2.0,
                    icon_w,
                    icon_h,
                )
                eartag_renderer.render(painter, icon_rect)
            else:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                painter.drawText(tab_rect, QtCore.Qt.AlignmentFlag.AlignCenter, display_id)

        painter.restore()

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> bool:
        """Handle mouse press events to detect clicks on bounding box tabs.

        Returns True if the event was handled.
        """
        if not self._enabled or not self._tab_rects:
            return False

        # QPointF in widget coordinates
        pos_f = (
            event.position()
            if hasattr(event, "position")
            else QtCore.QPointF(event.x(), event.y())
        )

        for identity, rect in self._tab_rects:
            if rect.contains(pos_f):
                self.id_label_clicked.emit(identity)
                return True

        return False

    @staticmethod
    def _get_accent_color() -> QtGui.QColor:
        """Get the accent color from the application palette."""
        palette = QApplication.palette()
        return palette.color(palette.ColorRole.Accent)

    @staticmethod
    def _get_highlight_color() -> QtGui.QColor:
        """Get the highlight color from the application palette."""
        palette = QApplication.palette()
        return palette.color(palette.ColorRole.Highlight)

    def _convert_coordinates(
        self, point: tuple[float, float], crop_rect: QtCore.QRect
    ) -> QtCore.QPointF:
        """Convert image coordinates to widget coordinates.

        Args:
            point (tuple[float, float]): The (x, y) coordinates in image space.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Returns:
            QtCore.QPointF: The converted point in widget coordinates.
        """
        frame_left = self.parent.scaled_pix_x
        frame_top = self.parent.scaled_pix_y
        cw = crop_rect.width()
        ch = crop_rect.height()

        # Guard against empty or zero-sized crop rect to avoid division by zero.
        if cw <= 0 or ch <= 0:
            # Fallback to the top-left of the visible frame.
            return QtCore.QPointF(frame_left, frame_top)

        # Compute scale factors
        scale_x = self.parent.scaled_pix_width / cw
        scale_y = self.parent.scaled_pix_height / ch

        x = frame_left + (point[0] - crop_rect.left()) * scale_x
        y = frame_top + (point[1] - crop_rect.top()) * scale_y

        return QtCore.QPointF(x, y)
