from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from jabs.overlay_drawing import LABEL_MARKER_SIZE, draw_label_marker, label_marker_color

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class LabelOverlay(Overlay):
    """Overlay for displaying manual or predicted labels on the video frame."""

    _GAP = 5  # gap between identity label and behavior label

    def __init__(self, parent: "FrameWithOverlaysWidget"):
        super().__init__(parent)

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints the label overlay on the current frame.

        Args:
            painter (QtGui.QPainter): The painter used to draw on the widget.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if not self._enabled or self.parent.pixmap().isNull():
            return

        self._overlay_labels(painter, crop_rect)

    def _overlay_labels(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        if self.parent.pose is None or self.parent.labels is None:
            return

        identities = self.parent.pose.identities

        for identity in identities:
            shape = self.parent.pose.get_identity_convex_hulls(identity)[self.parent.current_frame]
            if shape is None:
                continue

            center = shape.centroid
            widget_coords = self.parent.image_to_widget_coords_cropped(
                center.x, center.y, crop_rect
            )
            if widget_coords is None:
                continue  # skip if outside cropped region

            widget_x, widget_y = widget_coords

            # draw a square next to the centroid to indicate behavior label
            if self.parent.identity_overlay_mode == self.parent.IdentityOverlayMode.FLOATING:
                # if the identity overlay is floating, we draw the behavior label to the right of the identity label
                # since that usually looks better due to the line connecting the label to the centroid
                behavior_x = widget_x + self._GAP
            else:
                # if the identity overlay is not floating, we draw the behavior label to the left of the identity label
                # that leaves room for the identity label to be drawn
                behavior_x = widget_x - LABEL_MARKER_SIZE - self._GAP

            behavior_y = widget_y - LABEL_MARKER_SIZE

            label_val = int(self.parent.labels[identity][self.parent.current_frame])
            prediction_color = label_marker_color(label_val, self.parent.label_color_lut)

            draw_label_marker(painter, behavior_x, behavior_y, LABEL_MARKER_SIZE, prediction_color)
