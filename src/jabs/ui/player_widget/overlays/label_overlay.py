from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from jabs.overlay_drawing import (
    LABEL_MARKER_GAP,
    LABEL_MARKER_PAIR_GAP,
    LABEL_MARKER_SIZE,
    draw_label_marker,
    label_marker_color,
)

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class LabelOverlay(Overlay):
    """Overlay for displaying manual labels, predicted labels, or both, on the video frame.

    A marker is drawn for each label source the frame widget has values for, so giving it
    both the manual labels and the predictions shows them side by side. The manual label
    is always the first marker of the pair and the prediction the second, whichever side
    of the identity label the pair is drawn on.
    """

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
        if self.parent.pose is None:
            return

        # one entry per label source to draw a marker for, in the order they are drawn
        label_sources = [
            values
            for values in (self.parent.manual_labels, self.parent.predicted_labels)
            if values
        ]
        if not label_sources:
            return

        # width of the full group of markers, so it can be placed as a unit relative to
        # the identity label
        group_width = (
            len(label_sources) * LABEL_MARKER_SIZE
            + (len(label_sources) - 1) * LABEL_MARKER_PAIR_GAP
        )

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

            # draw a square next to the centroid for each behavior label we're showing
            if self.parent.identity_overlay_mode == self.parent.IdentityOverlayMode.FLOATING:
                # if the identity overlay is floating, we draw the behavior labels to the right of
                # the identity label since that usually looks better due to the line connecting the
                # label to the centroid
                group_x = widget_x + LABEL_MARKER_GAP
            else:
                # if the identity overlay is not floating, we draw the behavior labels to the left
                # of the identity label. that leaves room for the identity label to be drawn
                group_x = widget_x - group_width - LABEL_MARKER_GAP

            marker_y = widget_y - LABEL_MARKER_SIZE

            for position, values in enumerate(label_sources):
                if identity >= len(values):
                    # a label source can be short an identity if it was built for a different
                    # pose file. skipping the marker keeps the rest of the group in place.
                    continue

                label_val = int(values[identity][self.parent.current_frame])
                marker_color = label_marker_color(label_val, self.parent.label_color_lut)
                marker_x = group_x + position * (LABEL_MARKER_SIZE + LABEL_MARKER_PAIR_GAP)

                draw_label_marker(painter, marker_x, marker_y, LABEL_MARKER_SIZE, marker_color)
