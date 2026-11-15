from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from jabs.overlay_drawing import (
    SEGMENTATION_NO_PREDICTION_COLOR,
    draw_identity_segmentation,
    segmentation_prediction_color,
)

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class SegmentationOverlay(Overlay):
    """Overlay for displaying segmentation contours on the video frame.

    Disabled until the user turns it on with "View > Segmentation > Show Segmentation",
    and a no-op for a pose file that carries no contours. Contours are colored either by
    identity (the active identity stands out) or by each identity's prediction on the
    current frame, following the frame widget's ``segmentation_color_mode``.
    """

    def __init__(self, parent: "FrameWithOverlaysWidget") -> None:
        super().__init__(parent)
        self._enabled = False

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints segmentation contours for every identity on the current frame.

        Args:
            painter (QtGui.QPainter): The painter used to draw on the widget.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if not self._enabled or self.parent.pixmap().isNull() or self.parent.pose is None:
            return

        # Contours were a single pixel in image space when cv2 drew them into the frame,
        # so they thickened with the zoom. Scaling the pen keeps that appearance.
        zoom = self.parent.scaled_pix_width / max(crop_rect.width(), 1)
        line_width = max(1, round(zoom))

        by_prediction = (
            self.parent.segmentation_color_mode == self.parent.SegmentationColorMode.PREDICTION
        )

        for identity in self.parent.pose.identities:
            draw_identity_segmentation(
                painter,
                self.parent.pose,
                self.parent.current_frame,
                identity,
                to_output=lambda x, y: self.parent.image_to_widget_coords_cropped(x, y, crop_rect),
                line_width=line_width,
                active=self.parent.is_drawn_active(identity),
                color=self._prediction_color(identity) if by_prediction else None,
            )

    def _prediction_color(self, identity: int) -> QtGui.QColor:
        """Return the contour color for an identity's prediction on the current frame."""
        predictions = self.parent.segmentation_predictions
        frame = self.parent.current_frame
        if predictions is None or identity >= len(predictions):
            return SEGMENTATION_NO_PREDICTION_COLOR
        identity_predictions = predictions[identity]
        if frame >= len(identity_predictions):
            # predictions come from the saved record and can be shorter than the video
            return SEGMENTATION_NO_PREDICTION_COLOR
        return segmentation_prediction_color(
            identity_predictions[frame], self.parent.segmentation_color_lut
        )
