from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from jabs.overlay_drawing import draw_identity_segmentation

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class SegmentationOverlay(Overlay):
    """Overlay for displaying segmentation contours on the video frame.

    Disabled until the user turns it on with "View > Overlay Segmentation", and a no-op
    for a pose file that carries no contours.
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

        for identity in self.parent.pose.identities:
            draw_identity_segmentation(
                painter,
                self.parent.pose,
                self.parent.current_frame,
                identity,
                to_output=lambda x, y: self.parent.image_to_widget_coords_cropped(x, y, crop_rect),
                line_width=line_width,
                active=(identity == self.parent.active_identity),
            )
