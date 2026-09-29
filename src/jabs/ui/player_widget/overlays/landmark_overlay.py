from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from jabs.overlay_drawing import LANDMARK_LINE_WIDTH, LANDMARK_POINT_RADIUS, draw_landmarks

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class LandmarkOverlay(Overlay):
    """Overlay showing the arena landmarks a pose file records.

    Disabled until the user turns it on with "View > Overlay Landmarks", and a no-op
    for a pose file that records no static objects.
    """

    def __init__(self, parent: "FrameWithOverlaysWidget") -> None:
        super().__init__(parent)
        self._enabled = False

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints the arena landmarks on the current frame.

        Args:
            painter (QtGui.QPainter): The painter used to draw on the widget.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if not self._enabled or self.parent.pixmap().isNull() or self.parent.pose is None:
            return

        zoom = self.parent.scaled_pix_width / max(crop_rect.width(), 1)

        draw_landmarks(
            painter,
            self.parent.pose,
            to_output=lambda x, y: self.parent.image_to_widget_coords_cropped(x, y, crop_rect),
            point_radius=max(1, round(LANDMARK_POINT_RADIUS * zoom)),
            line_width=max(1, round(LANDMARK_LINE_WIDTH * zoom)),
        )
