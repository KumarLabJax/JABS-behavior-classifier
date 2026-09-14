from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from jabs.overlay_drawing import (
    TRACK_FUTURE_RADIUS,
    TRACK_LINE_WIDTH,
    TRACK_PAST_RADIUS,
    draw_identity_track,
)

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class TrackOverlay(Overlay):
    """Overlay showing where the active identity's nose has been and is going.

    Disabled until the user turns it on with "View > Show Track". Only the active
    identity is drawn: the track is there to follow one animal, and drawing every
    animal's would be unreadable.
    """

    def __init__(self, parent: "FrameWithOverlaysWidget") -> None:
        super().__init__(parent)
        self._enabled = False

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints the active identity's track on the current frame.

        Args:
            painter (QtGui.QPainter): The painter used to draw on the widget.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if (
            not self._enabled
            or self.parent.pixmap().isNull()
            or self.parent.pose is None
            or self.parent.active_identity is None
        ):
            return

        zoom = self.parent.scaled_pix_width / max(crop_rect.width(), 1)

        draw_identity_track(
            painter,
            self.parent.pose,
            self.parent.current_frame,
            self.parent.active_identity,
            to_output=lambda x, y: self.parent.image_to_widget_coords_cropped(x, y, crop_rect),
            future_radius=max(1, round(TRACK_FUTURE_RADIUS * zoom)),
            past_radius=max(1, round(TRACK_PAST_RADIUS * zoom)),
            line_width=max(1, round(TRACK_LINE_WIDTH * zoom)),
        )
