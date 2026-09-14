from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from jabs.overlay_drawing import (
    CLOSEST_FOV_MARKER_COLOR,
    CLOSEST_MARKER_COLOR,
    CLOSEST_MARKER_RADIUS,
    draw_identity_marker,
)

from ..closest_identity import HALF_FOV_DEGREES, closest_identity
from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class ClosestIdentityOverlay(Overlay):
    """Overlay marking the animal nearest the active identity.

    Two markers, as the cv2 drawing had: one on the nearest animal the subject can see,
    and one on the nearest animal overall when that is a different animal. Disabled
    until the user asks for it with the "?" key.

    Unlike the other overlays this one has to compute something - hull distances and
    view angles across every identity - so the answer is cached per frame and active
    identity. A repaint can be triggered by things that do not change either (a resize,
    a crop, another overlay being toggled), and that must not redo the measurement.
    """

    def __init__(self, parent: "FrameWithOverlaysWidget") -> None:
        super().__init__(parent)
        self._enabled = False
        self._cache_key: tuple[int, int] | None = None
        self._cached: tuple[int | None, int | None] = (None, None)

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints markers on the animals nearest the active identity.

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

        closest_fov_id, closest_id = self._closest_identities()
        if closest_fov_id is None and closest_id is None:
            return

        zoom = self.parent.scaled_pix_width / max(crop_rect.width(), 1)
        radius = max(1, round(CLOSEST_MARKER_RADIUS * zoom))

        # The in-view animal is drawn second so it stays visible when one animal is
        # both the nearest overall and the nearest in view.
        for identity, color in (
            (closest_id, CLOSEST_MARKER_COLOR),
            (closest_fov_id, CLOSEST_FOV_MARKER_COLOR),
        ):
            if identity is None:
                continue
            centroid = self.get_centroid(identity)
            if centroid is None:
                continue
            widget_coords = self.parent.image_to_widget_coords_cropped(
                centroid.x, centroid.y, crop_rect
            )
            if widget_coords is None:
                continue
            draw_identity_marker(painter, widget_coords[0], widget_coords[1], radius, color)

    def _closest_identities(self) -> tuple[int | None, int | None]:
        """Return ``(closest in view, closest overall)``, measured once per frame.

        The nearest animal overall is reported only when it is not already the nearest
        one in view, matching what the cv2 drawing marked.
        """
        key = (self.parent.current_frame, self.parent.active_identity)
        if key == self._cache_key:
            return self._cached

        frame, subject = key
        closest_fov_id = closest_identity(self.parent.pose, subject, frame, HALF_FOV_DEGREES)
        closest_id = closest_identity(self.parent.pose, subject, frame)
        if closest_id == closest_fov_id:
            closest_id = None

        self._cache_key = key
        self._cached = (closest_fov_id, closest_id)
        return self._cached
