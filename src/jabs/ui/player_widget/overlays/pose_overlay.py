from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui

from ..pose_drawing import KEYPOINT_SIZE, draw_identity_pose
from .overlay import Overlay

if TYPE_CHECKING:
    from jabs.ui.player_widget.frame_with_overlays import FrameWithOverlaysWidget


class PoseOverlay(Overlay):
    """Overlay for displaying pose keypoints and connecting line segments on the video frame."""

    def __init__(self, parent: "FrameWithOverlaysWidget") -> None:
        super().__init__(parent)

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints pose keypoints and connecting line segments on the current frame.

        Args:
            painter (QtGui.QPainter): The painter used to draw on the widget.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if not self._enabled or self.parent.pixmap().isNull():
            return

        if self.parent.pose_overlay_mode == self.parent.PoseOverlayMode.ALL:
            self._overlay_pose(painter, crop_rect, all_identities=True)
        elif self.parent.pose_overlay_mode == self.parent.PoseOverlayMode.ACTIVE_IDENTITY:
            self._overlay_pose(painter, crop_rect, all_identities=False)

    def _overlay_pose(
        self, painter: QtGui.QPainter, crop_rect: QtCore.QRect, all_identities: bool = False
    ) -> None:
        """Overlay pose estimation on the current frame.

        This method draws the pose estimation skeletons on the frame. If `all_identities` is True,
        it will draw all identities; otherwise, it will only draw the active identity with a more prominent color.

        Args:
            painter (QtGui.QPainter): The painter used to draw on the frame.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.
            all_identities (bool): If True, draw all identities; if False, only draw the active identity.
        """
        if self.parent.pose is None:
            return

        # Keypoint size scales with the on-screen zoom so markers stay a sensible size.
        zoom = self.parent.scaled_pix_width / max(crop_rect.width(), 1)
        keypoint_size = max(1, round(KEYPOINT_SIZE * zoom**0.8))

        for identity in self.parent.pose.identities:
            if not all_identities and identity != self.parent.active_identity:
                continue

            draw_identity_pose(
                painter,
                self.parent.pose,
                self.parent.current_frame,
                identity,
                to_output=lambda x, y: self.parent.image_to_widget_coords_cropped(x, y, crop_rect),
                keypoint_size=keypoint_size,
                line_width=3,
                active=(identity == self.parent.active_identity),
            )
