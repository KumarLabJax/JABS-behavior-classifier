from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui

from jabs.pose_estimation import PoseEstimation
from jabs.ui.colors import KEYPOINT_COLOR_MAP
from jabs.utils.pose_util import gen_line_fragments

from .overlay import Overlay

if TYPE_CHECKING:
    from jabs.ui.player_widget.frame_with_overlays import FrameWithOverlaysWidget


_LINE_SEGMENT_COLOR = QtGui.QColor(255, 255, 255, 128)  # color for the pose line segments
_KEYPOINT_SIZE = 3  # size of the keypoint circles


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

        # draw the pose estimation skeletons
        for identity in self.parent.pose.identities:
            if not all_identities and identity != self.parent.active_identity:
                continue

            points, mask = self.parent.pose.get_points(self.parent.current_frame, identity)

            if points is None or mask is None:
                continue

            # Adjust alpha for non-active identities
            if identity != self.parent.active_identity:
                line_color = QtGui.QColor(_LINE_SEGMENT_COLOR)
                line_color.setAlpha(line_color.alpha() // 3)  # More translucent
            else:
                line_color = _LINE_SEGMENT_COLOR

            pen = QtGui.QPen(line_color)
            pen.setWidth(3)
            painter.setPen(pen)

            for seg in gen_line_fragments(
                self.parent.pose.get_connected_segments(), np.flatnonzero(mask == 0)
            ):
                segment_points = [
                    self.parent.image_to_widget_coords_cropped(p[0], p[1], crop_rect)
                    for p in points[seg]
                ]
                # Filter out points outside the crop
                segment_points = [pt for pt in segment_points if pt is not None]

                # draw lines
                if len(segment_points) >= 2:
                    for i in range(len(segment_points) - 1):
                        painter.drawLine(
                            segment_points[i][0],
                            segment_points[i][1],
                            segment_points[i + 1][0],
                            segment_points[i + 1][1],
                        )

            # draw points at each keypoint of the pose (if it exists at this frame)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for keypoint in PoseEstimation.KeypointIndex:
                point_index = keypoint.value
                if mask[point_index]:
                    widget_coords = self.parent.image_to_widget_coords_cropped(
                        points[point_index][0], points[point_index][1], crop_rect
                    )
                    if widget_coords is None:
                        continue

                    widget_x, widget_y = widget_coords
                    color = KEYPOINT_COLOR_MAP[keypoint]
                    if identity != self.parent.active_identity:
                        # Make keypoints translucent for non-active identities
                        translucent_color = QtGui.QColor(color)
                        translucent_color.setAlpha(96)
                        painter.setBrush(translucent_color)
                    else:
                        painter.setBrush(color)

                    painter.drawEllipse(
                        QtCore.QPoint(widget_x, widget_y), _KEYPOINT_SIZE, _KEYPOINT_SIZE
                    )
