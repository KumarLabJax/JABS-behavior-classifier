from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui

from jabs.pose_estimation import PoseEstimation
from jabs.ui.colors import KEYPOINT_COLOR_MAP
from jabs.utils.pose_util import gen_line_fragments

from .overlay import Overlay

if TYPE_CHECKING:
    from jabs.ui.player_widget.frame_with_control_overlay import FrameWidgetWithInteractiveOverlays


_LINE_SEGMENT_COLOR = QtGui.QColor(255, 255, 255, 128)  # color for the pose line segments
_KEYPOINT_SIZE = 3  # size of the keypoint circles


class PoseOverlay(Overlay):
    """Overlay for displaying pose keypoints and skeletons on the video frames."""

    def __init__(self, parent: "FrameWidgetWithInteractiveOverlays") -> None:
        super().__init__(parent)

    def paint(self, painter: QtGui.QPainter) -> None:
        """Paints pose keypoints and skeletons on the current frame."""
        if not self._enabled or self.parent.pixmap().isNull():
            return

        if self.parent.pose_overlay_mode == self.parent.PoseOverlayMode.ALL:
            self._overlay_pose(painter, all_identities=True)
        elif self.parent.pose_overlay_mode == self.parent.PoseOverlayMode.ACTIVE_IDENTITY:
            self._overlay_pose(painter, all_identities=False)

    def _overlay_pose(self, painter: QtGui.QPainter, all_identities: bool = False) -> None:
        """Overlay pose estimation on the current frame.

        This method draws the pose estimation skeletons on the frame. If `all_identities` is True,
        it will draw all identities; otherwise, it will only draw the active identity with a more prominent color.

        Args:
            painter (QtGui.QPainter): The painter used to draw on the frame.
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
                    self.parent.image_to_widget_coords(p[0], p[1]) for p in points[seg]
                ]

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
                    widget_x, widget_y = self.parent.image_to_widget_coords(
                        points[point_index][0], points[point_index][1]
                    )

                    # Use the color map to get the color for the keypoint
                    # and make it translucent if it's not the active identity
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
