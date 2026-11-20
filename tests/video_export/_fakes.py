"""Shared stand-ins for the video-export tests."""

import numpy as np
from shapely.geometry import Polygon

from jabs.core.pose_skeleton import FULL_CONNECTED_SEGMENTS, KeypointIndex
from jabs.core.types import SegmentationWindow
from jabs.pose_estimation.pose_est_v6 import pack_segmentation

# Uniform frame value, so any pixel change is attributable to the overlay.
BACKGROUND = 60
WIDTH, HEIGHT, FRAMES = 160, 120, 10


class StubPose:
    """Minimal PoseEstimation stand-in covering what the renderer touches."""

    def __init__(
        self,
        num_frames: int = FRAMES,
        identities: list[int] | None = None,
        has_segmentation: bool = False,
        hulls_present: bool = True,
        contours: bool = False,
    ) -> None:
        self.num_frames = num_frames
        self.identities = [0] if identities is None else identities
        # Frames without a valid convex hull are a real case (no pose for that
        # identity), and the prediction marker has nowhere to sit on them.
        self._hulls_present = hulls_present
        # Segmentation is optional even in v6+ pose files, so the renderer keys off
        # this rather than the pose version.
        self.has_segmentation = has_segmentation
        self.has_bounding_boxes = False
        self.segmentation_calls: list[tuple[int, int]] = []
        # The PoseEstimation base class defaults this to an empty dict.
        self.static_objects: dict = {}
        # Contour data, shaped the way a pose file stores it: a fixed number of
        # slots and points per identity, padded with -1 where unused.
        self._contours = None
        if contours:
            self._contours = np.full((2, 6, 2), -1, dtype=np.int32)
            self._contours[0, :4] = [(20, 20), (40, 20), (40, 40), (20, 40)]
        self._frame_offsets = np.arange(num_frames)

    def segmentation_window(self, identity: int, start: int, stop: int) -> SegmentationWindow:
        """Record the request and return contours only when asked to carry them."""
        if not self.has_segmentation:
            raise NotImplementedError("no segmentation")
        self.segmentation_calls.append((start, identity))
        contours = self._contours
        if contours is None:
            contours = np.full((2, 6, 2), -1, dtype=np.int32)
        return pack_segmentation(
            np.repeat(contours[np.newaxis], stop - start, axis=0), None, start
        )

    @staticmethod
    def get_connected_segments():
        """Return the standard full skeleton connections."""
        return FULL_CONNECTED_SEGMENTS

    def get_points(self, frame_index: int, identity: int):
        """Return points offset per identity and per frame, all marked visible.

        The offsets matter: without them two identities would draw on top of each
        other, and a frame's pose would not move over time, so tests asserting on
        either would pass vacuously.

        Indexes a real frame-backed array so that asking for a frame beyond
        ``num_frames`` raises ``IndexError``, exactly as a real pose object does.
        """
        n_kp = len(KeypointIndex)
        base = self._frame_offsets[frame_index]
        points = np.array(
            [[20 + base + i * 3, 20 + identity * 40 + i * 2] for i in range(n_kp)],
            dtype=np.float32,
        )
        return points, np.ones(n_kp, dtype=np.uint8)

    def get_identity_convex_hulls(self, identity: int):
        """Return one convex hull per frame, offset the same way as the keypoints."""
        if not self._hulls_present:
            return [None] * self.num_frames
        hulls = []
        for frame in range(self.num_frames):
            x = 40 + frame
            y = 40 + identity * 40
            hulls.append(Polygon([(x, y), (x + 8, y), (x + 8, y + 8), (x, y + 8)]))
        return hulls
