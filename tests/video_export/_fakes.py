"""Shared stand-ins for the video-export tests."""

import numpy as np

from jabs.pose_estimation import PoseEstimation

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
    ) -> None:
        self.num_frames = num_frames
        self.identities = [0] if identities is None else identities
        # Segmentation is optional even in v6+ pose files, so the renderer keys off
        # this rather than the pose version.
        self.has_segmentation = has_segmentation
        # overlay_segmentation() checks the pose version before asking for contours
        self.format_major_version = 6 if has_segmentation else 5
        self.segmentation_calls: list[tuple[int, int]] = []
        self._frame_offsets = np.arange(num_frames)

    def get_segmentation_data_per_frame(self, frame_index: int, identity: int):
        """Record the request; no contours, matching a file without segmentation."""
        self.segmentation_calls.append((frame_index, identity))
        return None

    @staticmethod
    def get_connected_segments():
        """Return the standard full skeleton connections."""
        return PoseEstimation.FULL_CONNECTED_SEGMENTS

    def get_points(self, frame_index: int, identity: int):
        """Return points offset per identity and per frame, all marked visible.

        The offsets matter: without them two identities would draw on top of each
        other, and a frame's pose would not move over time, so tests asserting on
        either would pass vacuously.

        Indexes a real frame-backed array so that asking for a frame beyond
        ``num_frames`` raises ``IndexError``, exactly as a real pose object does.
        """
        n_kp = len(PoseEstimation.KeypointIndex)
        base = self._frame_offsets[frame_index]
        points = np.array(
            [[20 + base + i * 3, 20 + identity * 40 + i * 2] for i in range(n_kp)],
            dtype=np.float32,
        )
        return points, np.ones(n_kp, dtype=np.uint8)
