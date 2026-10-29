"""Closest and second-closest social distance features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from jabs.feature_extraction.feature_base_class import Feature

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

    from .social_distance import ClosestIdentityInfo


class ClosestDistances(Feature):
    """Compute hull and segmentation distances between a subject and its nearest neighbors.

    Provides the convex-hull distance to the nearest other animal, the convex-hull
    distance to the second-nearest other animal (for multi-animal arenas), and the
    segmentation contour distance to the nearest other animal (falling back to convex-hull
    distance when segmentation masks are unavailable).

    Args:
        poses: Pose estimation data for a video.
        pixel_scale: Scale factor to convert pixel distances to real-world units (cm).
        social_distance_info: Object providing pre-computed closest identity information.
    """

    _name = "closest_distances"
    _min_pose = 3

    def __init__(
        self,
        poses: PoseEstimation,
        pixel_scale: float,
        social_distance_info: ClosestIdentityInfo,
    ) -> None:
        super().__init__(poses, pixel_scale)
        self._social_distance_info = social_distance_info

    def per_frame(self, identity: int) -> dict[str, npt.NDArray[np.float32]]:
        """Compute per-frame social distance features for a specific identity.

        Args:
            identity: Identity to compute features for.

        Returns:
            Dictionary mapping distance feature names to per-frame ``float32`` arrays.
        """
        return {
            "closest social distance": self._social_distance_info.compute_distances(
                self._social_distance_info.closest_identities
            ),
            "second closest social distance": (
                self._social_distance_info.second_closest_distances.copy()
            ),
            "closest social segmentation distance": (
                self._social_distance_info.closest_segmentation_distances.copy()
            ),
        }
