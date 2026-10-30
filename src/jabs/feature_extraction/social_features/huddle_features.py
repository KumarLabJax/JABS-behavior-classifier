"""Huddle topology and partner quiescence social features."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from jabs.feature_extraction.feature_base_class import Feature

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

    from .social_distance import ClosestIdentityInfo

logger = logging.getLogger(__name__)


class HuddleFeatures(Feature):
    """Compute multi-animal contact count, connected group size, and partner speed per frame.

    These features enable classifiers to distinguish dyadic (2-animal) from multi-animal
    (3+ animal) huddles (including both triangular heaps and transitive linear chains
    ``A-B-C``) and to separate mutually quiescent huddling from unilateral contact when
    an active cagemate moves past a stationary subject.

    Args:
        poses: Pose estimation data for a video.
        pixel_scale: Scale factor to convert pixel distances to real-world units (cm).
        social_distance_info: Object providing pre-computed closest identity and group info.
    """

    _name = "huddle_features"
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
        """Compute per-frame huddle topology and partner speed features for an identity.

        Args:
            identity: Identity to compute features for.

        Returns:
            Dictionary mapping huddle feature names to per-frame ``float32`` arrays.
        """
        return {
            "social contact count": self._social_distance_info.social_contact_counts.copy(),
            "social connected group size": (
                self._social_distance_info.social_connected_group_sizes.copy()
            ),
            "social chain bottleneck distance": (
                self._social_distance_info.social_chain_bottleneck_distances.copy()
            ),
            "closest neighbor speed": self._social_distance_info.closest_neighbor_speeds.copy(),
            "second closest neighbor speed": (
                self._social_distance_info.second_closest_neighbor_speeds.copy()
            ),
        }
