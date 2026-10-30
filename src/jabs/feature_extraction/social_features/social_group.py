"""Social feature extraction group."""

from __future__ import annotations

import logging
import typing

from jabs.feature_extraction.feature_group_base_class import FeatureGroup
from jabs.pose_estimation import PoseEstimation

from ..feature_base_class import Feature
from .closest_distances import ClosestDistances
from .closest_fov_angles import ClosestFovAngles
from .closest_fov_distances import ClosestFovDistances
from .huddle_features import HuddleFeatures
from .pairwise_social_distances import (
    PairwiseSocialDistances,
    PairwiseSocialFovDistances,
)
from .social_distance import ClosestIdentityInfo, _VideoSocialCache

logger = logging.getLogger(__name__)


class SocialFeatureGroup(FeatureGroup):
    """A feature group for extracting social interaction features from pose estimation data.

    This class manages the computation and caching of various social features, such as closest distances,
    field-of-view angles, pairwise social distances, and multi-animal huddle/group metrics for a given
    subject identity. It initializes and provides access to feature modules relevant to social behavior analysis.

    Args:
        poses (PoseEstimation): Pose estimation data for a video.
        pixel_scale (float): Scale factor to convert pixel distances to real-world units (cm).
    """

    _name = "social"

    # build dictionary mapping feature name to class that implements it
    _features: typing.ClassVar[dict[str, type[Feature]]] = {
        ClosestDistances.name(): ClosestDistances,
        ClosestFovAngles.name(): ClosestFovAngles,
        ClosestFovDistances.name(): ClosestFovDistances,
        HuddleFeatures.name(): HuddleFeatures,
        PairwiseSocialDistances.name(): PairwiseSocialDistances,
        PairwiseSocialFovDistances.name(): PairwiseSocialFovDistances,
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float) -> None:
        super().__init__(poses, pixel_scale)
        self._video_cache: _VideoSocialCache | None = None
        self._closest_identities_cache: ClosestIdentityInfo | None = None
        self._closest_identities_cache_identity: int | None = None

    def _init_feature_mods(self, identity: int) -> dict[str, Feature]:
        """Initialize all of the feature modules specified in the current config.

        Args:
            identity: Subject identity to use when computing social features.

        Returns:
            Dictionary of initialized feature modules for this group.
        """
        if self._video_cache is None:
            self._video_cache = _VideoSocialCache.build(self._poses, self._pixel_scale)

        # cache the most recent ClosestIdentityInfo, it's needed by
        # the IdentityFeatures class when saving the social features to the
        # h5 file. It is only rebuilt when the identity changes, so computing
        # per-frame and then window features for one identity builds it once.
        if (
            self._closest_identities_cache is None
            or self._closest_identities_cache_identity != identity
        ):
            self._closest_identities_cache = ClosestIdentityInfo(
                self._poses,
                identity,
                self._pixel_scale,
                video_cache=self._video_cache,
            )
            self._closest_identities_cache_identity = identity

        # initialize all the feature modules specified in the current config
        return {
            feature: self._features[feature](
                self._poses, self._pixel_scale, self._closest_identities_cache
            )
            for feature in self._enabled_features
        }

    @property
    def closest_identities(self) -> ClosestIdentityInfo | None:
        """Return cached closest identities info."""
        return self._closest_identities_cache
