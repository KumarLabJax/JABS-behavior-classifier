import typing

from jabs.feature_extraction.feature_group_base_class import FeatureGroup
from jabs.pose_estimation import PoseEstimation

from ..feature_base_class import Feature
from .closest_distances import ClosestDistances
from .closest_fov_angles import ClosestFovAngles
from .closest_fov_distances import ClosestFovDistances
from .pairwise_social_distances import (
    PairwiseSocialDistances,
    PairwiseSocialFovDistances,
)
from .social_distance import ClosestIdentityInfo


class SocialFeatureGroup(FeatureGroup):
    """A feature group for extracting social interaction features from pose estimation data.

    This class manages the computation and caching of various social features, such as closest distances,
    field-of-view angles, and pairwise social distances, for a given subject identity. It initializes and
    provides access to feature modules relevant to social behavior analysis.

    Args:
        poses (PoseEstimation): Pose estimation data for a video.
        pixel_scale (float): Scale factor to convert pixel distances to real-world units (cm).
    """

    _name = "social"

    # build dictionary mapping feature name to class that implements it
    _features: typing.ClassVar[dict[str, Feature]] = {
        ClosestDistances.name(): ClosestDistances,
        ClosestFovAngles.name(): ClosestFovAngles,
        ClosestFovDistances.name(): ClosestFovDistances,
        PairwiseSocialDistances.name(): PairwiseSocialDistances,
        PairwiseSocialFovDistances.name(): PairwiseSocialFovDistances,
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
        self._closest_identities_cache = None

    def _init_feature_mods(self, identity: int):
        """initialize all of the feature modules specified in the current config

        Args:
            identity: subject identity to use when computing social
                features

        Returns:
            dictionary of initialized feature modules for this group
        """
        # cache the most recent ClosestIdentityInfo, it's needed by
        # the IdentityFeatures class when saving the social features to the
        # h5 file
        self._closest_identities_cache = ClosestIdentityInfo(
            self._poses, identity, self._pixel_scale
        )

        # initialize all the feature modules specified in the current config
        return {
            feature: self._features[feature](
                self._poses, self._pixel_scale, self._closest_identities_cache
            )
            for feature in self._enabled_features
        }

    @property
    def closest_identities(self):
        """return cached closet identities"""
        return self._closest_identities_cache
