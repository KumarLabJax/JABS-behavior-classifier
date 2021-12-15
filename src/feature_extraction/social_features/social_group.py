from src.feature_extraction.feature_group_base_class import FeatureGroup
from src.pose_estimation import PoseEstimation
# import all feature modules for this group
from . import ClosestDistances, ClosestFovDistances, ClosestFovAngles, \
    PairwiseSocialDistances, PairwiseSocialFovDistances
from .social_distance import ClosestIdentityInfo


class SocialFeatureGroup(FeatureGroup):

    _name = 'social'

    # build dictionary mapping feature name to class that implements it
    _features = {
        ClosestDistances.name(): ClosestDistances,
        ClosestFovAngles.name(): ClosestFovAngles,
        ClosestFovDistances.name(): ClosestFovDistances,
        PairwiseSocialDistances.name(): PairwiseSocialDistances,
        PairwiseSocialFovDistances.name(): PairwiseSocialFovDistances
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
        self._closest_identities_cache = None

    def _init_feature_mods(self, identity: int):
        """
        initialize all of the feature modules specified in the current config
        :param identity: subject identity to use when computing social features
        :return: dictionary of initialized feature modules for this group
        """

        # cache the most recent ClosestIdentityInfo, it's needed by
        # the IdentityFeatures class when saving the social features to the
        # h5 file
        self._closest_identities_cache = ClosestIdentityInfo(
            self._poses, identity, self._pixel_scale)

        # initialize all of the feature modules specified in the current config
        return {
            feature: self._features[feature](self._poses, self._pixel_scale,
                                             self._closest_identities_cache)
            for feature in self._enabled_features
        }

    @property
    def closest_identities(self):
        """
        return the closest identities computed during the last call to
        per_frame() (per_frame inherited from super class)
        """
        return self._closest_identities_cache
