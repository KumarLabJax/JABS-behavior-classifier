from src.feature_extraction.feature_group_base_class import FeatureGroup
from src.pose_estimation import PoseEstimation
# import all feature modules for this group
from . import ClosestDistances, ClosestFovDistances, ClosestFovAngles, \
    PairwiseSocialDistances, PairwiseSocialFovDistances
from .social_distance import ClosestIdentityInfo


class SocialFeatureGroup(FeatureGroup):

    # build dictionary mapping feature name to class that implements it
    _features = {
        ClosestDistances.name: ClosestDistances,
        ClosestFovAngles.name: ClosestFovAngles,
        ClosestFovDistances.name: ClosestFovDistances,
        PairwiseSocialDistances.name: PairwiseSocialDistances,
        PairwiseSocialFovDistances.name: PairwiseSocialFovDistances
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

        # by default, all features are turned on
        self._config = list(self._features.keys())

    def _init_feature_mods(self, identity: int):
        """
        initialize all of the feature modules specified in the current config
        :param identity: subject identity to use when computing social features
        :return: dictionary of initialized feature modules for this group
        """
        closest_identities = ClosestIdentityInfo(self._poses, identity,
                                                 self._pixel_scale)

        # initialize all of the feature modules specified in the current config
        return {
            feature: self._features[feature](self._poses, self._pixel_scale,
                                             closest_identities)
            for feature in self._config
        }
