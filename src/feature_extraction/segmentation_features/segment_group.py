from src.feature_extraction.feature_group_base_class import FeatureGroup
from src.pose_estimation import PoseEstimation
# import all feature modules for this group
from . import Moments

class SegmentationFeatureGroup(FeatureGroup):

    _name = 'segmentation'

    # build dictionary mapping feature name to class that implements it
    _features = {
        Moments.name(): Moments,
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
        self._closest_identities_cache = None

    def _init_feature_mods(self, identity: int):
        """
        initialize all of the feature modules specified in the current config
        :param identity: subject identity to use when computing segmentation features
        :return: dictionary of initialized feature modules for this group
        """

        return {
            feature: self._features[feature](self._poses, self._pixel_scale)
            for feature in self._enabled_features
        }
