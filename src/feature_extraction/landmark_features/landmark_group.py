from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_group_base_class import FeatureGroup
from .corner import DistanceToCorner, BearingToCorner, CornerDistanceInfo


class LandmarkFeatureGroup(FeatureGroup):
    _name = "landmark"

    # build a dictionary that maps a feature name to the class that
    # implements it
    _features = {
        DistanceToCorner.name(): DistanceToCorner,
        BearingToCorner.name(): BearingToCorner,
    }

    # maps static objects to the names of features derived from that object
    _feature_map = {
        'corners': [DistanceToCorner.name(), BearingToCorner.name()]
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

        # only enable the features supported by this particular pose file
        self._config = []
        for o in poses.static_objects:
            self._config.extend(self.static_object_features(o))

    def _init_feature_mods(self, identity: int):
        """
        initialize all of the feature modules specified in the current config
        :param identity: unused, specified by abstract base class
        :return: dictionary of initialized feature modules for this group
        """

        corner_distances = CornerDistanceInfo(self._poses, self._pixel_scale)

        return {
            DistanceToCorner.name(): DistanceToCorner(self._poses,
                                                      self._pixel_scale,
                                                      corner_distances),
            BearingToCorner.name(): BearingToCorner(self._poses,
                                                    self._pixel_scale,
                                                    corner_distances),
        }

    @classmethod
    def static_object_features(cls, static_object: str):
        """
        get a list of the features derived from a given static object (landmark)
        :param static_object: name of object (e.g. 'corners')
        :return: list of strings containing feature names derived from that
        object
        """
        try:
            return cls._feature_map[static_object]
        except KeyError:
            return []
