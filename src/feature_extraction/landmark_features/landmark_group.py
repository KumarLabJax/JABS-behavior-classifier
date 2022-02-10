from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_group_base_class import FeatureGroup
from .corner import DistanceToCorner, BearingToCorner, CornerDistanceInfo
from .lixit import DistanceToLixit


class LandmarkFeatureGroup(FeatureGroup):
    _name = "landmark"

    # build a dictionary that maps a feature name to the class that
    # implements it
    _features = {
        DistanceToCorner.name(): DistanceToCorner,
        BearingToCorner.name(): BearingToCorner,
        DistanceToLixit.name(): DistanceToLixit
    }

    # maps static objects to the names of features derived from that object
    _feature_map = {
        'corners': [DistanceToCorner.name(), BearingToCorner.name()],
        'lixit': [DistanceToLixit.name()]
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

        # only enable the features supported by this particular pose file
        self._enabled_features = []
        for o in poses.static_objects:
            self._enabled_features.extend(self.static_object_features(o))

    def _init_feature_mods(self, identity: int):
        """
        initialize all the feature modules specified in the current config
        :param identity: unused, specified by abstract base class
        :return: dictionary of initialized feature modules for this group
        """
        modules = {}

        corner_distances = CornerDistanceInfo(self._poses, self._pixel_scale)

        # initialize all the feature modules specified in the current config
        for feature in self._enabled_features:

            # the distance to corner and bearing to corner features use
            # some pre-computed data (so it doesn't have to be recomputed for
            # each). We need to special case the initialization of these
            # to pass in the "corner_distances" object
            if feature in [DistanceToCorner.name(), BearingToCorner.name()]:
                modules[feature] = self._features[feature](
                    self._poses, self._pixel_scale, corner_distances)
            else:
                modules[feature] = self._features[feature](
                    self._poses, self._pixel_scale)

        return modules

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
