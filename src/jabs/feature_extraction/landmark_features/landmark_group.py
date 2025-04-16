from jabs.pose_estimation import PoseEstimation
from jabs.feature_extraction.feature_group_base_class import FeatureGroup
from .corner import DistanceToCorner, BearingToCorner, CornerDistanceInfo
from .lixit import DistanceToLixit, BearingToLixit, LixitDistanceInfo
from .food_hopper import FoodHopper


class LandmarkFeatureGroup(FeatureGroup):
    _name = "landmark"

    # build a dictionary that maps a feature name to the class that
    # implements it
    _features = {
        DistanceToCorner.name(): DistanceToCorner,
        BearingToCorner.name(): BearingToCorner,
        DistanceToLixit.name(): DistanceToLixit,
        BearingToLixit.name(): BearingToLixit,
        FoodHopper.name(): FoodHopper
    }

    # maps static objects to the names of features derived from that object
    feature_map = {
        'corners': [DistanceToCorner.name(), BearingToCorner.name()],
        'lixit': [DistanceToLixit.name(), BearingToLixit.name()],
        'food_hopper': [FoodHopper.name()]
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

        # only enable the features supported by this particular pose file
        self._enabled_features = []
        for o in poses.static_objects:
            self._enabled_features.extend(self.static_object_features(o))

        self._corner_info = {}
        self._lixit_info = {}

    def _init_feature_mods(self, identity: int):
        """
        initialize all the feature modules specified in the current config
        :param identity: identity to initialize the features for
        :return: dictionary of initialized feature modules for this group
        """
        modules = {}

        if identity not in self._corner_info:
            self._corner_info[identity] = CornerDistanceInfo(self._poses, self._pixel_scale)

        if identity not in self._lixit_info:
            self._lixit_info[identity] = LixitDistanceInfo(self._poses, self._pixel_scale)

        # initialize all the feature modules specified in the current config
        for feature in self._enabled_features:

            # corner and lixit features require a special initializer with
            # the shared-info object
            if feature in [DistanceToCorner.name(), BearingToCorner.name()]:
                modules[feature] = self._features[feature](
                    self._poses, self._pixel_scale, self._corner_info[identity])
            elif feature in [DistanceToLixit.name(), BearingToLixit.name()]:
                modules[feature] = self._features[feature](
                    self._poses, self._pixel_scale, self._lixit_info[identity])
            else:
                modules[feature] = self._features[feature](
                    self._poses, self._pixel_scale)

        return modules

    def get_corner_info(self, identity: int):
        """
        gets the corner info for a specific identity
        :param identity: identity to get info object for
        :return: CornerDistanceInfo object for the requested identity
        """
        if identity not in self._corner_info:
            self._corner_info[identity] = CornerDistanceInfo(self._poses, self._pixel_scale)
        return self._corner_info[identity]

    def get_lixit_info(self, identity: int):
        """
        gets the lixit info for a specific identity
        :param identity: identity to get the info object for
        :return: LixitDistanceInfo object for the requested identity
        """
        if identity not in self._lixit_info:
            self._lixit_info[identity] = LixitDistanceInfo(self._poses, self._pixel_scale)
        return self._lixit_info[identity]

    @classmethod
    def static_object_features(cls, static_object: str):
        """
        get a list of the features derived from a given static object (landmark)
        :param static_object: name of object (e.g. 'corners')
        :return: list of strings containing feature names derived from that
        object
        """
        try:
            return cls.feature_map[static_object]
        except KeyError:
            return []

    @classmethod
    def static_object_per_frame_features(cls, feature_name: str):
        """
        get a list of per frame features derived from a static object feature
        :param feature_name: name of feature to retrieve the per frame feature names
        :return: list of strings containing per frame features derived from that object
        """
        try:
            return cls._features[feature_name].feature_names()
        except KeyError:
            return []

    @classmethod
    def static_object_window_features(cls, feature_name: str):
        """
        get a list of window features derived from a static object feature
        :param feature_name: name of feature to retrieve the window feature names
        :return: list of strings containing window features derived from that object
        """
        try:
            return list(cls._features[feature_name]._window_operations.keys())
        except KeyError:
            return []

    @classmethod
    def get_supported_objects(cls):
        """
        get a list of all objects supported
        :return: list of the object names
        """
        return cls.feature_map.keys()

    @classmethod
    def get_objects_from_features(cls, features: list):
        """
        gets a list of objects required to generate features
        this is the reverse of static_object_features
        :param features: list of features which may include static object features
        :return: list of objects needed to generate the features
        """
        found_objects = []
        for cur_feature in features:
            for obj, feats in cls.feature_map.items():
                if cur_feature in feats:
                    found_objects.append(obj)
        return list(set(found_objects))

    @classmethod
    def get_feature_names(cls, static_objects: list = None):
        """
        get the features supported
        :param static_objects: list of static objects. if None, get features for all supported static objects
        :return:
        """
        if static_objects is None:
            valid_objects = cls.get_supported_objects()
        else:
            valid_objects = [x for x in static_objects if x in cls.get_supported_objects()]

        per_frame_features = {}
        window_features = {}
        for static_object in valid_objects:
            object_features = cls.static_object_features(static_object)
            for current_feature in object_features:
                object_feature_list = []
                window_feature_dict = {}
                per_frame_feature_names = cls.static_object_per_frame_features(current_feature)
                window_feature_mods = cls.static_object_window_features(current_feature)
                for frame_feature in per_frame_feature_names:
                    window_feature_dict.update({frame_feature: window_feature_mods})
                object_feature_list += per_frame_feature_names
                per_frame_features.update({current_feature: object_feature_list})
                window_features.update({current_feature: window_feature_dict})

        return per_frame_features, window_features
