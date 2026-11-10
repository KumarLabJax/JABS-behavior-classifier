from typing import ClassVar

from jabs.feature_extraction.feature_group_base_class import FeatureGroup
from jabs.pose_estimation import PoseEstimation

from ..feature_base_class import Feature
from .corner import BearingToCorner, CornerDistanceInfo, DistanceToCorner
from .food_hopper import FoodHopper
from .lixit import BearingToLixit, DistanceToLixit, LixitDistanceInfo, MouseLixitAngle


class LandmarkFeatureGroup(FeatureGroup):
    """
    Feature group for extracting features related to static landmarks in the environment.

    This class manages the extraction of features derived from static objects such as corners,
    lixits, and food hoppers, using pose estimation data. It enables and initializes only the
    features supported by the provided pose file and static objects. The class provides methods
    to retrieve feature modules, shared info objects, and to query supported features and objects.

    Args:
        poses (PoseEstimation): Pose estimation handler for the current session.
        pixel_scale (float): Scale factor to convert pixel coordinates to real-world units.

    Class Attributes:
        _features (ClassVar[dict[str, type]]): Maps feature names to their implementing classes.
        feature_map (ClassVar[dict[str, list[str]]]): Maps static object names to feature names.

    Methods:
        get_corner_info(identity): Get or create the CornerDistanceInfo for an identity.
        get_lixit_info(identity): Get or create the LixitDistanceInfo for an identity.
        static_object_features(static_object): List features derived from a static object.
        get_supported_objects(): List all supported static objects.
        get_objects_from_features(features): Get required objects for a list of features.
    """

    _name = "landmark"

    # build a dictionary that maps a feature name to the class that
    # implements it
    _features: ClassVar[dict[str, type[Feature]]] = {
        DistanceToCorner.name(): DistanceToCorner,
        BearingToCorner.name(): BearingToCorner,
        DistanceToLixit.name(): DistanceToLixit,
        BearingToLixit.name(): BearingToLixit,
        FoodHopper.name(): FoodHopper,
        MouseLixitAngle.name(): MouseLixitAngle,
    }

    # maps static objects to the names of features derived from that object
    feature_map: ClassVar[dict[str, list[str]]] = {
        "corners": [DistanceToCorner.name(), BearingToCorner.name()],
        "lixit": [
            DistanceToLixit.name(),
            BearingToLixit.name(),
            MouseLixitAngle.name(),
        ],
        "food_hopper": [FoodHopper.name()],
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

        # only enable the features supported by this particular pose file
        self._enabled_features = []
        for o in poses.static_objects:
            for feature in self.static_object_features(o):
                # make sure the feature is supported before adding it to the enabled features
                if self._features[feature].is_supported(
                    poses.format_major_version,
                    set(poses.static_objects.keys()),
                    lixit_keypoints=poses.num_lixit_keypoints,
                ):
                    self._enabled_features.append(feature)

        self._corner_info = {}
        self._lixit_info = {}

    def _init_feature_mods(self, identity: int):
        """initialize all the feature modules specified in the current config

        Args:
            identity: identity to initialize the features for

        Returns:
            dictionary of initialized feature modules for this group
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
                    self._poses, self._pixel_scale, self._corner_info[identity]
                )
            elif feature in [
                DistanceToLixit.name(),
                BearingToLixit.name(),
                MouseLixitAngle.name(),
            ]:
                modules[feature] = self._features[feature](
                    self._poses, self._pixel_scale, self._lixit_info[identity]
                )
            else:
                modules[feature] = self._features[feature](self._poses, self._pixel_scale)

        return modules

    def get_corner_info(self, identity: int):
        """gets the corner info for a specific identity

        Args:
            identity: identity to get info object for

        Returns:
            CornerDistanceInfo object for the requested identity
        """
        if identity not in self._corner_info:
            self._corner_info[identity] = CornerDistanceInfo(self._poses, self._pixel_scale)
        return self._corner_info[identity]

    def get_lixit_info(self, identity: int):
        """gets the lixit info for a specific identity

        Args:
            identity: identity to get the info object for

        Returns:
            LixitDistanceInfo object for the requested identity
        """
        if identity not in self._lixit_info:
            self._lixit_info[identity] = LixitDistanceInfo(self._poses, self._pixel_scale)
        return self._lixit_info[identity]

    @classmethod
    def static_object_features(cls, static_object: str):
        """get a list of the features derived from a given static object (landmark)

        Args:
            static_object: name of object (e.g. 'corners')

        Returns:
            list of strings containing feature names derived from that object
        """
        try:
            return cls.feature_map[static_object]
        except KeyError:
            return []

    @classmethod
    def get_supported_objects(cls):
        """get a list of all objects supported

        Returns:
            list of the object names
        """
        return cls.feature_map.keys()

    @classmethod
    def get_objects_from_features(cls, features: list):
        """gets a list of objects required to generate features

        this is the reverse of static_object_features

        Args:
            features: list of features which may include static object features

        Returns:
            list of objects needed to generate the features
        """
        found_objects = []
        for cur_feature in features:
            for obj, feats in cls.feature_map.items():
                if cur_feature in feats:
                    found_objects.append(obj)
        return list(set(found_objects))
