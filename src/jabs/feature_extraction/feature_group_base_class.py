import abc
import typing

from jabs.pose_estimation import PoseEstimation

from .feature_base_class import Feature


class FeatureGroup(abc.ABC):
    """Abstract base class for groups of related feature extraction modules.

    This class manages a collection of feature modules, providing methods to compute per-frame and windowed features
    for a given subject identity. It also handles enabling/disabling features, querying supported features based on
    pose version and static objects, and provides class-level metadata.

    Methods:
        per_frame(identity): Compute per-frame features for a specific identity.
        window(identity, window_size, per_frame_values): Compute windowed features for a specific identity.
        enabled_features: Property returning the names of currently enabled features.
        module_names(): Class method returning all feature names in this group.
        name(): Class method returning the group name.
        get_supported_feature_modules(pose_version, static_objects, **kwargs): Class method returning supported features.
        _init_feature_mods(identity): Abstract method to initialize feature modules for an identity.
    """

    # to be defined in subclass
    _features: typing.ClassVar[dict[str, Feature]] = {}
    _name = None

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__()
        self._enabled_features = []
        self._poses = poses
        self._pixel_scale = pixel_scale
        if self._name is None:
            raise NotImplementedError("Base class must override _name class member")

        # _features above defines all features that are part of this group,
        # but self._enabled_features lists which features are currently enabled.
        # by default, all features are turned on
        self._enabled_features = list(self._features.keys())

    def per_frame(self, identity: int) -> dict:
        """compute the value of the per frame features for a specific identity

        Args:
            identity: identity to compute features for

        Returns:
            dict where each key is the name of a feature module included in this FeatureGroup
        """
        feature_modules = self._init_feature_mods(identity)
        return {name: mod.per_frame(identity) for name, mod in feature_modules.items()}

    def window(self, identity: int, window_size: int, per_frame_values: dict) -> dict:
        """compute window feature values for a given identities per frame values

        Args:
            identity: subject identity
            window_size: window size NOTE: (actual window size is 2 *
                window_size + 1)
            per_frame_values: per frame feature values

        Returns:
            dictionary where keys are feature module names that are part
            of this FeatureGroup. The value for each element is the window feature
            dict returned by that module.
        """
        feature_modules = self._init_feature_mods(identity)
        return {
            name: mod.window(identity, window_size, per_frame_values[name])
            for name, mod in feature_modules.items()
        }

    @property
    def enabled_features(self):
        """return the names of the features that are currently enabled in this group"""
        return self._enabled_features

    @abc.abstractmethod
    def _init_feature_mods(self, identity: int) -> dict:
        pass

    @classmethod
    def module_names(cls):
        """return the names of the features in this group"""
        return list(cls._features.keys())

    @classmethod
    def name(cls):
        """return the name of this feature group"""
        return cls._name

    @classmethod
    def get_supported_feature_modules(
        cls,
        pose_version: int,
        static_objects: set[str],
        **kwargs,
    ) -> list[str]:
        """Get the features supported by this group based on the pose version, static objects, and optional additional attributes

        Args:
            pose_version (int): version of the pose estimation file
            static_objects (set[str]): set of static objects available to the project
            **kwargs: additional keyword arguments that may be used by specific feature classes
        """
        features = []
        for feature_name, feature_class in cls._features.items():
            if feature_class.is_supported(pose_version, static_objects, **kwargs):
                features.append(feature_name)
        return features
