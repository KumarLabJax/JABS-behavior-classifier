import abc
import typing

from jabs.pose_estimation import PoseEstimation
from .feature_base_class import Feature


class FeatureGroup(abc.ABC):

    # to be defined in subclass
    _features: typing.Dict[str, typing.Type[Feature]] = {}
    _name = None

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__()
        self._enabled_features = []
        self._poses = poses
        self._pixel_scale = pixel_scale
        if self._name is None:
            raise NotImplementedError(
                "Base class must override _name class member")

        # _features above defines all features that are part of this group,
        # but self._enabled_features lists which features are currently enabled.
        # by default, all features are turned on
        self._enabled_features = list(self._features.keys())

    def per_frame(self, identity: int) -> typing.Dict:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: dict where each key is the name of a feature module included
        in this FeatureGroup
        """
        feature_modules = self._init_feature_mods(identity)
        return {
            name: mod.per_frame(identity) for name, mod in
            feature_modules.items()
        }

    def window(self, identity: int, window_size: int,
               per_frame_values: typing.Dict) -> typing.Dict:
        """
        compute window feature values for a given identities per frame values
        :param identity: subject identity
        :param window_size: window size
          NOTE: (actual window size is 2 * window_size + 1)
        :param per_frame_values: per frame feature values
        :return: dictionary where keys are feature module names that are part
        of this FeatureGroup. The value for each element is the window feature
        dict returned by that module.
        """
        feature_modules = self._init_feature_mods(identity)
        return {
            name: mod.window(identity, window_size, per_frame_values[name]) for name, mod in
            feature_modules.items()
        }

    @property
    def enabled_features(self):
        return self._enabled_features

    @abc.abstractmethod
    def _init_feature_mods(self, identity: int) -> dict:
        pass

    @classmethod
    def module_names(cls):
        return list(cls._features.keys())

    @classmethod
    def name(cls):
        return cls._name

    @classmethod
    def get_supported_feature_modules(
            cls,
            pose_version: int,
            static_objects: typing.List[str]
    ) -> typing.List[str]:
        """

        :param pose_version:
        :param static_objects:
        :return:
        """
        features = []
        for feature_name, feature_class in cls._features.items():
            if feature_class.is_supported(pose_version, static_objects):
                features.append(feature_name)

        return features
