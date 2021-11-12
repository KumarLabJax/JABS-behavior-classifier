import abc
import typing

from src.pose_estimation import PoseEstimation


class FeatureGroup(abc.ABC):

    # to be defined in subclass
    _features = {}
    _name = None

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__()
        self._config = []
        self._poses = poses
        self._pixel_scale = pixel_scale
        if self._name is None:
            raise NotImplementedError(
                "Base class must override _name class member")

        # _features above defines all features that are part of this group,
        # but self._config lists which features are currently enabled
        # by default, all features are turned on
        self._config = list(self._features.keys())

    def per_frame(self, identity: int) -> dict:
        feature_modules = self._init_feature_mods(identity)
        return {
            name: mod.per_frame(identity) for name, mod in
            feature_modules.items()
        }

    def window(self, identity: int, window_size: int,
               per_frame_values: typing.Dict) -> typing.Dict:
        feature_modules = self._init_feature_mods(identity)
        return {
            name: mod.window(identity, window_size, per_frame_values[name]) for name, mod in
            feature_modules.items()
        }

    @property
    def feature_names(self):
        return {
            feature: self._features[feature].feature_names()
            for feature in self._config
        }

    @property
    def window_feature_names(self):
        features = {}
        for feature_mod in sorted(self._config):
            features[feature_mod] = {}
            for feature_name in self._features[feature_mod].feature_names():
                features[feature_mod][feature_name] = list(self._features[feature_mod]._window_operations.keys())
        return features

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: list):
        self._config = config

    @abc.abstractmethod
    def _init_feature_mods(self, identity: int) -> dict:
        pass

    @classmethod
    def module_names(cls):
        return list(cls._features.keys())

    @classmethod
    def name(cls):
        return cls._name
