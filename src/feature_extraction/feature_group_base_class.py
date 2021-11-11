import abc
import typing

from src.pose_estimation import PoseEstimation


class FeatureGroup(abc.ABC):

    # to be defined in subclass
    _features = {}

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__()
        self._config = []
        self._poses = poses
        self._pixel_scale = pixel_scale

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
    def config(self):
        return self._config

    @config.setter
    def config(self, config: list):
        self._config = config

    @abc.abstractmethod
    def _init_feature_mods(self, identity: int) -> dict:
        pass
