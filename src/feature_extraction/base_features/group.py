import typing

import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_group_base_class import FeatureGroup

from . import Angles, PairwisePointDistances, PointMask, PointSpeeds, \
    CentroidVelocityMag, CentroidVelocityDir


class BaseFeatureGroup(FeatureGroup):

    _features = {
        PairwisePointDistances.name(): PairwisePointDistances,
        Angles.name(): Angles,
        PointMask.name(): PointMask,
        PointSpeeds.name(): PointSpeeds,
        CentroidVelocityDir.name(): CentroidVelocityDir,
        CentroidVelocityMag.name(): CentroidVelocityMag,
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

        # by default, all features are turned on
        self._config = list(self._features.keys())

    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> typing.Dict:
        """
        over ride window method so we can apply a speical case to
        point_mask
        """
        # call super class window() method and then massage the output
        values = super().window(identity, window_size, per_frame_values)
        # point_mask is not included in window features, remove it from the
        # dictionary before returning it
        del values['point_mask']
        return values

    def _init_feature_mods(self, identity: int):
        """
        initialize all of the feature modules specified in the current config
        :param identity: unused, specified by abstract base class
        :return: dictionary of initialized feature modules for this group
        """
        return {
            feature: self._features[feature](self._poses, self._pixel_scale)
            for feature in self._config
        }
