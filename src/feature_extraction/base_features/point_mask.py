import typing

import numpy as np

from src.feature_extraction.feature_base_class import Feature
from src.pose_estimation import PoseEstimation


class PointMask(Feature):

    _name = 'point_mask'

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    @property
    def feature_names(self) -> typing.List[str]:
        return [
            f"{p.name} point mask" for p in PoseEstimation.KeypointIndex
        ]

    def per_frame(self, identity: int) -> np.ndarray:
        # not really anything to compute for this feature, just return the
        # point mask from the PoseEstimation object for this identity
        return self._poses.get_identity_point_mask(identity)

    def window(self, identity: int, window_size: int,
               per_frame_values: np.ndarray) -> typing.Dict:
        # we do not compute window features for the point mask
        return {}
