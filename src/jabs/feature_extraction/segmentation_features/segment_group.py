import typing

from jabs.feature_extraction.feature_group_base_class import FeatureGroup
from jabs.pose_estimation import PoseEstimation

from ..feature_base_class import Feature

# import all feature modules for this group
from .hu_moments import HuMoments
from .moment_cache import MomentInfo
from .moments import Moments
from .shape_descriptors import ShapeDescriptors


class SegmentationFeatureGroup(FeatureGroup):
    """A feature group for extracting segmentation features from pose estimation data."""

    _name = "segmentation"

    # build dictionary mapping feature name to class that implements it
    _features: typing.ClassVar[dict[str, Feature]] = {
        Moments.name(): Moments,
        ShapeDescriptors.name(): ShapeDescriptors,
        HuMoments.name(): HuMoments,
    }

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)
        self._moments_cache = None

    def _init_feature_mods(self, identity: int):
        """initialize all of the feature modules specified in the current config

        Args:
            identity: subject identity to use when computing segmentation features

        Returns:
            dictionary of initialized feature modules for this group
        """
        self._moments_cache = MomentInfo(self._poses, identity, self._pixel_scale)

        return {
            feature: self._features[feature](self._poses, self._pixel_scale, self._moments_cache)
            for feature in self._enabled_features
        }
