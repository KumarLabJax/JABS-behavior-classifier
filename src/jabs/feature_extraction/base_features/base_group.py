import typing

from jabs.feature_extraction.feature_group_base_class import FeatureGroup

from ..feature_base_class import Feature
from .angles import Angles
from .angular_velocity import AngularVelocity
from .centroid_velocity import CentroidVelocityDir, CentroidVelocityMag
from .pairwise_distances import PairwisePointDistances
from .point_speeds import PointSpeeds
from .point_velocities import PointVelocityDirs


class BaseFeatureGroup(FeatureGroup):
    """Base class for feature extraction groups."""

    _name = "base"

    # build a dictionary that maps a feature name to the class that
    # implements it
    _features: typing.ClassVar[dict[str, Feature]] = {
        PairwisePointDistances.name(): PairwisePointDistances,
        Angles.name(): Angles,
        AngularVelocity.name(): AngularVelocity,
        PointSpeeds.name(): PointSpeeds,
        PointVelocityDirs.name(): PointVelocityDirs,
        CentroidVelocityDir.name(): CentroidVelocityDir,
        CentroidVelocityMag.name(): CentroidVelocityMag,
    }

    def _init_feature_mods(self, identity: int):
        """initialize all the feature modules specified in the current config

        Args:
            identity: unused, specified by abstract base class

        Returns:
            dictionary of initialized feature modules for this group
        """
        return {
            feature: self._features[feature](self._poses, self._pixel_scale)
            for feature in self._enabled_features
        }
