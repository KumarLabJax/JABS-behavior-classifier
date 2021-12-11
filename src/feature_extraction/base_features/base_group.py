
from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_group_base_class import FeatureGroup

from . import Angles, PairwisePointDistances, PointSpeeds, \
    CentroidVelocityMag, CentroidVelocityDir, NoseVelocityDir, \
    NoseVelocityMag, BaseTailVelocityDir, BaseTailVelocityMag, \
    LeftFrontPawVelocityDir, LeftFrontPawVelocityMag, \
    RightFrontPawVelocityDir, RightFrontPawVelocityMag, \
    AngularVelocity


class BaseFeatureGroup(FeatureGroup):

    _name = 'base'

    # build a dictionary that maps a feature name to the class that
    # implements it
    _features = {
        PairwisePointDistances.name(): PairwisePointDistances,
        Angles.name(): Angles,
        AngularVelocity.name(): AngularVelocity,
        PointSpeeds.name(): PointSpeeds,
        CentroidVelocityDir.name(): CentroidVelocityDir,
        CentroidVelocityMag.name(): CentroidVelocityMag,
        NoseVelocityDir.name(): NoseVelocityDir,
        NoseVelocityMag.name(): NoseVelocityMag,
        BaseTailVelocityDir.name(): BaseTailVelocityDir,
        BaseTailVelocityMag.name(): BaseTailVelocityMag,
        LeftFrontPawVelocityDir.name(): LeftFrontPawVelocityDir,
        LeftFrontPawVelocityMag.name(): LeftFrontPawVelocityMag,
        RightFrontPawVelocityDir.name(): RightFrontPawVelocityDir,
        RightFrontPawVelocityMag.name(): RightFrontPawVelocityMag,
    }

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
