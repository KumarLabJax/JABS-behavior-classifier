"""Unit tests for the BaseFeatureGroup class."""

import pytest

from jabs.feature_extraction.base_features import (
    Angles,
    AngularVelocity,
    BaseFeatureGroup,
    CentroidVelocityDir,
    CentroidVelocityMag,
    PairwisePointDistances,
    PointSpeeds,
    PointVelocityDirs,
)
from jabs.pose_estimation import PoseEstimation


def test_base_feature_group_name():
    """Test that the group name is set correctly."""
    assert BaseFeatureGroup._name == "base"


def test_base_feature_group_features_dict():
    """Test that the _features dictionary contains all expected features."""
    expected_features = {
        "pairwise_distances": PairwisePointDistances,
        "angles": Angles,
        "angular_velocity": AngularVelocity,
        "point_speeds": PointSpeeds,
        "point_velocity_dirs": PointVelocityDirs,
        "centroid_velocity_dir": CentroidVelocityDir,
        "centroid_velocity_mag": CentroidVelocityMag,
    }

    assert BaseFeatureGroup._features == expected_features


@pytest.mark.parametrize(
    "enabled_features",
    [list(BaseFeatureGroup._features), ["angles", "point_speeds"], []],
    ids=["all-features", "subset", "none"],
)
def test_base_feature_group_init_feature_mods(
    pose_est_v5: PoseEstimation, enabled_features: list[str]
) -> None:
    """Test that _init_feature_mods initializes exactly the enabled feature modules.

    Args:
        pose_est_v5: Pose estimation fixture.
        enabled_features: Names of the features enabled on the group.
    """
    feature_group = BaseFeatureGroup(pose_est_v5, pose_est_v5.cm_per_pixel)
    feature_group._enabled_features = enabled_features

    # Initialize feature modules for identity 0
    feature_mods = feature_group._init_feature_mods(identity=0)

    # Should have one module per enabled feature, and nothing else
    assert len(feature_mods) == len(enabled_features)
    assert set(feature_mods.keys()) == set(enabled_features)
    if not enabled_features:
        assert feature_mods == {}

    # Check that each feature module is an instance of the correct class
    for feature_name in enabled_features:
        assert feature_name in feature_mods
        assert isinstance(feature_mods[feature_name], BaseFeatureGroup._features[feature_name])


def test_base_feature_group_all_features_have_correct_interface(pose_est_v5):
    """Test that all feature classes implement the required interface."""
    pixel_scale = pose_est_v5.cm_per_pixel

    # Test each feature class
    for feature_name, feature_class in BaseFeatureGroup._features.items():
        # Should be able to instantiate with pose and pixel_scale
        feature_instance = feature_class(pose_est_v5, pixel_scale)

        # Should have a name() class method that returns the expected name
        assert feature_class.name() == feature_name

        # Should have a per_frame method
        assert hasattr(feature_instance, "per_frame")
        assert callable(feature_instance.per_frame)

        # per_frame should return a dict
        result = feature_instance.per_frame(identity=0)
        assert isinstance(result, dict)
