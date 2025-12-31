"""Unit tests for the BaseFeatureGroup class."""

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


def test_base_feature_group_instantiation(pose_est_v5):
    """Test that BaseFeatureGroup can be instantiated."""
    pixel_scale = pose_est_v5.cm_per_pixel
    feature_group = BaseFeatureGroup(pose_est_v5, pixel_scale)

    assert feature_group is not None


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


def test_base_feature_group_init_feature_mods(pose_est_v5):
    """Test that _init_feature_mods initializes feature modules correctly."""
    pixel_scale = pose_est_v5.cm_per_pixel

    # Enable all features
    feature_group = BaseFeatureGroup(pose_est_v5, pixel_scale)
    feature_group._enabled_features = list(BaseFeatureGroup._features.keys())

    # Initialize feature modules for identity 0
    feature_mods = feature_group._init_feature_mods(identity=0)

    # Should have one module per enabled feature
    assert len(feature_mods) == len(BaseFeatureGroup._features)

    # Check that each feature module is an instance of the correct class
    for feature_name, feature_class in BaseFeatureGroup._features.items():
        assert feature_name in feature_mods
        assert isinstance(feature_mods[feature_name], feature_class)


def test_base_feature_group_init_feature_mods_subset(pose_est_v5):
    """Test that _init_feature_mods only initializes enabled features."""
    pixel_scale = pose_est_v5.cm_per_pixel

    # Enable only a subset of features
    enabled_features = ["angles", "point_speeds"]
    feature_group = BaseFeatureGroup(pose_est_v5, pixel_scale)
    feature_group._enabled_features = enabled_features

    # Initialize feature modules for identity 0
    feature_mods = feature_group._init_feature_mods(identity=0)

    # Should only have modules for enabled features
    assert len(feature_mods) == len(enabled_features)
    assert set(feature_mods.keys()) == set(enabled_features)

    # Check that each feature module is an instance of the correct class
    for feature_name in enabled_features:
        assert feature_name in feature_mods
        assert isinstance(feature_mods[feature_name], BaseFeatureGroup._features[feature_name])


def test_base_feature_group_init_feature_mods_empty(pose_est_v5):
    """Test that _init_feature_mods handles empty enabled features list."""
    pixel_scale = pose_est_v5.cm_per_pixel

    # Enable no features
    feature_group = BaseFeatureGroup(pose_est_v5, pixel_scale)
    feature_group._enabled_features = []

    # Initialize feature modules for identity 0
    feature_mods = feature_group._init_feature_mods(identity=0)

    # Should have no modules
    assert len(feature_mods) == 0
    assert feature_mods == {}


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
