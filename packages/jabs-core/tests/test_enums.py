"""Unit tests for jabs.core.enums module."""

import json

import pytest

from jabs.core.enums import (
    DEFAULT_CV_GROUPING_STRATEGY,
    ClassifierType,
    CrossValidationGroupingStrategy,
    ProjectDistanceUnit,
)


class TestProjectDistanceUnit:
    """Tests for ProjectDistanceUnit enum."""

    def test_pixel_value(self):
        """Test that PIXEL has correct integer value."""
        assert ProjectDistanceUnit.PIXEL == 0
        assert ProjectDistanceUnit.PIXEL.value == 0

    def test_cm_value(self):
        """Test that CM has correct integer value."""
        assert ProjectDistanceUnit.CM == 1
        assert ProjectDistanceUnit.CM.value == 1

    def test_is_int_enum(self):
        """Test that ProjectDistanceUnit values are integers."""
        assert isinstance(ProjectDistanceUnit.PIXEL.value, int)
        assert isinstance(ProjectDistanceUnit.CM.value, int)

    def test_comparison_with_int(self):
        """Test that enum values can be compared with integers."""
        assert ProjectDistanceUnit.PIXEL == 0
        assert ProjectDistanceUnit.CM == 1

    def test_all_members(self):
        """Test that all expected members exist."""
        members = list(ProjectDistanceUnit)
        assert len(members) == 2
        assert ProjectDistanceUnit.PIXEL in members
        assert ProjectDistanceUnit.CM in members

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, ProjectDistanceUnit.PIXEL),
            (1, ProjectDistanceUnit.CM),
        ],
    )
    def test_from_value(self, value, expected):
        """Test creating enum from integer value."""
        assert ProjectDistanceUnit(value) == expected

    def test_invalid_value_raises(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            ProjectDistanceUnit(99)


class TestClassifierType:
    """Tests for ClassifierType enum."""

    def test_random_forest_value(self):
        """Test RANDOM_FOREST has correct value."""
        assert ClassifierType.RANDOM_FOREST.value == "Random Forest"

    def test_catboost_value(self):
        """Test CATBOOST has correct value."""
        assert ClassifierType.CATBOOST.value == "CatBoost"

    def test_xgboost_value(self):
        """Test XGBOOST has correct value."""
        assert ClassifierType.XGBOOST.value == "XGBoost"

    def test_is_str_enum(self):
        """Test that ClassifierType inherits from str."""
        assert isinstance(ClassifierType.RANDOM_FOREST, str)
        assert isinstance(ClassifierType.CATBOOST, str)
        assert isinstance(ClassifierType.XGBOOST, str)

    def test_string_comparison(self):
        """Test that enum values can be compared with strings."""
        assert ClassifierType.RANDOM_FOREST == "Random Forest"
        assert ClassifierType.CATBOOST == "CatBoost"
        assert ClassifierType.XGBOOST == "XGBoost"

    def test_all_members(self):
        """Test that all expected members exist."""
        members = list(ClassifierType)
        assert len(members) == 3
        assert ClassifierType.RANDOM_FOREST in members
        assert ClassifierType.CATBOOST in members
        assert ClassifierType.XGBOOST in members

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("Random Forest", ClassifierType.RANDOM_FOREST),
            ("CatBoost", ClassifierType.CATBOOST),
            ("XGBoost", ClassifierType.XGBOOST),
        ],
    )
    def test_from_value(self, value, expected):
        """Test creating enum from string value."""
        assert ClassifierType(value) == expected

    def test_invalid_value_raises(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            ClassifierType("InvalidClassifier")

    def test_json_serialization(self):
        """Test that ClassifierType can be serialized to JSON."""
        # str enum should serialize automatically
        data = {"classifier": ClassifierType.RANDOM_FOREST}
        json_str = json.dumps(data)
        assert "Random Forest" in json_str


class TestCrossValidationGroupingStrategy:
    """Tests for CrossValidationGroupingStrategy enum."""

    def test_individual_value(self):
        """Test INDIVIDUAL has correct value."""
        assert CrossValidationGroupingStrategy.INDIVIDUAL.value == "Individual Animal"

    def test_video_value(self):
        """Test VIDEO has correct value."""
        assert CrossValidationGroupingStrategy.VIDEO.value == "Video"

    def test_is_str_enum(self):
        """Test that CrossValidationGroupingStrategy inherits from str."""
        assert isinstance(CrossValidationGroupingStrategy.INDIVIDUAL, str)
        assert isinstance(CrossValidationGroupingStrategy.VIDEO, str)

    def test_string_comparison(self):
        """Test that enum values can be compared with strings."""
        assert CrossValidationGroupingStrategy.INDIVIDUAL == "Individual Animal"
        assert CrossValidationGroupingStrategy.VIDEO == "Video"

    def test_all_members(self):
        """Test that all expected members exist."""
        members = list(CrossValidationGroupingStrategy)
        assert len(members) == 2
        assert CrossValidationGroupingStrategy.INDIVIDUAL in members
        assert CrossValidationGroupingStrategy.VIDEO in members

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("Individual Animal", CrossValidationGroupingStrategy.INDIVIDUAL),
            ("Video", CrossValidationGroupingStrategy.VIDEO),
        ],
    )
    def test_from_value(self, value, expected):
        """Test creating enum from string value."""
        assert CrossValidationGroupingStrategy(value) == expected

    def test_invalid_value_raises(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            CrossValidationGroupingStrategy("InvalidStrategy")

    def test_json_serialization(self):
        """Test that CrossValidationGroupingStrategy can be serialized to JSON."""
        data = {"strategy": CrossValidationGroupingStrategy.INDIVIDUAL}
        json_str = json.dumps(data)
        assert "Individual Animal" in json_str

    def test_json_deserialization(self):
        """Test that CrossValidationGroupingStrategy can be deserialized from JSON."""
        json_str = '{"strategy": "Individual Animal"}'
        data = json.loads(json_str)
        strategy = CrossValidationGroupingStrategy(data["strategy"])
        assert strategy == CrossValidationGroupingStrategy.INDIVIDUAL


class TestDefaultCVGroupingStrategy:
    """Tests for DEFAULT_CV_GROUPING_STRATEGY constant."""

    def test_default_is_individual(self):
        """Test that default strategy is INDIVIDUAL."""
        assert DEFAULT_CV_GROUPING_STRATEGY == CrossValidationGroupingStrategy.INDIVIDUAL

    def test_default_is_valid_strategy(self):
        """Test that default is a valid CrossValidationGroupingStrategy."""
        assert DEFAULT_CV_GROUPING_STRATEGY in CrossValidationGroupingStrategy

    def test_default_value_string(self):
        """Test the string value of default strategy."""
        assert DEFAULT_CV_GROUPING_STRATEGY.value == "Individual Animal"


class TestEnumImports:
    """Tests for enum module imports and exports."""

    def test_all_enums_importable_from_enums_module(self):
        """Test that all enums can be imported from jabs.core.enums."""
        from jabs.core.enums import (
            DEFAULT_CV_GROUPING_STRATEGY,
            ClassifierType,
            CrossValidationGroupingStrategy,
            ProjectDistanceUnit,
        )

        assert ClassifierType is not None
        assert CrossValidationGroupingStrategy is not None
        assert ProjectDistanceUnit is not None
        assert DEFAULT_CV_GROUPING_STRATEGY is not None


def test_jabs_pose_version_is_int_aligned():
    """JabsPoseVersion members are aligned to the legacy integer majors."""
    from jabs.core.enums import JabsPoseVersion

    assert JabsPoseVersion.V2 == 2
    assert JabsPoseVersion.V3 == 3
    assert int(JabsPoseVersion.V2) == 2


def test_jabs_pose_version_ordered():
    """IntEnum members compare numerically, matching the legacy integer majors."""
    from jabs.core.enums import JabsPoseVersion

    assert JabsPoseVersion.V2 < JabsPoseVersion.V3
