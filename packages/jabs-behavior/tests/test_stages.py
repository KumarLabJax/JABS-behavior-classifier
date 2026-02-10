from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from jabs.behavior.postprocessing.stages import (
    BoutDurationFilterStage,
    BoutStitchingStage,
    GapInterpolationStage,
    PostprocessingStage,
    stage_registry,
)
from jabs.behavior.postprocessing.stages.postprocessing_stage import KwargHelp, StageHelp

"""Tests for the PostprocessingStage infrastructure (registry, base classes, help system).

Individual stage implementations are tested in their own dedicated test files.
"""


class TestStageRegistry:
    """Tests for filter_registry function."""

    def test_registry_returns_dict(self):
        """Test that filter_registry returns a dictionary."""
        registry = stage_registry()
        assert isinstance(registry, dict)

    def test_registry_contains_duration_stage(self):
        """Test that registry contains duration stage."""
        registry = stage_registry()
        assert "BoutDurationFilterStage" in registry
        assert registry["BoutDurationFilterStage"] == BoutDurationFilterStage

    def test_registry_contains_stitching_stage(self):
        """Test that registry contains stitching stage."""
        registry = stage_registry()
        assert "BoutStitchingStage" in registry
        assert registry["BoutStitchingStage"] == BoutStitchingStage

    def test_registry_contains_interpolation_stage(self):
        """Test that registry contains interpolation stage."""
        registry = stage_registry()
        assert "GapInterpolationStage" in registry
        assert registry["GapInterpolationStage"] == GapInterpolationStage

    def test_registry_keys_match_filter_names(self):
        """Test that all registry keys match stage name attributes."""
        registry = stage_registry()

        for key, filter_class in registry.items():
            assert key == filter_class.__name__


class TestPostprocessingStage:
    """Tests for PostprocessingStage abstract class."""

    def test_base_stage_cannot_be_instantiated(self):
        """Test that PostprocessingStage cannot be instantiated directly due to abstract methods."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PostprocessingStage()


class TestConcreteStageImplementations:
    """Tests to ensure concrete stages properly implement PostprocessingStage."""

    def test_duration_filter_implements_apply(self):
        """Test that BoutDurationFilterStage implements apply method."""
        stage_obj = BoutDurationFilterStage(min_duration=5)
        classes = np.array([0, 1, 0])
        probabilities = np.full_like(classes, 0.5, dtype=float)

        # Should not raise NotImplementedError
        result = stage_obj.apply(classes, probabilities)
        assert isinstance(result, np.ndarray)

    def test_duration_filter_implements_help(self):
        """Test that BoutDurationFilterStage implements help method."""
        stage_obj = BoutDurationFilterStage(min_duration=5)

        # Should not raise NotImplementedError
        help_info = stage_obj.help()
        assert isinstance(help_info, StageHelp)

    def test_stitching_stage_implements_apply(self):
        """Test that BoutStitchingStage implements apply method."""
        stage_obj = BoutStitchingStage(max_stitch_gap=3)
        classes = np.array([0, 1, 0])
        probabilities = np.full_like(classes, 0.5, dtype=float)

        # Should not raise NotImplementedError
        result = stage_obj.apply(classes, probabilities)
        assert isinstance(result, np.ndarray)

    def test_stitching_stage_implements_help(self):
        """Test that BoutStitchingStage implements help method."""
        stage_obj = BoutStitchingStage(max_stitch_gap=3)

        # Should not raise NotImplementedError
        help_info = stage_obj.help()
        assert isinstance(help_info, StageHelp)


class TestStageHelp:
    """Tests for StageHelp dataclass."""

    def test_stage_help_creation(self):
        """Test that StageHelp can be created."""
        help_info = StageHelp(
            description="Test stage",
            kwargs={"param1": KwargHelp(description="Test param", type="int")},
        )

        assert help_info.description == "Test stage"
        assert "param1" in help_info.kwargs
        assert help_info.kwargs["param1"].description == "Test param"
        assert help_info.kwargs["param1"].type == "int"

    def test_stage_help_is_frozen(self):
        """Test that StageHelp is immutable (frozen)."""
        help_info = StageHelp(description="Test", kwargs={})

        with pytest.raises(FrozenInstanceError):
            help_info.description = "Changed"

    def test_kwarg_help_creation(self):
        """Test that KwargHelp can be created."""
        kwarg = KwargHelp(description="Test parameter", type="str")

        assert kwarg.description == "Test parameter"
        assert kwarg.type == "str"

    def test_kwarg_help_is_frozen(self):
        """Test that KwargHelp is immutable (frozen)."""
        kwarg = KwargHelp(description="Test", type="int")

        with pytest.raises(FrozenInstanceError):
            kwarg.description = "Changed"
