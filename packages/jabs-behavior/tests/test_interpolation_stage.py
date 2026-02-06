import numpy as np
import pytest

from jabs.behavior.events import ClassLabels
from jabs.behavior.postprocessing.stages import GapInterpolationStage


class TestInterpolationFilter:
    """Tests for InterpolationFilter."""

    def test_constructor_valid(self):
        """Test constructor with valid parameters."""
        filter_obj = GapInterpolationStage(max_interpolation_gap=5)
        assert filter_obj.config["max_interpolation_gap"] == 5

    def test_constructor_missing_interpolate_frames_max(self):
        """Test that constructor raises error when interpolate_frames_max is missing."""
        with pytest.raises(ValueError, match="max_interpolation_gap must be specified"):
            GapInterpolationStage()

    def test_constructor_invalid_interpolate_frames_max_non_positive(self):
        """Test that constructor raises error when interpolate_frames_max is not positive."""
        with pytest.raises(ValueError, match="max_interpolation_gap must be a positive integer"):
            GapInterpolationStage(max_interpolation_gap=0)

        with pytest.raises(ValueError, match="max_interpolation_gap must be a positive integer"):
            GapInterpolationStage(max_interpolation_gap=-5)

    def test_constructor_invalid_interpolate_frames_max_not_int(self):
        """Test that constructor raises error when interpolate_frames_max is not an integer."""
        with pytest.raises(ValueError, match="max_interpolation_gap must be a positive integer"):
            GapInterpolationStage(max_interpolation_gap=5.5)

        with pytest.raises(ValueError, match="max_interpolation_gap must be a positive integer"):
            GapInterpolationStage(max_interpolation_gap="5")

    def test_apply_interpolates_short_none_gaps(self):
        """Test that apply interpolates short NONE gaps."""
        # Pattern: BEHAVIOR(3), NONE(2), BEHAVIOR(3)
        # With interpolate_frames_max=3, should interpolate NONE gap (2 <= 3)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = GapInterpolationStage(max_interpolation_gap=3)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # NONE gap should be filled with surrounding behavior
        expected = np.array([ClassLabels.BEHAVIOR] * 8)
        np.testing.assert_array_equal(result, expected)

    def test_apply_does_not_interpolate_long_none_gaps(self):
        """Test that apply does not interpolate long NONE gaps."""
        # Pattern: BEHAVIOR(3), NONE(4), BEHAVIOR(3)
        # With interpolate_frames_max=3, should NOT interpolate (4 > 3)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = GapInterpolationStage(max_interpolation_gap=3)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # Should remain unchanged
        np.testing.assert_array_equal(result, classes)

    def test_apply_interpolates_at_boundary(self):
        """Test interpolation when gap equals max_interpolation_gap."""
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        # Gap is 3, filter uses <=, so with interpolate_frames_max=3 should interpolate
        filter_obj = GapInterpolationStage(max_interpolation_gap=3)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)
        expected = np.array([ClassLabels.BEHAVIOR] * 7)
        np.testing.assert_array_equal(result, expected)

        # With max_interpolation_gap=2, gap (3) > 2, so should NOT interpolate
        filter_obj = GapInterpolationStage(max_interpolation_gap=2)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)
        np.testing.assert_array_equal(result, classes)

    def test_apply_multiple_none_gaps(self):
        """Test interpolation with multiple NONE gaps."""
        # Pattern: BEHAVIOR(2), NONE(1), BEHAVIOR(2), NONE(1), BEHAVIOR(2)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = GapInterpolationStage(max_interpolation_gap=2)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # All NONE gaps (size 1, 1 <= 2) should be interpolated
        expected = np.array([ClassLabels.BEHAVIOR] * 8)
        np.testing.assert_array_equal(result, expected)

    def test_apply_none_between_different_states(self):
        """Test interpolation of NONE between different behavior states."""
        # Pattern: BEHAVIOR(2), NONE(1), NOT_BEHAVIOR(2)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )

        filter_obj = GapInterpolationStage(max_interpolation_gap=1)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # NONE (size 1, 1 <= 1) should be split between BEHAVIOR and NOT_BEHAVIOR
        # With odd duration (1), previous gets floor(1/2)=0, next gets ceil(1/2)=1
        expected = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_apply_does_not_affect_not_behavior(self):
        """Test that apply does not interpolate NOT_BEHAVIOR gaps."""
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = GapInterpolationStage(max_interpolation_gap=5)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # Should remain unchanged (only NONE is interpolated)
        np.testing.assert_array_equal(result, classes)

    def test_apply_empty_array(self):
        """Test apply with empty array."""
        classes = np.array([])
        filter_obj = GapInterpolationStage(max_interpolation_gap=5)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        assert len(result) == 0

    def test_apply_all_none(self):
        """Test apply with array of all NONE."""
        classes = np.array([ClassLabels.NONE] * 10)
        filter_obj = GapInterpolationStage(max_interpolation_gap=5)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # Should remain unchanged (no surrounding bouts to interpolate with)
        np.testing.assert_array_equal(result, classes)

    def test_apply_none_at_boundaries(self):
        """Test interpolation of NONE at start/end (merges with neighbors)."""
        # Pattern: NONE(2), BEHAVIOR(3), NONE(2)
        classes = np.array(
            [
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
            ]
        )

        filter_obj = GapInterpolationStage(max_interpolation_gap=5)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # Boundary NONE bouts are interpolated (deleted and merged with neighbors)
        # First NONE bout (index 0) merges with BEHAVIOR (takes BEHAVIOR state)
        # Last NONE bout (index 2) merges with BEHAVIOR (takes BEHAVIOR state)
        # Result: All BEHAVIOR
        expected = np.array([ClassLabels.BEHAVIOR] * 7)
        np.testing.assert_array_equal(result, expected)

    def test_apply_mixed_gap_sizes(self):
        """Test with mixed gap sizes (some short, some long)."""
        # Pattern: BEHAVIOR(2), NONE(1), BEHAVIOR(2), NONE(5), BEHAVIOR(2)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = GapInterpolationStage(max_interpolation_gap=2)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # First gap (size 1, 1 <= 2) should be interpolated, second gap (size 5, 5 > 2) should not
        expected = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_apply_single_frame_none(self):
        """Test interpolation of single frame NONE gaps."""
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = GapInterpolationStage(max_interpolation_gap=1)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = filter_obj.apply(classes, probabilities)

        # All should be behavior (single frame NONE gaps interpolated, 1 <= 1)
        expected = np.array([ClassLabels.BEHAVIOR] * 5)
        np.testing.assert_array_equal(result, expected)

    def test_help_method(self):
        """Test that help method returns valid FilterHelp."""
        filter_obj = GapInterpolationStage(max_interpolation_gap=5)
        help_info = filter_obj.help()

        assert help_info.description is not None
        assert isinstance(help_info.kwargs, dict)

    def test_config_property(self):
        """Test that filter has config property."""
        filter_obj = GapInterpolationStage(max_interpolation_gap=7)
        assert hasattr(filter_obj, "config")
        assert isinstance(filter_obj.config, dict)

    def test_stores_config(self):
        """Test that filter stores its configuration correctly."""
        filter_obj = GapInterpolationStage(max_interpolation_gap=7)
        assert "max_interpolation_gap" in filter_obj.config
        assert filter_obj.config["max_interpolation_gap"] == 7
