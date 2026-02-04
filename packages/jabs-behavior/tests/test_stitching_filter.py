import numpy as np
import pytest

from jabs.behavior.events import ClassLabels
from jabs.behavior.postprocessing.stages import BoutStitchingStage


class TestStitchingStage:
    """Tests for BoutStitchingStage."""

    def test_constructor_valid(self):
        """Test constructor with valid parameters."""
        filter_obj = BoutStitchingStage(max_stitch_gap=3)
        assert filter_obj.config["max_stitch_gap"] == 3

    def test_constructor_missing_max_stitch_gap(self):
        """Test that constructor raises error when max_stitch_gap is missing."""
        with pytest.raises(ValueError, match="max_stitch_gap must be specified"):
            BoutStitchingStage()

    def test_constructor_invalid_max_stitch_gap_non_positive(self):
        """Test that constructor raises error when max_stitch_gap is not positive."""
        with pytest.raises(ValueError, match="max_stitch_gap must be a positive integer"):
            BoutStitchingStage(max_stitch_gap=0)

        with pytest.raises(ValueError, match="max_stitch_gap must be a positive integer"):
            BoutStitchingStage(max_stitch_gap=-3)

    def test_constructor_invalid_max_stitch_gap_not_int(self):
        """Test that constructor raises error when max_stitch_gap is not an integer."""
        with pytest.raises(ValueError, match="max_stitch_gap must be a positive integer"):
            BoutStitchingStage(max_stitch_gap=3.5)

        with pytest.raises(ValueError, match="max_stitch_gap must be a positive integer"):
            BoutStitchingStage(max_stitch_gap="3")

    def test_apply_stitches_short_gaps(self):
        """Test that apply stitches behavior bouts separated by short NOT_BEHAVIOR gaps."""
        # Pattern: BEHAVIOR(3), NOT_BEHAVIOR(2), BEHAVIOR(3)
        # With max_stitch_gap=2, should stitch into single BEHAVIOR bout (gap 2 <= 2)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = BoutStitchingStage(max_stitch_gap=2)
        result = filter_obj.apply(classes)

        expected = np.array([ClassLabels.BEHAVIOR] * 8)
        np.testing.assert_array_equal(result, expected)

    def test_apply_does_not_stitch_long_gaps(self):
        """Test that apply does not stitch bouts separated by long NOT_BEHAVIOR gaps."""
        # Pattern: BEHAVIOR(3), NOT_BEHAVIOR(4), BEHAVIOR(3)
        # With max_stitch_gap=3, should NOT stitch (gap 4 > 3)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = BoutStitchingStage(max_stitch_gap=3)
        result = filter_obj.apply(classes)

        # Should remain unchanged
        np.testing.assert_array_equal(result, classes)

    def test_apply_stitches_at_boundary(self):
        """Test stitching when gap equals max_stitch_gap."""
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        # Gap is 3, filter uses <=, so with max_stitch_gap=3 should stitch
        filter_obj = BoutStitchingStage(max_stitch_gap=3)
        result = filter_obj.apply(classes)
        expected = np.array([ClassLabels.BEHAVIOR] * 7)
        np.testing.assert_array_equal(result, expected)

        # With max_stitch_gap=2, gap (3) > 2, so should NOT stitch
        filter_obj = BoutStitchingStage(max_stitch_gap=2)
        result = filter_obj.apply(classes)
        np.testing.assert_array_equal(result, classes)

    def test_apply_multiple_gaps(self):
        """Test stitching with multiple gaps."""
        # Pattern: BEHAVIOR(2), NOT_BEHAVIOR(1), BEHAVIOR(2), NOT_BEHAVIOR(1), BEHAVIOR(2)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = BoutStitchingStage(max_stitch_gap=1)
        result = filter_obj.apply(classes)

        # All gaps (size 1, 1 <= 1) should be stitched
        expected = np.array([ClassLabels.BEHAVIOR] * 8)
        np.testing.assert_array_equal(result, expected)

    def test_apply_does_not_affect_behavior_bouts(self):
        """Test that apply does not modify behavior bouts themselves."""
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,  # Long behavior bout
            ]
        )

        filter_obj = BoutStitchingStage(max_stitch_gap=2)
        result = filter_obj.apply(classes)

        # Should remain unchanged
        np.testing.assert_array_equal(result, classes)

    def test_apply_handles_none_label(self):
        """Test that apply does not stitch across NONE labels."""
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = BoutStitchingStage(max_stitch_gap=3)
        result = filter_obj.apply(classes)

        # NONE labels should not be treated as NOT_BEHAVIOR for stitching
        np.testing.assert_array_equal(result, classes)

    def test_apply_empty_array(self):
        """Test apply with empty array."""
        classes = np.array([])
        filter_obj = BoutStitchingStage(max_stitch_gap=2)
        result = filter_obj.apply(classes)

        assert len(result) == 0

    def test_apply_all_same_state(self):
        """Test apply with array of all same state."""
        classes = np.array([ClassLabels.BEHAVIOR] * 10)
        filter_obj = BoutStitchingStage(max_stitch_gap=2)
        result = filter_obj.apply(classes)

        # Should remain unchanged
        np.testing.assert_array_equal(result, classes)

    def test_apply_alternating_pattern(self):
        """Test apply with alternating pattern."""
        # Pattern: BEHAVIOR(1), NOT_BEHAVIOR(1), BEHAVIOR(1), NOT_BEHAVIOR(1), BEHAVIOR(1)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = BoutStitchingStage(max_stitch_gap=2)
        result = filter_obj.apply(classes)

        # All should be stitched together
        expected = np.array([ClassLabels.BEHAVIOR] * 5)
        np.testing.assert_array_equal(result, expected)

    def test_apply_mixed_gap_sizes(self):
        """Test with mixed gap sizes (some short, some long)."""
        # Pattern: BEHAVIOR(2), NOT_BEHAVIOR(1), BEHAVIOR(2), NOT_BEHAVIOR(5), BEHAVIOR(2)
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )

        filter_obj = BoutStitchingStage(max_stitch_gap=2)
        result = filter_obj.apply(classes)

        # First gap (size 1, 1 <= 2) should be stitched, second gap (size 5, 5 > 2) should not
        expected = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_help_method(self):
        """Test that help method returns valid FilterHelp."""
        filter_obj = BoutStitchingStage(max_stitch_gap=3)
        help_info = filter_obj.help()

        assert help_info.description is not None
        assert "max_stitch_gap" in help_info.kwargs
        assert help_info.kwargs["max_stitch_gap"].type == "int"

    def test_name_attribute(self):
        """Test that filter has correct name attribute."""
        assert BoutStitchingStage.name == "bout_stitching_stage"

    def test_config_property(self):
        """Test that filter has config property."""
        filter_obj = BoutStitchingStage(max_stitch_gap=5)
        assert hasattr(filter_obj, "config")
        assert isinstance(filter_obj.config, dict)

    def test_stores_config(self):
        """Test that filter stores its configuration correctly."""
        filter_obj = BoutStitchingStage(max_stitch_gap=5)
        assert "max_stitch_gap" in filter_obj.config
        assert filter_obj.config["max_stitch_gap"] == 5
