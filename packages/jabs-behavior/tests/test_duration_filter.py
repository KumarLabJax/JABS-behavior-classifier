import numpy as np
import pytest

from jabs.behavior.events import ClassLabels
from jabs.behavior.postprocessing.stages import BoutDurationFilterStage


class TestDurationFilter:
    """Tests for DurationFilter."""

    def test_constructor_valid(self):
        """Test constructor with valid parameters."""
        filter_obj = BoutDurationFilterStage(min_duration=5)
        assert filter_obj.config["min_duration"] == 5

    def test_constructor_missing_min_duration(self):
        """Test that constructor raises error when min_duration is missing."""
        with pytest.raises(ValueError, match="min_duration must be specified"):
            BoutDurationFilterStage()

    def test_constructor_invalid_min_duration_non_positive(self):
        """Test that constructor raises error when min_duration is not positive."""
        with pytest.raises(ValueError, match="min_duration must be a positive integer"):
            BoutDurationFilterStage(min_duration=0)

        with pytest.raises(ValueError, match="min_duration must be a positive integer"):
            BoutDurationFilterStage(min_duration=-5)

    def test_constructor_invalid_min_duration_not_int(self):
        """Test that constructor raises error when min_duration is not an integer."""
        with pytest.raises(ValueError, match="min_duration must be a positive integer"):
            BoutDurationFilterStage(min_duration=5.5)

        with pytest.raises(ValueError, match="min_duration must be a positive integer"):
            BoutDurationFilterStage(min_duration="5")

    def test_apply_removes_short_behavior_bouts(self):
        """Test that apply removes behavior bouts shorter than min_duration."""
        # Pattern: NOT_BEHAVIOR(3), BEHAVIOR(2), NOT_BEHAVIOR(3), BEHAVIOR(5), NOT_BEHAVIOR(2)
        # With min_duration=3, should remove the first BEHAVIOR bout (duration 2)
        classes = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )
        probabilities = np.full_like(classes, 0.5, dtype=float)

        filter_obj = BoutDurationFilterStage(min_duration=3)
        result = filter_obj.apply(classes, probabilities)

        expected = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_apply_keeps_long_behavior_bouts(self):
        """Test that apply keeps behavior bouts longer than or equal to min_duration."""
        classes = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )
        probabilities = np.full_like(classes, 0.5, dtype=float)

        filter_obj = BoutDurationFilterStage(min_duration=3)
        result = filter_obj.apply(classes, probabilities)

        # Should remain unchanged
        np.testing.assert_array_equal(result, classes)

    def test_apply_does_not_affect_not_behavior(self):
        """Test that apply only targets BEHAVIOR bouts, not NOT_BEHAVIOR bouts."""
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,  # Short NOT_BEHAVIOR bout
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )
        probabilities = np.full_like(classes, 0.5, dtype=float)

        filter_obj = BoutDurationFilterStage(min_duration=5)
        result = filter_obj.apply(classes, probabilities)

        # Both BEHAVIOR bouts (3 frames each) are removed because they're < 5
        # When boundary bouts are deleted, they merge with neighbors
        # First BEHAVIOR bout (index 0) merges with NOT_BEHAVIOR (takes NOT_BEHAVIOR state)
        # Last BEHAVIOR bout (index 2) merges with NOT_BEHAVIOR (takes NOT_BEHAVIOR state)
        # Result: All NOT_BEHAVIOR
        expected = np.array([ClassLabels.NOT_BEHAVIOR] * 7)
        np.testing.assert_array_equal(result, expected)

    def test_apply_handles_none_label(self):
        """Test that apply handles NONE labels correctly."""
        classes = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )
        probabilities = np.full_like(classes, 0.5, dtype=float)

        filter_obj = BoutDurationFilterStage(min_duration=5)
        result = filter_obj.apply(classes, probabilities)

        # Short BEHAVIOR bout (duration 2) should be removed
        # When deleting the BEHAVIOR bout, it gets split between NONE and NOT_BEHAVIOR
        expected = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NONE,
                ClassLabels.NONE,
                ClassLabels.NONE,  # floor(2/2) = 1 frame added to NONE
                ClassLabels.NOT_BEHAVIOR,  # ceil(2/2) = 1 frame added to NOT_BEHAVIOR
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_apply_empty_array(self):
        """Test apply with empty array."""
        classes = np.array([])
        probabilities = np.array([])
        filter_obj = BoutDurationFilterStage(min_duration=5)
        result = filter_obj.apply(classes, probabilities)

        assert len(result) == 0

    def test_apply_all_same_state(self):
        """Test apply with array of all same state."""
        classes = np.array([ClassLabels.BEHAVIOR] * 10)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        filter_obj = BoutDurationFilterStage(min_duration=5)
        result = filter_obj.apply(classes, probabilities)

        # Should remain unchanged (single long bout)
        np.testing.assert_array_equal(result, classes)

    def test_apply_multiple_short_bouts(self):
        """Test apply removes multiple short bouts."""
        classes = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,  # Short (2)
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,  # Short (1)
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,  # Short (3)
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )
        probabilities = np.full_like(classes, 0.5, dtype=float)

        filter_obj = BoutDurationFilterStage(min_duration=4)
        result = filter_obj.apply(classes, probabilities)

        # All behavior bouts should be removed
        expected = np.array([ClassLabels.NOT_BEHAVIOR] * len(classes))
        np.testing.assert_array_equal(result, expected)

    def test_apply_uniform_short_behavior_vector(self):
        """Test apply on a vector that is a single short BEHAVIOR bout.

        There is no neighboring bout to absorb the removed frames, so the frames become
        NOT_BEHAVIOR directly: the same outcome a neighboring NOT_BEHAVIOR bout would have
        produced, rather than keeping a bout the filter rejected.
        """
        filter_obj = BoutDurationFilterStage(min_duration=5)
        classes = np.full(3, ClassLabels.BEHAVIOR, dtype=np.int8)

        result = filter_obj.apply(classes, np.zeros(3, dtype=np.float32))

        np.testing.assert_array_equal(result, np.full(3, ClassLabels.NOT_BEHAVIOR))
        assert result.dtype == classes.dtype

    def test_apply_short_behavior_bout_matches_with_and_without_a_neighbor(self):
        """A short BEHAVIOR bout is suppressed whether or not other frames surround it.

        Guards the inconsistency of the same bout being kept when it is alone in the vector
        and dropped when a NOT_BEHAVIOR bout happens to follow it.
        """
        filter_obj = BoutDurationFilterStage(min_duration=5)
        alone = np.full(3, ClassLabels.BEHAVIOR, dtype=np.int8)
        with_neighbor = np.array(
            [ClassLabels.BEHAVIOR] * 3 + [ClassLabels.NOT_BEHAVIOR] * 3, dtype=np.int8
        )

        result_alone = filter_obj.apply(alone, np.zeros(3, dtype=np.float32))
        result_with_neighbor = filter_obj.apply(with_neighbor, np.zeros(6, dtype=np.float32))

        assert set(np.unique(result_alone)) == {ClassLabels.NOT_BEHAVIOR}
        assert set(np.unique(result_with_neighbor)) == {ClassLabels.NOT_BEHAVIOR}

    def test_apply_uniform_behavior_vector_at_min_duration_is_kept(self):
        """A lone bout that meets the minimum duration is not filtered."""
        filter_obj = BoutDurationFilterStage(min_duration=5)
        classes = np.full(5, ClassLabels.BEHAVIOR, dtype=np.int8)

        result = filter_obj.apply(classes, np.zeros(5, dtype=np.float32))

        np.testing.assert_array_equal(result, classes)

    def test_apply_uniform_short_none_vector_is_untouched(self):
        """A lone short bout of NONE is not a BEHAVIOR bout, so the filter ignores it."""
        filter_obj = BoutDurationFilterStage(min_duration=5)
        classes = np.full(3, ClassLabels.NONE, dtype=np.int8)

        result = filter_obj.apply(classes, np.zeros(3, dtype=np.float32))

        np.testing.assert_array_equal(result, classes)

    def test_help_method(self):
        """Test that help method returns valid FilterHelp."""
        filter_obj = BoutDurationFilterStage(min_duration=5)
        help_info = filter_obj.help()

        assert help_info.description is not None
        assert "min_duration" in help_info.kwargs
        assert help_info.kwargs["min_duration"].type == "int"

    def test_config_property(self):
        """Test that filter has config property."""
        filter_obj = BoutDurationFilterStage(min_duration=10)
        assert hasattr(filter_obj, "config")
        assert isinstance(filter_obj.config, dict)

    def test_stores_config(self):
        """Test that filter stores its configuration correctly."""
        filter_obj = BoutDurationFilterStage(min_duration=10)
        assert "min_duration" in filter_obj.config
        assert filter_obj.config["min_duration"] == 10
