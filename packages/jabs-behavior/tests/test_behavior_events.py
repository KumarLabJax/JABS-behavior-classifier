import numpy as np
import pytest

from jabs.behavior.events import BehaviorEvents, ClassLabels


class TestClassLabels:
    """Tests for ClassLabels enum."""

    def test_class_labels_values(self):
        """Test that ClassLabels have expected values."""
        assert ClassLabels.NONE.value == -1
        assert ClassLabels.NOT_BEHAVIOR.value == 0
        assert ClassLabels.BEHAVIOR.value == 1


class TestBehaviorEvents:
    """Tests for BehaviorEvents class."""

    def test_constructor_valid_data(self):
        """Test constructor with valid RLE data."""
        starts = np.array([0, 5, 10])
        durations = np.array([5, 5, 5])
        states = np.array([0, 1, 0])

        events = BehaviorEvents(starts, durations, states)

        np.testing.assert_array_equal(events.starts, starts)
        np.testing.assert_array_equal(events.durations, durations)
        np.testing.assert_array_equal(events.states, states)

    def test_constructor_invalid_length_mismatch(self):
        """Test that constructor raises error when array lengths don't match."""
        starts = np.array([0, 5])
        durations = np.array([5, 5, 5])
        states = np.array([0, 1])

        with pytest.raises(ValueError, match="All inputs must be of same length"):
            BehaviorEvents(starts, durations, states)

    def test_constructor_invalid_starts_not_sorted(self):
        """Test that constructor raises error when starts are not sorted."""
        starts = np.array([0, 10, 5])
        durations = np.array([5, 5, 5])
        states = np.array([0, 1, 0])

        with pytest.raises(ValueError, match="Starts of events are not ascending"):
            BehaviorEvents(starts, durations, states)

    def test_constructor_invalid_starts_durations_alignment(self):
        """Test that constructor raises error when starts+durations don't align."""
        starts = np.array([0, 5, 12])
        durations = np.array([5, 5, 5])
        states = np.array([0, 1, 0])

        with pytest.raises(
            ValueError, match="Starts \\+ duration must yield the next start value"
        ):
            BehaviorEvents(starts, durations, states)

    def test_constructor_invalid_repeated_states(self):
        """Test that constructor raises error when states repeat."""
        starts = np.array([0, 5, 10])
        durations = np.array([5, 5, 5])
        states = np.array([0, 0, 1])

        with pytest.raises(ValueError, match="States cannot repeat"):
            BehaviorEvents(starts, durations, states)

    def test_from_vector_simple(self):
        """Test from_vector with simple input."""
        vector = np.array([0, 0, 1, 1, 0])
        events = BehaviorEvents.from_vector(vector)

        np.testing.assert_array_equal(events.starts, [0, 2, 4])
        np.testing.assert_array_equal(events.durations, [2, 2, 1])
        np.testing.assert_array_equal(events.states, [0, 1, 0])

    def test_from_vector_single_state(self):
        """Test from_vector with vector of single state."""
        vector = np.array([1, 1, 1, 1])
        events = BehaviorEvents.from_vector(vector)

        np.testing.assert_array_equal(events.starts, [0])
        np.testing.assert_array_equal(events.durations, [4])
        np.testing.assert_array_equal(events.states, [1])

    def test_from_vector_empty(self):
        """Test from_vector with empty input."""
        vector = np.array([])
        events = BehaviorEvents.from_vector(vector)

        assert len(events.starts) == 0
        assert len(events.durations) == 0
        assert len(events.states) == 0

    def test_to_vector_simple(self):
        """Test to_vector with simple RLE data."""
        starts = np.array([0, 3, 5])
        durations = np.array([3, 2, 3])
        states = np.array([0, 1, 0])

        events = BehaviorEvents(starts, durations, states)
        vector = events.to_vector()

        expected = np.array([0, 0, 0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(vector, expected)

    def test_to_vector_empty(self):
        """Test to_vector with empty RLE data."""
        starts = np.array([])
        durations = np.array([])
        states = np.array([])

        events = BehaviorEvents(starts, durations, states)
        vector = events.to_vector()

        assert len(vector) == 0

    def test_from_vector_to_vector_roundtrip(self):
        """Test that from_vector and to_vector are inverses."""
        original = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0])
        events = BehaviorEvents.from_vector(original)
        result = events.to_vector()

        np.testing.assert_array_equal(result, original)

    def test_delete_bouts_matching_borders(self):
        """Test delete_bouts when bordering states match."""
        # Pattern: [0, 0, 1, 1, 0, 0]
        # Delete the bout at index 1 (the 1s), borders match (both 0)
        starts = np.array([0, 2, 4])
        durations = np.array([2, 2, 2])
        states = np.array([0, 1, 0])

        events = BehaviorEvents(starts, durations, states)
        events.delete_bouts([1])

        # Should merge into single bout of 0s
        np.testing.assert_array_equal(events.starts, [0])
        np.testing.assert_array_equal(events.durations, [6])
        np.testing.assert_array_equal(events.states, [0])

    def test_delete_bouts_non_matching_borders_even_duration(self):
        """Test delete_bouts when borders don't match and duration is even."""
        # Pattern: [0, 0, 1, 1, 2, 2]
        # Delete the bout at index 1 (the 1s), borders don't match (0 and 2)
        starts = np.array([0, 2, 4])
        durations = np.array([2, 2, 2])
        states = np.array([0, 1, 2])

        events = BehaviorEvents(starts, durations, states)
        events.delete_bouts([1])

        # Should split evenly: 0s get 1 frame, 2s get 1 frame
        np.testing.assert_array_equal(events.starts, [0, 3])
        np.testing.assert_array_equal(events.durations, [3, 3])
        np.testing.assert_array_equal(events.states, [0, 2])

    def test_delete_bouts_non_matching_borders_odd_duration(self):
        """Test delete_bouts when borders don't match and duration is odd."""
        # Pattern: [0, 0, 1, 1, 1, 2, 2]
        # Delete the bout at index 1 (the 1s with duration 3)
        starts = np.array([0, 2, 5])
        durations = np.array([2, 3, 2])
        states = np.array([0, 1, 2])

        events = BehaviorEvents(starts, durations, states)
        events.delete_bouts([1])

        # Previous gets floor(3/2)=1, next gets ceil(3/2)=2
        np.testing.assert_array_equal(events.starts, [0, 3])
        np.testing.assert_array_equal(events.durations, [3, 4])
        np.testing.assert_array_equal(events.states, [0, 2])

    def test_delete_bouts_first_index(self):
        """Test delete_bouts at first index (merges with next bout)."""
        starts = np.array([0, 2, 4])
        durations = np.array([2, 2, 2])
        states = np.array([0, 1, 0])  # NOT_BEHAVIOR(2), BEHAVIOR(2), NOT_BEHAVIOR(2)

        events = BehaviorEvents(starts, durations, states)
        events.delete_bouts([0])

        # First bout deleted and merged with next (takes next bout's state)
        # Result: BEHAVIOR(4), NOT_BEHAVIOR(2)
        np.testing.assert_array_equal(events.starts, [0, 4])
        np.testing.assert_array_equal(events.durations, [4, 2])
        np.testing.assert_array_equal(events.states, [1, 0])

    def test_delete_bouts_last_index(self):
        """Test delete_bouts at last index (merges with previous bout)."""
        starts = np.array([0, 2, 4])
        durations = np.array([2, 2, 2])
        states = np.array([0, 1, 0])  # NOT_BEHAVIOR(2), BEHAVIOR(2), NOT_BEHAVIOR(2)

        events = BehaviorEvents(starts, durations, states)
        events.delete_bouts([2])

        # Last bout deleted and merged with previous (takes previous bout's state)
        # Result: NOT_BEHAVIOR(2), BEHAVIOR(4)
        np.testing.assert_array_equal(events.starts, [0, 2])
        np.testing.assert_array_equal(events.durations, [2, 4])
        np.testing.assert_array_equal(events.states, [0, 1])

    def test_delete_bouts_multiple(self):
        """Test delete_bouts with multiple indices."""
        # Pattern: [0, 0, 1, 1, 0, 0, 2, 2, 0, 0]
        starts = np.array([0, 2, 4, 6, 8])
        durations = np.array([2, 2, 2, 2, 2])
        states = np.array([0, 1, 0, 2, 0])

        events = BehaviorEvents(starts, durations, states)
        events.delete_bouts([1, 3])

        # After deleting indices 1 and 3, should have merged 0s
        np.testing.assert_array_equal(events.starts, [0])
        np.testing.assert_array_equal(events.durations, [10])
        np.testing.assert_array_equal(events.states, [0])

    def test_delete_bouts_empty_list(self):
        """Test delete_bouts with empty list."""
        starts = np.array([0, 2, 4])
        durations = np.array([2, 2, 2])
        states = np.array([0, 1, 0])

        events = BehaviorEvents(starts, durations, states)
        events.delete_bouts([])

        # Should remain unchanged
        np.testing.assert_array_equal(events.starts, [0, 2, 4])
        np.testing.assert_array_equal(events.durations, [2, 2, 2])
        np.testing.assert_array_equal(events.states, [0, 1, 0])

    def test_rle_simple(self):
        """Test _rle with simple input."""
        vector = np.array([1, 1, 2, 2, 2, 1])
        starts, durations, states = BehaviorEvents._rle(vector)

        np.testing.assert_array_equal(starts, [0, 2, 5])
        np.testing.assert_array_equal(durations, [2, 3, 1])
        np.testing.assert_array_equal(states, [1, 2, 1])

    def test_rle_empty(self):
        """Test _rle with empty input."""
        vector = np.array([])
        starts, durations, states = BehaviorEvents._rle(vector)

        assert len(starts) == 0
        assert len(durations) == 0
        assert len(states) == 0

    def test_rle_single_value(self):
        """Test _rle with single value."""
        vector = np.array([5])
        starts, durations, states = BehaviorEvents._rle(vector)

        np.testing.assert_array_equal(starts, [0])
        np.testing.assert_array_equal(durations, [1])
        np.testing.assert_array_equal(states, [5])

    def test_properties_are_readonly(self):
        """Test that properties return arrays (not settable through property)."""
        starts = np.array([0, 5, 10])
        durations = np.array([5, 5, 5])
        states = np.array([0, 1, 0])

        events = BehaviorEvents(starts, durations, states)

        # Should be able to get properties
        assert events.starts is not None
        assert events.durations is not None
        assert events.states is not None
