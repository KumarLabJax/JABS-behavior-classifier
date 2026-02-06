import numpy as np
import pytest

from jabs.behavior.events import ClassLabels
from jabs.behavior.postprocessing import PostprocessingPipeline
from jabs.behavior.postprocessing.stages import BoutDurationFilterStage, BoutStitchingStage


class TestPostprocessingPipeline:
    """Tests for PostprocessingPipeline."""

    def test_constructor_empty_config(self):
        """Test constructor with empty config."""
        pipeline = PostprocessingPipeline({})
        assert len(pipeline._filters) == 0

    def test_constructor_single_filter(self):
        """Test constructor with single filter."""
        config = {"BoutDurationFilterStage": {"min_duration": 5}}
        pipeline = PostprocessingPipeline(config)

        assert len(pipeline._filters) == 1
        assert isinstance(pipeline._filters[0], BoutDurationFilterStage)
        assert pipeline._filters[0].config["min_duration"] == 5

    def test_constructor_multiple_filters(self):
        """Test constructor with multiple filters."""
        config = {
            "BoutDurationFilterStage": {"min_duration": 5},
            "BoutStitchingStage": {"max_stitch_gap": 3},
        }
        pipeline = PostprocessingPipeline(config)

        assert len(pipeline._filters) == 2
        assert isinstance(pipeline._filters[0], BoutDurationFilterStage)
        assert isinstance(pipeline._filters[1], BoutStitchingStage)

    def test_constructor_unrecognized_filter(self):
        """Test that constructor raises error for unrecognized filter."""
        config = {"NonExistentFilter": {}}

        with pytest.raises(ValueError, match="Filter 'NonExistentFilter' is not recognized"):
            PostprocessingPipeline(config)

    def test_constructor_filter_with_none_kwargs(self):
        """Test constructor when filter config value is None."""
        config = {"BoutDurationFilterStage": None}

        with pytest.raises(ValueError):
            # Should fail because DurationFilter requires min_duration
            PostprocessingPipeline(config)

    def test_run_empty_pipeline(self):
        """Test run with empty pipeline."""
        pipeline = PostprocessingPipeline({})
        classes = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )

        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = pipeline.run(classes, probabilities)
        # Should remain unchanged
        np.testing.assert_array_equal(result, classes)

    def test_run_single_filter(self):
        """Test run with single filter."""
        config = {"BoutDurationFilterStage": {"min_duration": 3}}
        pipeline = PostprocessingPipeline(config)

        classes = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,  # Short bout (2)
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )

        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = pipeline.run(classes, probabilities)

        # Short behavior bout should be removed
        expected = np.array([ClassLabels.NOT_BEHAVIOR] * 6)
        np.testing.assert_array_equal(result, expected)

    def test_run_multiple_filters_sequential(self):
        """Test that filters are applied sequentially."""
        config = {
            "BoutStitchingStage": {"max_stitch_gap": 2},
            "BoutDurationFilterStage": {"min_duration": 5},
        }
        pipeline = PostprocessingPipeline(config)

        # Pattern: BEHAVIOR(2), NOT_BEHAVIOR(1), BEHAVIOR(2), NOT_BEHAVIOR(5)
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
            ]
        )

        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = pipeline.run(classes, probabilities)

        # First, stitching should combine first two BEHAVIOR bouts -> BEHAVIOR(5)
        # Then, duration filter should keep it (duration >= 5)
        # Final: BEHAVIOR(5), NOT_BEHAVIOR(5)
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
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_run_order_matters(self):
        """Test that filter order affects results."""
        # Apply duration filter first, then stitching
        config1 = {
            "BoutDurationFilterStage": {"min_duration": 4},
            "BoutStitchingStage": {"max_stitch_gap": 2},
        }
        pipeline1 = PostprocessingPipeline(config1)

        # Apply stitching first, then duration filter
        config2 = {
            "BoutStitchingStage": {"max_stitch_gap": 2},
            "BoutDurationFilterStage": {"min_duration": 4},
        }
        pipeline2 = PostprocessingPipeline(config2)

        # Pattern: NOT_BEHAVIOR(2), BEHAVIOR(2), NOT_BEHAVIOR(1), BEHAVIOR(2), NOT_BEHAVIOR(2)
        # Short BEHAVIOR bouts are NOT at boundaries so they can be removed
        classes = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )

        probabilities = np.full_like(classes, 0.5, dtype=float)
        result1 = pipeline1.run(classes, probabilities)
        result2 = pipeline2.run(classes, probabilities)

        # Pipeline 1: Duration filter removes both short BEHAVIOR bouts first
        # After removal, NOT_BEHAVIOR sections merge
        expected1 = np.array([ClassLabels.NOT_BEHAVIOR] * 9)

        # Pipeline 2: Stitching removes ALL NOT_BEHAVIOR bouts <= 2 (including boundaries!)
        # Removes indices [0, 2, 4]:
        #   - Index 4 (last): merges with previous BEHAVIOR
        #   - Index 2 (middle): merges surrounding BEHAVIORs
        #   - Index 0 (first): merges with next BEHAVIOR (takes BEHAVIOR state)
        # Result after stitching: BEHAVIOR(9)
        # Duration filter: BEHAVIOR(9) >= 4, so nothing removed
        expected2 = np.array([ClassLabels.BEHAVIOR] * 9)

        np.testing.assert_array_equal(result1, expected1)
        np.testing.assert_array_equal(result2, expected2)

    def test_run_empty_array(self):
        """Test run with empty array."""
        config = {
            "BoutDurationFilterStage": {"min_duration": 5},
            "BoutStitchingStage": {"max_stitch_gap": 3},
        }
        pipeline = PostprocessingPipeline(config)

        classes = np.array([])
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = pipeline.run(classes, probabilities)

        assert len(result) == 0

    def test_run_all_same_state(self):
        """Test run with array of all same state."""
        config = {
            "BoutDurationFilterStage": {"min_duration": 5},
            "BoutStitchingStage": {"max_stitch_gap": 3},
        }
        pipeline = PostprocessingPipeline(config)

        classes = np.array([ClassLabels.BEHAVIOR] * 10)
        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = pipeline.run(classes, probabilities)

        # Should remain unchanged
        np.testing.assert_array_equal(result, classes)

    def test_run_complex_sequence(self):
        """Test run with complex sequence of behaviors."""
        config = {
            "BoutStitchingStage": {"max_stitch_gap": 2},
            "BoutDurationFilterStage": {"min_duration": 4},
        }
        pipeline = PostprocessingPipeline(config)

        # Complex pattern
        classes = np.array(
            [
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,  # (3)
                ClassLabels.NOT_BEHAVIOR,  # gap (1)
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,  # (2)
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,  # gap (3)
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,  # (3)
                ClassLabels.NOT_BEHAVIOR,  # gap (1)
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,  # (2)
                ClassLabels.NOT_BEHAVIOR,
                ClassLabels.NOT_BEHAVIOR,
            ]
        )

        probabilities = np.full_like(classes, 0.5, dtype=float)
        result = pipeline.run(classes, probabilities)

        # After stitching (removes ALL NOT_BEHAVIOR bouts <= 2, including boundaries):
        # Removes indices [0, 2, 6, 8]:
        #   - Index 8 (last, NOT_BEHAVIOR(2)): merges with previous BEHAVIOR(2) → BEHAVIOR(4)
        #   - Index 6 (NOT_BEHAVIOR(1)): merges surrounding BEHAVIORs → BEHAVIOR(9)
        #   - Index 2 (NOT_BEHAVIOR(1)): merges surrounding BEHAVIORs → BEHAVIOR(6)
        #   - Index 0 (first, NOT_BEHAVIOR(2)): merges with next BEHAVIOR → BEHAVIOR(8)
        # Result: BEHAVIOR(8), NOT_BEHAVIOR(3), BEHAVIOR(9)
        # After duration filter (min 4): All bouts >= 4, so nothing removed
        expected = np.array(
            [
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
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
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
                ClassLabels.BEHAVIOR,
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_constructor_preserves_filter_config(self):
        """Test that filter configurations are properly passed to filters."""
        config = {
            "BoutDurationFilterStage": {"min_duration": 10},
        }
        pipeline = PostprocessingPipeline(config)

        assert pipeline._filters[0].config["min_duration"] == 10
