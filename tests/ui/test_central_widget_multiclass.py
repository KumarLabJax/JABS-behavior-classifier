"""Focused tests for CentralWidget multiclass prediction decomposition helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    from jabs.core.enums import CrossValidationGroupingStrategy
    from jabs.ui.main_window.central_widget import CentralWidget

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def test_decompose_multiclass_prediction_rows() -> None:
    """Class-index predictions are decomposed to 0/1/2 rows per class."""
    predicted_class = np.array([-1, 0, 2, 1], dtype=np.int8)
    probabilities = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
            [0.2, 0.6, 0.2],
        ],
        dtype=np.float32,
    )

    per_class_preds, per_class_probs = CentralWidget._decompose_multiclass_prediction_rows(
        predicted_class,
        probabilities,
    )

    assert len(per_class_preds) == 3
    assert len(per_class_probs) == 3

    np.testing.assert_array_equal(per_class_preds[0], np.array([0, 2, 1, 1], dtype=np.int16))
    np.testing.assert_array_equal(per_class_preds[1], np.array([0, 1, 1, 2], dtype=np.int16))
    np.testing.assert_array_equal(per_class_preds[2], np.array([0, 1, 2, 1], dtype=np.int16))
    np.testing.assert_array_equal(per_class_probs[0], probabilities[:, 0])
    np.testing.assert_array_equal(per_class_probs[1], probabilities[:, 1])
    np.testing.assert_array_equal(per_class_probs[2], probabilities[:, 2])


def test_decompose_multiclass_prediction_rows_shape_checks() -> None:
    """Invalid shapes raise clear ValueError messages."""
    with pytest.raises(ValueError, match="predicted_class must be 1-D"):
        CentralWidget._decompose_multiclass_prediction_rows(
            np.zeros((2, 2), dtype=np.int8),
            np.zeros((2, 3), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="probabilities must be 2-D"):
        CentralWidget._decompose_multiclass_prediction_rows(
            np.zeros(2, dtype=np.int8),
            np.zeros(2, dtype=np.float32),
        )

    with pytest.raises(ValueError, match="frame length mismatch"):
        CentralWidget._decompose_multiclass_prediction_rows(
            np.zeros(3, dtype=np.int8),
            np.zeros((2, 3), dtype=np.float32),
        )


def test_get_multiclass_prediction_rows_falls_back_on_invalid_shape() -> None:
    """Invalid saved prediction shapes fall back to empty timeline rows."""
    dummy = SimpleNamespace(
        _pose_est=SimpleNamespace(num_identities=1),
        _player_widget=SimpleNamespace(num_frames=4),
        _controls=SimpleNamespace(behaviors=["Walk", "Run"]),
        _predictions={0: np.array([0, 1, 2, 0], dtype=np.int8)},
        _probabilities={0: np.array([0.1, 0.7, 0.2, 0.0], dtype=np.float32)},
        _decompose_multiclass_prediction_rows=CentralWidget._decompose_multiclass_prediction_rows,
    )

    prediction_rows, probability_rows = CentralWidget._get_multiclass_prediction_rows(dummy)

    assert len(prediction_rows) == 1
    assert len(probability_rows) == 1
    assert len(prediction_rows[0]) == 3
    assert len(probability_rows[0]) == 3
    for row in prediction_rows[0]:
        np.testing.assert_array_equal(row, np.zeros(4, dtype=np.int16))
    for row in probability_rows[0]:
        np.testing.assert_array_equal(row, np.zeros(4, dtype=np.float32))


def test_get_multiclass_prediction_rows_falls_back_on_frame_mismatch() -> None:
    """Mismatched frame lengths fall back to empty timeline rows."""
    dummy = SimpleNamespace(
        _pose_est=SimpleNamespace(num_identities=1),
        _player_widget=SimpleNamespace(num_frames=4),
        _controls=SimpleNamespace(behaviors=["Walk", "Run"]),
        _predictions={0: np.array([0, 1, 2], dtype=np.int8)},
        _probabilities={0: np.array([[0.1, 0.7, 0.2]] * 3, dtype=np.float32)},
        _decompose_multiclass_prediction_rows=CentralWidget._decompose_multiclass_prediction_rows,
    )

    prediction_rows, probability_rows = CentralWidget._get_multiclass_prediction_rows(dummy)

    assert len(prediction_rows) == 1
    assert len(probability_rows) == 1
    assert len(prediction_rows[0]) == 3
    assert len(probability_rows[0]) == 3
    for row in prediction_rows[0]:
        np.testing.assert_array_equal(row, np.zeros(4, dtype=np.int16))
    for row in probability_rows[0]:
        np.testing.assert_array_equal(row, np.zeros(4, dtype=np.float32))


def test_count_multiclass_valid_logo_splits_individual() -> None:
    """Valid-split counting matches multiclass LOGO constraints for per-identity groups."""
    counts_by_behavior = {
        "None": {
            "video_a.avi": {
                0: {"fragmented_frame_counts": (20, 0)},
                1: {"fragmented_frame_counts": (20, 0)},
                2: {"fragmented_frame_counts": (20, 0)},
            }
        },
        "Walk": {
            "video_a.avi": {
                0: {"fragmented_frame_counts": (20, 0)},
                1: {"fragmented_frame_counts": (20, 0)},
                2: {"fragmented_frame_counts": (20, 0)},
            }
        },
        "Run": {
            "video_a.avi": {
                0: {"fragmented_frame_counts": (0, 0)},
                1: {"fragmented_frame_counts": (20, 0)},
                2: {"fragmented_frame_counts": (20, 0)},
            }
        },
    }

    valid = CentralWidget._count_multiclass_valid_logo_splits(
        counts_by_behavior=counts_by_behavior,
        behavior_names=["None", "Walk", "Run"],
        grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
        threshold=20,
    )

    assert valid == 3


def test_count_multiclass_valid_logo_splits_video_grouping() -> None:
    """Video grouping aggregates identities per video before validity checks."""
    counts_by_behavior = {
        "None": {
            "video_a.avi": {0: {"fragmented_frame_counts": (20, 0)}},
            "video_b.avi": {0: {"fragmented_frame_counts": (20, 0)}},
        },
        "Walk": {
            "video_a.avi": {0: {"fragmented_frame_counts": (20, 0)}},
            "video_b.avi": {0: {"fragmented_frame_counts": (20, 0)}},
        },
        "Run": {
            "video_a.avi": {0: {"fragmented_frame_counts": (20, 0)}},
            "video_b.avi": {0: {"fragmented_frame_counts": (0, 0)}},
        },
    }

    valid = CentralWidget._count_multiclass_valid_logo_splits(
        counts_by_behavior=counts_by_behavior,
        behavior_names=["None", "Walk", "Run"],
        grouping_strategy=CrossValidationGroupingStrategy.VIDEO,
        threshold=20,
    )

    assert valid == 1
