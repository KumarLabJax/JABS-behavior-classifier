"""Focused tests for CentralWidget multiclass prediction decomposition helpers."""

import numpy as np
import pytest

try:
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
