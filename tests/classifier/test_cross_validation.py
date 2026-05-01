"""Tests for cross-validation helpers."""

import numpy as np
import pandas as pd

from jabs.classifier.cross_validation import run_leave_one_group_out_cv


class _NoSplitClassifier:
    """Classifier test double reporting no valid LOGO splits."""

    @staticmethod
    def get_leave_one_group_out_max(_labels, _groups) -> int:
        return 0

    @staticmethod
    def leave_one_group_out(*_args, **_kwargs):
        raise AssertionError("leave_one_group_out should not be called when max splits is zero")


def test_run_leave_one_group_out_cv_returns_empty_when_no_valid_splits() -> None:
    """No valid CV splits should not raise; CV is skipped with empty results."""
    features = {
        "per_frame": pd.DataFrame({"a": [1.0, 2.0]}),
        "window": pd.DataFrame({"b": [3.0, 4.0]}),
        "labels": np.array([0, 1], dtype=np.int8),
        "groups": np.array([0, 1], dtype=np.int32),
    }
    status_messages: list[str] = []
    results = run_leave_one_group_out_cv(
        classifier=_NoSplitClassifier(),
        project=object(),
        features=features,
        group_mapping={},
        behavior="Walk",
        k=1,
        status_callback=status_messages.append,
    )

    assert results == []
    assert any("skipping CV" in msg for msg in status_messages)
