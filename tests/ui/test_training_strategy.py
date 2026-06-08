"""Tests for training-strategy helpers (no Qt required)."""

import numpy as np

from jabs.ui.training_strategy import _included_row_mask


def test_included_row_mask_none_without_exclusions():
    """No excluded groups -> None so callers skip filtering."""
    assert _included_row_mask({"groups": np.array([0, 0, 1, 1])}) is None
    assert _included_row_mask({"groups": np.array([0, 0, 1, 1]), "excluded_groups": set()}) is None


def test_included_row_mask_filters_excluded_rows():
    """Rows whose group is excluded are masked out."""
    features = {"groups": np.array([0, 0, 1, 1, 2, 2]), "excluded_groups": {1}}
    mask = _included_row_mask(features)
    assert mask.tolist() == [True, True, False, False, True, True]
