"""Tests for training-strategy helpers (no Qt required)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from jabs.ui.training_strategy import (
        BinaryTrainingStrategy,
        MultiClassTrainingStrategy,
        _included_row_mask,
    )

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def test_included_row_mask_none_without_exclusions():
    """No excluded groups -> None so callers skip filtering."""
    assert _included_row_mask({"groups": np.array([0, 0, 1, 1])}) is None
    assert _included_row_mask({"groups": np.array([0, 0, 1, 1]), "excluded_groups": set()}) is None


def test_included_row_mask_filters_excluded_rows():
    """Rows whose group is excluded are masked out."""
    features = {"groups": np.array([0, 0, 1, 1, 2, 2]), "excluded_groups": {1}}
    mask = _included_row_mask(features)
    assert mask.tolist() == [True, True, False, False, True, True]


def test_binary_prepare_final_training_applies_behavior_settings():
    """The binary strategy sets the behavior name and settings the final fit needs.

    Cross-validation folds normally do this as a side effect, so a run with zero
    folds reached the final fit with the settings unset and raised
    "Project settings for classifier unset".
    """
    classifier = SimpleNamespace(behavior_name=None, set_project_settings=MagicMock())
    project = SimpleNamespace()
    strategy = BinaryTrainingStrategy(classifier, project, "Walk", (1, 2))

    strategy.prepare_final_training()

    assert classifier.behavior_name == "Walk"
    classifier.set_project_settings.assert_called_once_with(project, "Walk")


def test_multiclass_prepare_final_training_applies_captured_settings():
    """The multi-class strategy applies the settings it captured at construction."""
    settings = {"window_size": 5}
    classifier = SimpleNamespace(project_settings=settings, set_dict_settings=MagicMock())
    strategy = MultiClassTrainingStrategy(classifier, SimpleNamespace(), "Walk")

    strategy.prepare_final_training()

    classifier.set_dict_settings.assert_called_once_with(settings)
