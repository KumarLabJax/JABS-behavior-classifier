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
    project = _project_with_postprocessing(evaluate=False, stages=[])
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


_STITCH_STAGE = {
    "stage_name": "BoutStitchingStage",
    "enabled": True,
    "parameters": {"max_stitch_gap": 3},
}


def _project_with_postprocessing(evaluate: bool, stages: list[dict]) -> SimpleNamespace:
    """Build a project stub whose settings manager answers the postprocessing reads."""
    return SimpleNamespace(
        settings_manager=SimpleNamespace(
            evaluate_postprocessing_in_cv=lambda _behavior: evaluate,
            postprocessing_config=lambda _behavior: stages,
        )
    )


def test_binary_strategy_reports_postprocessing_from_project_settings():
    """The binary strategy sources the decision itself rather than being told."""
    project = _project_with_postprocessing(evaluate=True, stages=[_STITCH_STAGE])
    strategy = BinaryTrainingStrategy(SimpleNamespace(), project, "Walk", (1, 2))

    assert strategy.evaluate_postprocessing is True
    assert strategy.postprocessing_stages == [_STITCH_STAGE]


def test_binary_strategy_omits_stages_when_evaluation_is_off():
    """With the behavior's flag off, nothing is reported even if stages exist."""
    project = _project_with_postprocessing(evaluate=False, stages=[_STITCH_STAGE])
    strategy = BinaryTrainingStrategy(SimpleNamespace(), project, "Walk", (1, 2))

    assert strategy.evaluate_postprocessing is False
    assert strategy.postprocessing_stages is None


def test_binary_strategy_drops_disabled_stages():
    """Only enabled stages are recorded, matching what the pipeline would run."""
    project = _project_with_postprocessing(
        evaluate=True,
        stages=[{"stage_name": "GapInterpolationStage", "enabled": False}, _STITCH_STAGE],
    )
    strategy = BinaryTrainingStrategy(SimpleNamespace(), project, "Walk", (1, 2))

    assert strategy.postprocessing_stages == [_STITCH_STAGE]


def test_binary_strategy_keeps_the_postprocessing_config_it_captured():
    """Editing the settings after the strategy is built changes neither CV nor the report."""
    stages = [dict(_STITCH_STAGE)]
    project = _project_with_postprocessing(evaluate=True, stages=stages)
    strategy = BinaryTrainingStrategy(SimpleNamespace(), project, "Walk", (1, 2))

    stages[0]["enabled"] = False
    stages.append({"stage_name": "GapInterpolationStage", "enabled": True})
    project.settings_manager.evaluate_postprocessing_in_cv = lambda _behavior: False

    assert strategy.evaluate_postprocessing is True
    assert strategy.postprocessing_config == [_STITCH_STAGE]
    assert strategy.postprocessing_stages == [_STITCH_STAGE]


def test_multiclass_strategy_never_requests_postprocessing():
    """Postprocessing is binary-only, so the multi-class strategy inherits the defaults.

    The strategy type is the mode check: this is what lets ``TrainingThread``
    stay mode-agnostic instead of re-deriving ``is_multiclass`` itself.
    """
    classifier = SimpleNamespace(project_settings={"window_size": 5})
    # even a project configured to evaluate must not pull multi-class in
    project = _project_with_postprocessing(evaluate=True, stages=[_STITCH_STAGE])
    strategy = MultiClassTrainingStrategy(classifier, project, "Walk")

    assert strategy.evaluate_postprocessing is False
    assert strategy.postprocessing_stages is None
