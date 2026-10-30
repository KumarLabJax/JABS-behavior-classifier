from __future__ import annotations

import pytest

from jabs.behavior.postprocessing.stages import BoutStitchingStage

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.settings_dialog.postprocessing_group import (
        DurationStageSettingsGroup,
        InterpolationStageSettingsGroup,
        SingleParameterStageSettingsGroup,
        StitchingStageSettingsGroup,
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

_STAGE_GROUP_CLASSES = (
    []
    if SKIP_UI_TESTS
    else [
        InterpolationStageSettingsGroup,
        StitchingStageSettingsGroup,
        DurationStageSettingsGroup,
    ]
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for UI-related tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.mark.parametrize(
    "group_class", _STAGE_GROUP_CLASSES, ids=lambda group_class: group_class.__name__
)
def test_stage_group_round_trips_and_resets_to_stage_default(
    group_class: type[SingleParameterStageSettingsGroup],
) -> None:
    """A stage group loads its saved parameter and resets when none is saved.

    Loading a configuration with no entry for a stage must leave that stage's
    controls showing the stage's own default rather than whatever the previous
    configuration happened to put there.
    """
    stage_name = group_class._stage.__name__
    parameter = group_class._parameter
    default = group_class._stage.help().kwargs[parameter].default
    group = group_class()

    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": False,
        "parameters": {parameter: default},
    }

    group.set_values({stage_name: {"enabled": True, "parameters": {parameter: 42}}})
    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": True,
        "parameters": {parameter: 42},
    }

    # a saved configuration with no entry for this stage falls back to the default
    group.set_values({})
    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": False,
        "parameters": {parameter: default},
    }


@pytest.mark.parametrize(
    "group_class", _STAGE_GROUP_CLASSES, ids=lambda group_class: group_class.__name__
)
@pytest.mark.parametrize("value", [900, 2**31 - 1], ids=["above_old_cap", "spinbox_maximum"])
def test_stage_group_keeps_large_saved_parameter(
    group_class: type[SingleParameterStageSettingsGroup], value: int
) -> None:
    """A saved parameter above the old 100-frame cap, up to the spin box maximum, round-trips."""
    stage_name = group_class._stage.__name__
    parameter = group_class._parameter
    group = group_class()

    group.set_values({stage_name: {"enabled": True, "parameters": {parameter: value}}})
    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": True,
        "parameters": {parameter: value},
    }


@pytest.mark.parametrize(
    "group_class", _STAGE_GROUP_CLASSES, ids=lambda group_class: group_class.__name__
)
def test_stage_group_clamps_parameter_beyond_spinbox_range(
    group_class: type[SingleParameterStageSettingsGroup],
) -> None:
    """A saved parameter too large for a QSpinBox is clamped instead of raising."""
    stage_name = group_class._stage.__name__
    parameter = group_class._parameter
    group = group_class()

    group.set_values({stage_name: {"enabled": True, "parameters": {parameter: 2**31}}})
    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": True,
        "parameters": {parameter: 2**31 - 1},
    }


def test_stage_group_missing_attribute_fails_at_definition() -> None:
    """A subclass that leaves a required attribute unset is rejected when defined."""
    with pytest.raises(TypeError, match="_doc_heading"):

        class _Incomplete(SingleParameterStageSettingsGroup):
            _stage = BoutStitchingStage
            _parameter = "max_stitch_gap"
            _group_title = "Stitching"
            _enable_row_label = "Enable:"
            _enable_checkbox_label = "Enable"
            _parameter_row_label = "Gap:"


def test_stage_group_undocumented_parameter_fails_at_definition() -> None:
    """A subclass naming a parameter its stage does not document is rejected when defined."""
    with pytest.raises(TypeError, match="not_a_parameter"):

        class _WrongParameter(SingleParameterStageSettingsGroup):
            _stage = BoutStitchingStage
            _parameter = "not_a_parameter"
            _group_title = "Stitching"
            _doc_heading = "Stitching"
            _enable_row_label = "Enable:"
            _enable_checkbox_label = "Enable"
            _parameter_row_label = "Gap:"
