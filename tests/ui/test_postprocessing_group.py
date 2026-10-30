from __future__ import annotations

import logging

import pytest

from jabs.behavior.postprocessing.stages import BoutStitchingStage

try:
    from PySide6.QtWidgets import QApplication, QSpinBox

    from jabs.ui.settings_dialog.postprocessing_group import (
        DurationStageSettingsGroup,
        InterpolationStageSettingsGroup,
        SingleParameterStageSettingsGroup,
        StitchingStageSettingsGroup,
        _ParameterSpinBox,
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
_INT_MAX = 0 if SKIP_UI_TESTS else SingleParameterStageSettingsGroup._QSPINBOX_INT_MAX
_LOGGER_NAME = "jabs.ui.settings_dialog.postprocessing_group"


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
@pytest.mark.parametrize("value", [900, _INT_MAX], ids=["above_old_cap", "spinbox_maximum"])
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
@pytest.mark.parametrize(
    ("value", "expected"),
    [(_INT_MAX + 1, _INT_MAX), (0, 1), (-1, 1)],
    ids=["above_maximum", "zero", "negative"],
)
def test_stage_group_clamps_out_of_range_parameter(
    group_class: type[SingleParameterStageSettingsGroup],
    value: int,
    expected: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A saved parameter outside the spin box range is clamped with a warning."""
    stage_name = group_class._stage.__name__
    parameter = group_class._parameter
    group = group_class()

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        group.set_values({stage_name: {"enabled": True, "parameters": {parameter: value}}})

    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": True,
        "parameters": {parameter: expected},
    }
    assert "outside the range" in caplog.text


@pytest.mark.parametrize(
    "group_class", _STAGE_GROUP_CLASSES, ids=lambda group_class: group_class.__name__
)
@pytest.mark.parametrize("value", [5.5, "10", None, True], ids=["float", "string", "null", "bool"])
def test_stage_group_non_integer_parameter_uses_default(
    group_class: type[SingleParameterStageSettingsGroup],
    value: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A saved parameter that is not an integer falls back to the default with a warning."""
    stage_name = group_class._stage.__name__
    parameter = group_class._parameter
    default = group_class._stage.help().kwargs[parameter].default
    group = group_class()

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        group.set_values({stage_name: {"enabled": True, "parameters": {parameter: value}}})

    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": True,
        "parameters": {parameter: default},
    }
    assert "is not an integer" in caplog.text


@pytest.mark.parametrize(
    "group_class", _STAGE_GROUP_CLASSES, ids=lambda group_class: group_class.__name__
)
@pytest.mark.parametrize("parameters", [None, [5], "5"], ids=["null", "list", "string"])
def test_stage_group_non_dict_parameters_uses_default(
    group_class: type[SingleParameterStageSettingsGroup], parameters: object
) -> None:
    """A stage whose ``parameters`` isn't a dict (``null`` is schema-valid) loads its default."""
    stage_name = group_class._stage.__name__
    parameter = group_class._parameter
    default = group_class._stage.help().kwargs[parameter].default
    group = group_class()

    group.set_values({stage_name: {"enabled": True, "parameters": parameters}})

    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": True,
        "parameters": {parameter: default},
    }


@pytest.mark.parametrize(
    "group_class", _STAGE_GROUP_CLASSES, ids=lambda group_class: group_class.__name__
)
def test_stage_group_entry_without_enabled_is_enabled(
    group_class: type[SingleParameterStageSettingsGroup],
) -> None:
    """A stage entry with no ``enabled`` key shows as enabled, as PostprocessingPipeline runs it."""
    stage_name = group_class._stage.__name__
    parameter = group_class._parameter
    group = group_class()

    group.set_values({stage_name: {"parameters": {parameter: 5}}})

    assert group.get_values() == {
        "stage_name": stage_name,
        "enabled": True,
        "parameters": {parameter: 5},
    }


def test_parameter_spinbox_is_sized_for_display_maximum() -> None:
    """The parameter spin box is sized like one whose maximum is the display maximum."""
    display_maximum = SingleParameterStageSettingsGroup._PARAMETER_DISPLAY_MAXIMUM
    spinbox = _ParameterSpinBox(display_maximum)
    spinbox.setRange(1, _INT_MAX)
    display_reference = QSpinBox()
    display_reference.setRange(1, display_maximum)
    full_range_reference = QSpinBox()
    full_range_reference.setRange(1, _INT_MAX)

    width = spinbox.sizeHint().width()
    assert width == display_reference.sizeHint().width()
    assert width < full_range_reference.sizeHint().width()


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
