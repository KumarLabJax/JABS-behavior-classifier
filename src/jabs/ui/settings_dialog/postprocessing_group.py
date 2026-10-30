"""Settings groups for the per-behavior prediction postprocessing stages."""

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QSpinBox, QWidget

from jabs.behavior.postprocessing.stages import (
    BoutDurationFilterStage,
    BoutStitchingStage,
    GapInterpolationStage,
    PostprocessingStage,
)
from jabs.core.constants import EVALUATE_POSTPROCESSING_IN_CV_KEY

from .settings_group import SettingsGroup

logger = logging.getLogger(__name__)


class SingleParameterStageSettingsGroup(SettingsGroup):
    """Settings group for a postprocessing stage with one integer parameter.

    Every stage in the pipeline is configured the same way: a checkbox that enables
    it and a spin box holding its single integer tuning parameter, both described by
    the stage's own :meth:`~jabs.behavior.postprocessing.stages.PostprocessingStage.help`.
    Subclasses supply the stage class, the parameter name and the labels; the widget
    construction, documentation and the save/load round trip live here so the groups
    cannot drift apart.

    Subclasses must set every class attribute listed below; a subclass that leaves one
    unset, or names a parameter its stage does not document, fails when it is defined.

    Attributes:
        _stage: Stage class this group configures.
        _parameter: Name of the stage's single integer keyword argument.
        _group_title: Title shown in the group box header.
        _doc_heading: Heading of the collapsible documentation section.
        _enable_row_label: Label of the row holding the enable checkbox.
        _enable_checkbox_label: Text shown beside the enable checkbox.
        _parameter_row_label: Label of the row holding the parameter spin box.
    """

    _stage: type[PostprocessingStage]
    _parameter: str
    _group_title: str
    _doc_heading: str
    _enable_row_label: str
    _enable_checkbox_label: str
    _parameter_row_label: str

    # Bounds of the parameter spin box. Every stage counts frames, so the same
    # range applies to all of them. The stages only require a positive integer, so
    # the maximum is the largest value a QSpinBox holds; a lower cap would silently
    # rewrite larger saved values (e.g. set from the CLI) when the dialog is saved.
    _PARAMETER_MINIMUM = 1
    _PARAMETER_MAXIMUM = 2**31 - 1

    _REQUIRED_ATTRIBUTES = (
        "_stage",
        "_parameter",
        "_group_title",
        "_doc_heading",
        "_enable_row_label",
        "_enable_checkbox_label",
        "_parameter_row_label",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check that a subclass sets every required attribute and a documented parameter.

        Raises:
            TypeError: If a required class attribute is unset, or the stage's help does
                not document ``_parameter``.
        """
        super().__init_subclass__(**kwargs)
        missing = [name for name in cls._REQUIRED_ATTRIBUTES if not hasattr(cls, name)]
        if missing:
            raise TypeError(f"{cls.__name__} must define {', '.join(missing)}")
        if cls._parameter not in (cls._stage.help().kwargs or {}):
            raise TypeError(
                f"{cls.__name__}: {cls._stage.__name__}.help() does not document "
                f"parameter {cls._parameter!r}"
            )

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the settings group.

        Args:
            parent: Parent widget for this settings group.
        """
        super().__init__(self._group_title, parent)

    def _create_controls(self) -> None:
        """Create the enable checkbox and the parameter spin box."""
        # SettingsGroup builds the controls before the documentation, so the help
        # fetched here is reused by _create_documentation() and set_values().
        self._stage_help = self._stage.help()
        stage_help = self._stage_help
        kwarg_help = stage_help.kwargs[self._parameter]

        self._enable_checkbox = QCheckBox(self._enable_checkbox_label)
        self._enable_checkbox.setToolTip(stage_help.description)
        self.add_control_row(self._enable_row_label, self._enable_checkbox)

        self._parameter_spinbox = QSpinBox()
        self._parameter_spinbox.setRange(self._PARAMETER_MINIMUM, self._PARAMETER_MAXIMUM)
        self._parameter_spinbox.setValue(kwarg_help.default)
        self._parameter_spinbox.setToolTip(kwarg_help.description)
        self.add_control_row(self._parameter_row_label, self._parameter_spinbox)

    def _create_documentation(self) -> QLabel:
        """Create help documentation describing the stage and its parameter."""
        stage_help = self._stage_help
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            f"""
            <h3>{self._doc_heading}</h3>

            <p>{stage_help.description_long}</p>

            <ul>
              <li><b>{self._parameter_row_label}</b>
                {stage_help.kwargs[self._parameter].description}</li>
            </ul>
            """
        )
        return help_label

    def get_values(self) -> dict:
        """Get the current settings for this stage.

        Returns:
            Dictionary with the stage name, whether the stage is enabled, and its
            parameter value.
        """
        return {
            "stage_name": self._stage.__name__,
            "enabled": self._enable_checkbox.isChecked(),
            "parameters": {self._parameter: self._parameter_spinbox.value()},
        }

    def set_values(self, values: dict) -> None:
        """Set this stage's controls from a saved postprocessing configuration.

        Settings this stage has no saved entry for fall back to the stage's own
        defaults, so loading a configuration always leaves the controls describing
        that configuration rather than whatever was shown before.

        Args:
            values: Postprocessing settings keyed by stage name. Entries for other
                stages are ignored.
        """
        stage_name = self._stage.__name__
        if stage_name not in values:
            logger.debug("No saved settings for %s; using stage defaults", stage_name)
        stage_config = values.get(stage_name, {})
        self._enable_checkbox.setChecked(stage_config.get("enabled", False))
        self._parameter_spinbox.setValue(
            stage_config.get("parameters", {}).get(
                self._parameter, self._stage_help.kwargs[self._parameter].default
            )
        )


class InterpolationStageSettingsGroup(SingleParameterStageSettingsGroup):
    """Settings group for the Bout Gap Interpolation stage."""

    _stage = GapInterpolationStage
    _parameter = "max_interpolation_gap"
    _group_title = "Interpolation"
    _doc_heading = "Interpolation Stage"
    _enable_row_label = "Enable Interpolation Filter:"
    _enable_checkbox_label = "Enable Interpolation Stage"
    _parameter_row_label = "Max Frames to Interpolate:"


class StitchingStageSettingsGroup(SingleParameterStageSettingsGroup):
    """Settings group for the Bout Stitching stage."""

    _stage = BoutStitchingStage
    _parameter = "max_stitch_gap"
    _group_title = "Stitching Stage"
    _doc_heading = "Stitching Stage"
    _enable_row_label = "Enable Stitching Stage:"
    _enable_checkbox_label = "Enable Stitching Stage"
    _parameter_row_label = "Max Stitch Gap:"


class DurationStageSettingsGroup(SingleParameterStageSettingsGroup):
    """Settings group for the Duration Filtering stage."""

    _stage = BoutDurationFilterStage
    _parameter = "min_duration"
    _group_title = "Duration Stage"
    _doc_heading = "Duration Stage"
    _enable_row_label = "Enable Duration Stage:"
    _enable_checkbox_label = "Enable Duration Stage"
    _parameter_row_label = "Minimum Bout Duration:"


class PostprocessingEvaluationSettingsGroup(SettingsGroup):
    """Settings group controlling postprocessing evaluation during cross-validation."""

    def __init__(self, parent=None):
        """Initialize the postprocessing evaluation settings group."""
        super().__init__("Cross-Validation Evaluation", parent)

    def _create_controls(self) -> None:
        """Create the settings controls."""
        self._evaluate_checkbox = QCheckBox("Evaluate postprocessing during cross-validation")
        self._evaluate_checkbox.setToolTip(
            "Also report cross-validation metrics with the stages above applied, "
            "so you can see how they affect classifier performance."
        )
        self.add_control_row("Evaluate in Cross-Validation:", self._evaluate_checkbox)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for the cross-validation evaluation setting."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>Evaluating Postprocessing During Cross-Validation</h3>

            <p>When enabled, each cross-validation iteration reports a second set of
            metrics with the enabled postprocessing stages applied. The training report
            shows both, so you can compare raw classifier performance against
            performance after stitching, duration filtering, and interpolation.</p>

            <p>Because the stages reason about contiguous bouts, the held-out animal's
            <i>entire</i> track is predicted before the stages are applied - the same way
            predictions are generated when you classify. Metrics are then computed only
            on the labeled frames, where ground truth exists. This is what makes the
            comparison meaningful: filters that depend on gaps between bouts, or on
            frames with no prediction at all, behave exactly as they would at
            prediction time.</p>

            <p><b>Note:</b> This makes training slower, because every held-out animal's
            full track is predicted in addition to training. The added cost is roughly
            one classification pass over the labeled animals. It is slower still the
            first time a behavior is trained after a feature cache is cleared or
            invalidated, since the full-track features have to be computed rather than
            read from the cache.</p>

            <p><b>Note:</b> Prediction postprocessing is only available for binary
            classifiers, so this setting has no effect in multi-class mode.</p>
            """
        )
        return help_label

    def get_values(self) -> dict:
        """
        Get the current cross-validation evaluation setting.

        Returns:
            Dictionary with the setting name and its current value.
        """
        return {EVALUATE_POSTPROCESSING_IN_CV_KEY: self._evaluate_checkbox.isChecked()}

    def set_values(self, values: dict) -> None:
        """
        Set the cross-validation evaluation setting.

        Args:
            values: Dictionary with setting names and their desired values.
        """
        self._evaluate_checkbox.setChecked(
            bool(values.get(EVALUATE_POSTPROCESSING_IN_CV_KEY, False))
        )
