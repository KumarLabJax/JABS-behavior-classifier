from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QSpinBox

from jabs.behavior.postprocessing.stages import (
    BoutDurationFilterStage,
    BoutStitchingStage,
    GapInterpolationStage,
)
from jabs.core.constants import EVALUATE_POSTPROCESSING_IN_CV_KEY

from .settings_group import SettingsGroup


class InterpolationStageSettingsGroup(SettingsGroup):
    """Settings group for the Bout Gap Interpolation stage."""

    def __init__(self, parent=None):
        """Initialize the Interpolation settings group."""
        super().__init__("Interpolation", parent)

    def _create_controls(self) -> None:
        """Create the settings controls."""
        stage_help = GapInterpolationStage.help()
        self._interpolation_checkbox = QCheckBox("Enable Interpolation Stage")
        self._interpolation_checkbox.setToolTip(stage_help.description)
        self.add_control_row("Enable Interpolation Filter:", self._interpolation_checkbox)

        self._interpolation_max_frames_spinbox = QSpinBox()
        self._interpolation_max_frames_spinbox.setRange(1, 100)
        self._interpolation_max_frames_spinbox.setValue(
            stage_help.kwargs["max_interpolation_gap"].default
        )
        self._interpolation_max_frames_spinbox.setToolTip(
            stage_help.kwargs["max_interpolation_gap"].description
        )
        self.add_control_row("Max Frames to Interpolate:", self._interpolation_max_frames_spinbox)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for post-processing settings."""
        stage_help = GapInterpolationStage.help()
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            f"""
            <h3>Interpolation Stage</h3>

            <p>{stage_help.description_long}</p>
 
            <ul>
              <li><b>Max Frames to Interpolate:</b> {stage_help.kwargs["max_interpolation_gap"].description}</li>
            </ul>
            """
        )
        return help_label

    def get_values(self) -> dict:
        """
        Get current postprocessing settings values.

        Returns:
            Dictionary with setting names and their current values.
        """
        return {
            "stage_name": GapInterpolationStage.__name__,
            "enabled": self._interpolation_checkbox.isChecked(),
            "parameters": {
                "max_interpolation_gap": self._interpolation_max_frames_spinbox.value(),
            },
        }

    def set_values(self, values: dict) -> None:
        """
        Set postprocessing settings values.

        Args:
            values: Dictionary with setting names and their desired values.
        """
        stage_config = values.get(GapInterpolationStage.__name__, {})
        self._interpolation_checkbox.setChecked(stage_config.get("enabled", False))
        self._interpolation_max_frames_spinbox.setValue(
            stage_config.get("parameters", {}).get(
                "max_interpolation_gap",
                GapInterpolationStage.help().kwargs["max_interpolation_gap"].default,
            )
        )


class StitchingStageSettingsGroup(SettingsGroup):
    """Settings group for the Bout Stitching stage."""

    def __init__(self, parent=None):
        """Initialize the Stitching settings group."""
        super().__init__("Stitching Stage", parent)

    def _create_controls(self) -> None:
        """Create the settings controls."""
        stage_help = BoutStitchingStage.help()
        self._stitching_checkbox = QCheckBox("Enable Stitching Stage")
        self._stitching_checkbox.setToolTip(stage_help.description)
        self.add_control_row("Enable Stitching Stage:", self._stitching_checkbox)

        self._stitching_max_gap = QSpinBox()
        self._stitching_max_gap.setRange(1, 100)
        self._stitching_max_gap.setValue(stage_help.kwargs["max_stitch_gap"].default)
        self._stitching_max_gap.setToolTip(stage_help.kwargs["max_stitch_gap"].description)
        self.add_control_row("Max Stitch Gap:", self._stitching_max_gap)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for post-processing settings."""
        stage_help = BoutStitchingStage.help()
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            f"""
            <h3>Stitching Stage</h3>

            <p>{stage_help.description_long}</p>

            <ul>
              <li>
                <b>Max Stitch Gap:</b> {stage_help.kwargs["max_stitch_gap"].description}
              </li>
            </ul>
            """
        )
        return help_label

    def get_values(self) -> dict:
        """
        Get current postprocessing settings values.

        Returns:
            Dictionary with setting names and their current values.
        """
        return {
            "stage_name": BoutStitchingStage.__name__,
            "enabled": self._stitching_checkbox.isChecked(),
            "parameters": {
                "max_stitch_gap": self._stitching_max_gap.value(),
            },
        }

    def set_values(self, values: dict) -> None:
        """
        Set postprocessing settings values.

        Args:
            values: Dictionary with setting names and their desired values.
        """
        stage_config = values.get(BoutStitchingStage.__name__, {})
        self._stitching_checkbox.setChecked(stage_config.get("enabled", False))
        self._stitching_max_gap.setValue(
            stage_config.get("parameters", {}).get(
                "max_stitch_gap", self._stitching_max_gap.value()
            )
        )


class DurationStageSettingsGroup(SettingsGroup):
    """Settings group for the Duration Filtering stage."""

    def __init__(self, parent=None):
        """Initialize the Duration Filtering settings group."""
        super().__init__("Duration Stage", parent)

    def _create_controls(self) -> None:
        """Create the settings controls."""
        stage_help = BoutDurationFilterStage.help()
        self._duration_checkbox = QCheckBox("Enable Duration Stage")
        self._duration_checkbox.setToolTip(stage_help.description)
        self.add_control_row("Enable Duration Stage:", self._duration_checkbox)

        self._duration_min_frames_spinbox = QSpinBox()
        self._duration_min_frames_spinbox.setRange(1, 100)
        self._duration_min_frames_spinbox.setValue(stage_help.kwargs["min_duration"].default)
        self._duration_min_frames_spinbox.setToolTip(stage_help.kwargs["min_duration"].description)
        self.add_control_row("Minimum Bout Duration:", self._duration_min_frames_spinbox)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for post-processing settings."""
        stage_help = BoutDurationFilterStage.help()
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            f"""
            <h3>Duration Stage</h3>

            <p>{stage_help.description_long}</p>

            <ul>
              <li><b>Minimum Bout Duration:</b> {stage_help.kwargs["min_duration"].description}</li>
            </ul>
            """
        )
        return help_label

    def get_values(self) -> dict:
        """
        Get current postprocessing settings values.

        Returns:
            Dictionary with setting names and their current values.
        """
        return {
            "stage_name": BoutDurationFilterStage.__name__,
            "enabled": self._duration_checkbox.isChecked(),
            "parameters": {
                "min_duration": self._duration_min_frames_spinbox.value(),
            },
        }

    def set_values(self, values: dict) -> None:
        """
        Set postprocessing settings values.

        Args:
            values: Dictionary with setting names and their desired values.
        """
        stage_config = values.get(BoutDurationFilterStage.__name__, {})
        self._duration_checkbox.setChecked(stage_config.get("enabled", False))
        self._duration_min_frames_spinbox.setValue(
            stage_config.get("parameters", {}).get(
                "min_duration",
                BoutDurationFilterStage.help().kwargs["min_duration"].default,
            )
        )


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
