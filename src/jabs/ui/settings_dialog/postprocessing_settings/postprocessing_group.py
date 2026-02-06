from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QSpinBox

from jabs.behavior.postprocessing.stages import (
    BoutDurationFilterStage,
    BoutStitchingStage,
    GapInterpolationStage,
)

from ..settings_group import SettingsGroup


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
        self._interpolation_max_frames_spinbox.setValue(5)
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
            "config": {
                "enabled": self._interpolation_checkbox.isChecked(),
                "max_interpolation_gap": self._interpolation_max_frames_spinbox.value(),
            },
        }

    def set_values(self, values: dict) -> None:
        """
        Set postprocessing settings values.

        Args:
            values: Dictionary with setting names and their desired values.
        """
        filter_values = values.get(GapInterpolationStage.__name__, {})
        self._interpolation_checkbox.setChecked(filter_values.get("enabled", False))
        self._interpolation_max_frames_spinbox.setValue(
            filter_values.get("max_interpolation_gap", 5)
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
        self._stitching_max_gap.setValue(3)
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
            "config": {
                "enabled": self._stitching_checkbox.isChecked(),
                "max_stitch_gap": self._stitching_max_gap.value(),
            },
        }

    def set_values(self, values: dict) -> None:
        """
        Set postprocessing settings values.

        Args:
            values: Dictionary with setting names and their desired values.
        """
        filter_values = values.get(BoutStitchingStage.__name__, {})
        self._stitching_checkbox.setChecked(filter_values.get("enabled", False))
        self._stitching_max_gap.setValue(
            filter_values.get("max_stitch_gap", self._stitching_max_gap.value())
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
        self._duration_min_frames_spinbox.setValue(5)
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
            "config": {
                "enabled": self._duration_checkbox.isChecked(),
                "min_duration": self._duration_min_frames_spinbox.value(),
            },
        }

    def set_values(self, values: dict) -> None:
        """
        Set postprocessing settings values.

        Args:
            values: Dictionary with setting names and their desired values.
        """
        filter_values = values.get(BoutDurationFilterStage.__name__, {})
        self._duration_checkbox.setChecked(filter_values.get("enabled", False))
        self._duration_min_frames_spinbox.setValue(filter_values.get("min_duration", 3))
