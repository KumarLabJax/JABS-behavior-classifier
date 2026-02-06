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
        self._interpolation_checkbox = QCheckBox("Enable Interpolation Stage")
        self._interpolation_checkbox.setToolTip(
            "Fill short gaps in predicted behavior bouts by interpolating missing frames."
        )
        self.add_control_row("Enable Interpolation Filter:", self._interpolation_checkbox)

        self._interpolation_max_frames_spinbox = QSpinBox()
        self._interpolation_max_frames_spinbox.setRange(1, 100)
        self._interpolation_max_frames_spinbox.setValue(5)
        self._interpolation_max_frames_spinbox.setToolTip(
            "Maximum number of consecutive frames to interpolate."
        )
        self.add_control_row("Max Frames to Interpolate:", self._interpolation_max_frames_spinbox)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for post-processing settings."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>Interpolation Stage</h3>

            <p>
              The Interpolation Stage fills short gaps in predictions (such as when there is missing pose) by
              interpolating the class for the missing frames. The missing frames are interpolated using the
              surrounding classes -- if the class on both sides of the gap is the same, the gap is filled
              with that class. If the classes differ, the is gap is split between the two classes so that the
              first half matches the previous class and the second half matches the following class.
            </p>
 
            <ul>
              <li><b>Enable Interpolation Stage:</b> Check this box to activate the interpolation stage.</li>
              <li><b>Max Frames to Interpolate:</b> Specify the maximum number of consecutive frames that can be 
              interpolated. Gaps shorter than or equal to this value will be filled in.</li>
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
        self._stitching_checkbox = QCheckBox("Enable Stitching Stage")
        self._stitching_checkbox.setToolTip(
            "Stitches together behavior bouts separated by short gaps."
        )
        self.add_control_row("Enable Stitching Stage:", self._stitching_checkbox)

        self._stitching_max_gap = QSpinBox()
        self._stitching_max_gap.setRange(1, 100)
        self._stitching_max_gap.setValue(3)
        self._stitching_max_gap.setToolTip(
            "Maximum number of consecutive frames between bouts to stitch."
        )
        self.add_control_row("Max Stitch Gap:", self._stitching_max_gap)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for post-processing settings."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>Stitching Stage</h3>

            <p>The Stitching Stage connects behavior bouts that are separated by short gaps of not-behavior prediction.</p>

            <ul>
              <li><b>Enable Stitching Stage:</b> Check this box to activate the stitching stage.</li>
                <li>
                  <b>Max Stitch Gap:</b> Specify the maximum number of consecutive frames between bouts that 
                  can be stitched together. Gaps shorter than or equal to this value will be removed by merging
                  the surrounding bouts.
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
        self._duration_checkbox = QCheckBox("Enable Duration Stage")
        self._duration_checkbox.setToolTip(
            "Removes short behavior bouts below a minimum duration threshold."
        )
        self.add_control_row("Enable Duration Stage:", self._duration_checkbox)

        self._duration_min_frames_spinbox = QSpinBox()
        self._duration_min_frames_spinbox.setRange(1, 100)
        self._duration_min_frames_spinbox.setValue(5)
        self._duration_min_frames_spinbox.setToolTip(
            "Minimum duration (in frames) for a behavior bout to be kept."
        )
        self.add_control_row("Minimum Bout Duration:", self._duration_min_frames_spinbox)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for post-processing settings."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>Duration Stage</h3>

            <p>The Duration Stage removes short bouts of predicted behavior.</p>

            <ul>
              <li><b>Enable Duration Stage:</b> Check this box to activate the duration stage.</li>
              <li><b>Minimum Bout Duration:</b> Specify the minimum bout length (in frames). Any bout shorter than this will be removed.</li>
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
