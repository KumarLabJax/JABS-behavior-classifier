"""Classifier mode settings group for configuring binary vs. multi-class training."""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLabel, QSizePolicy

from jabs.core.constants import CLASSIFIER_MODE_KEY
from jabs.core.enums import DEFAULT_CLASSIFIER_MODE, ClassifierMode

from .settings_group import SettingsGroup

logger = logging.getLogger(__name__)

_CLASSIFIER_MODE_LABELS: dict[ClassifierMode, str] = {
    ClassifierMode.BINARY: "Binary",
    ClassifierMode.MULTICLASS: "Multi-class",
}


class ClassifierModeSettingsGroup(SettingsGroup):
    """Settings group for classifier mode configuration.

    Controls whether JABS trains one binary classifier per behavior or a single
    multi-class classifier across all behaviors simultaneously.
    """

    def __init__(self, parent=None):
        """Initialize the classifier mode settings group."""
        super().__init__("Classifier Mode", parent)

    def _create_controls(self) -> None:
        """Create the classifier mode combo box control."""
        self._mode_combo = QComboBox()
        for enum_val in ClassifierMode:
            self._mode_combo.addItem(_CLASSIFIER_MODE_LABELS[enum_val], enum_val)
        self._mode_combo.setCurrentIndex(self._mode_combo.findData(DEFAULT_CLASSIFIER_MODE))
        self._mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.add_control_row("Mode:", self._mode_combo)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for classifier mode settings."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>What is Classifier Mode?</h3>
            <p>Classifier mode controls whether JABS trains one classifier per behavior
            or a single multi-class classifier across all behaviors simultaneously.</p>

            <ul>
              <li><b>Binary (default):</b> Trains one independent binary classifier for 
              the currently active behavior. Each classifier predicts whether a given
              frame contains that behavior or not. This is the standard JABS mode and
              works well for non-exclusive behaviors.</li>

              <li><b>Multi-class:</b> Trains a single classifier across all annotated
              behaviors at once. The classifier assigns each frame to exactly one behavior
              class (or background). Multi-class mode is appropriate when behaviors are
              mutually exclusive.</li>
            </ul>

            <p><b>Note:</b> In multi-class mode, behaviors must be mutually exclusive.
            Frames labeled with more than one behavior simultaneously will cause a conflict and must be resolved before training.</p>
            """
        )
        help_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        return help_label

    def get_values(self) -> dict:
        """Get current classifier mode settings values.

        Returns:
            Dictionary with setting names and their current values.
        """
        return {CLASSIFIER_MODE_KEY: self._mode_combo.currentData()}

    def set_values(self, values: dict) -> None:
        """Set classifier mode settings values from a dictionary.

        Args:
            values: Dictionary with setting names and values to apply.
        """
        mode = values.get(CLASSIFIER_MODE_KEY, DEFAULT_CLASSIFIER_MODE)
        try:
            enum_val = ClassifierMode(mode)
            index = self._mode_combo.findData(enum_val)
        except ValueError:
            index = -1

        if index >= 0:
            self._mode_combo.setCurrentIndex(index)
        else:
            logger.warning(
                f"Invalid classifier mode value: {mode}. Defaulting to {DEFAULT_CLASSIFIER_MODE}."
            )
            self._mode_combo.setCurrentIndex(self._mode_combo.findData(DEFAULT_CLASSIFIER_MODE))
