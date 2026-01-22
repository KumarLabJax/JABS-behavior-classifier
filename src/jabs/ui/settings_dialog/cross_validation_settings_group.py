"""Cross-validation settings group for configuring model training and validation."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLabel, QSizePolicy

from jabs.constants import CV_GROUPING_KEY
from jabs.types import DEFAULT_CV_GROUPING_STRATEGY, CrossValidationGroupingStrategy

from .settings_group import SettingsGroup


class CrossValidationSettingsGroup(SettingsGroup):
    """
    Settings group for cross-validation configuration.

    This group controls how data is split during model training and validation.
    """

    def __init__(self, parent=None):
        """Initialize the cross-validation settings group."""
        super().__init__("Cross-Validation", parent)

    def _create_controls(self) -> None:
        """Create the cross-validation settings controls."""
        # Cross-validation grouping combo box
        self._cv_grouping_combo = QComboBox()
        # Add enum values as items, storing the enum as userData
        for enum_val in CrossValidationGroupingStrategy:
            self._cv_grouping_combo.addItem(enum_val.value, enum_val)
        self._cv_grouping_combo.setCurrentIndex(
            self._cv_grouping_combo.findData(DEFAULT_CV_GROUPING_STRATEGY)
        )
        self._cv_grouping_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        self.add_control_row("CV Grouping:", self._cv_grouping_combo)

    def _create_documentation(self):
        """Create help documentation for cross-validation settings."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>What is Cross-Validation Grouping?</h3>
            <p>Cross-validation grouping determines how training data is split when 
            evaluating model performance using leave-one-group-out cross-validation.</p>
            
            <ul>
              <li><b>Individual Animal:</b> Each group represents a single animal identity 
              within a single video. During cross-validation, all labeled data for one 
              animal from one video is held out for validation while the remaining animals' 
              data is used for training.</li>
              
              <li><b>Video:</b> Each group represents a single video recording. During 
              cross-validation, all labeled data from one video is held out for validation 
              while data from other videos is used for training.</li>
            </ul>
            
            <p><b>Note:</b> For cross-validation to work properly, you need labeled data 
            from multiple groups (multiple animals or multiple videos, depending on the 
            grouping method selected). For rare behaviors, it may be easier to meet the 
            minimum label requirements per group at the video level rather than at the 
            individual animal level within a single video.</p>
            """
        )
        help_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        return help_label

    def get_values(self) -> dict:
        """
        Get current cross-validation settings values.

        Returns:
            Dictionary with setting names and their current values.
        """
        # Return the enum, not just the string
        return {
            CV_GROUPING_KEY: self._cv_grouping_combo.currentData(),
        }

    def set_values(self, values: dict) -> None:
        """
        Set cross-validation settings values from a dictionary.

        Args:
            values: Dictionary with setting names and values to apply.
        """
        cv_grouping = values.get(CV_GROUPING_KEY, CrossValidationGroupingStrategy.INDIVIDUAL)

        # The grouping setting is saved as the string value, try to convert string from settings dict back to enum
        try:
            enum_val = CrossValidationGroupingStrategy(cv_grouping)
            index = self._cv_grouping_combo.findData(enum_val)
        except ValueError:
            # Invalid setting value, we'll treat it as not found
            index = -1

        if index >= 0:
            self._cv_grouping_combo.setCurrentIndex(index)
        else:
            # Fall back to default if invalid value or not found
            self._cv_grouping_combo.setCurrentIndex(
                self._cv_grouping_combo.findData(CrossValidationGroupingStrategy.INDIVIDUAL)
            )
