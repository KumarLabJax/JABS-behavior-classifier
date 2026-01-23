from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QSizePolicy

from ..constants import SESSION_TRACKING_ENABLED_KEY
from .settings_group import SettingsGroup


class SessionTrackingSettingsGroup(SettingsGroup):
    """Settings group for JABS Session Tracking."""

    def __init__(self, parent=None):
        """Initialize the session tracking settings group.

        Args:
            parent (QWidget | None): Parent widget for this settings group.
        """
        super().__init__("Cross-Validation", parent)

    def _create_controls(self) -> None:
        """Create the settings controls."""
        self._session_tracking_enabled = QCheckBox(self)
        self._session_tracking_enabled.setToolTip("Enable Session Tracking")
        self.add_control_row("Enable Session Tracking:", self._session_tracking_enabled)

    def _create_documentation(self):
        """Create help documentation for session tracking settings."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>What is Session Tracking?</h3>
            <p>JABS Session Tracking keeps a detailed log of labeling activity. Logged activity includes:</p>
            
            <ul>
                <li>When the project is opened or closed</li>
                <li>When videos are opened and closed</li>
                <li>When labels are added or deleted</li>
                <li>When a classifier is trained</li>
                <li>When the JABS application loses or regains focus</li>
            </ul>
            
            <p>Session logs are saved in the jabs/session directory inside the JABS project.</p>
            
            <p>Note: JABS session tracking is not enabled by default, and is not commonly used. It is primarily intended for usage analysis purposes.</p>
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
        return {
            SESSION_TRACKING_ENABLED_KEY: self._session_tracking_enabled.isChecked(),
        }

    def set_values(self, values: dict) -> None:
        """Load setting values from dictionary.

        Args:
            values: Dictionary with setting names and values to apply.
        """
        self._session_tracking_enabled.setChecked(values.get(SESSION_TRACKING_ENABLED_KEY, False))
