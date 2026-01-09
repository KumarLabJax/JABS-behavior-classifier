"""Custom message dialog for displaying errors, warnings, and information."""

from enum import Enum
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MessageType(Enum):
    """Enum for message dialog types."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class MessageDialog(QDialog):
    """Custom dialog for displaying messages with optional details.

    Supports error, warning, and informational messages. Can display
    an expandable details section for additional information like stack traces.
    """

    def __init__(
        self,
        message: str,
        title: str | None = None,
        details: str | None = None,
        message_type: MessageType = MessageType.ERROR,
        parent=None,
    ):
        """Initialize the message dialog.

        Args:
            message: The main message to display
            title: Dialog window title (default: determined by message_type)
            details: Optional detailed information (e.g., stack trace) that can be expanded
            message_type: Type of message (ERROR, WARNING, or INFO)
            parent: Parent widget
        """
        super().__init__(parent)

        # Set default title based on message type if not provided
        if title is None:
            title_map = {
                MessageType.ERROR: "Error",
                MessageType.WARNING: "Warning",
                MessageType.INFO: "Information",
            }
            title = title_map.get(message_type, "Message")

        self.setWindowTitle(title)
        self.setMinimumWidth(500)

        self._message_type = message_type
        self._details = details
        self._details_widget = None
        self._toggle_button = None

        # Main layout
        main_layout = QVBoxLayout()

        # Top section with icon and message
        top_layout = QHBoxLayout()

        # Icon
        icon_label = QLabel()
        icon_path, icon_tooltip = self._get_icon_path()
        if icon_path and icon_path.exists():
            pixmap = QPixmap(str(icon_path))
            # Scale the icon to a reasonable size
            scaled_pixmap = pixmap.scaled(
                128,
                128,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(scaled_pixmap)
            # Set tooltip if provided
            if icon_tooltip:
                icon_label.setToolTip(icon_tooltip)
        else:
            # Fallback to text emoji if no icon available
            icon_label.setText(self._get_fallback_icon())
            icon_label.setStyleSheet("font-size: 48pt;")

        top_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignTop)

        # Message layout
        message_layout = QVBoxLayout()

        # Title/heading
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        message_layout.addWidget(title_label)

        message_layout.addSpacing(10)

        # Main message
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextFormat(Qt.TextFormat.RichText)
        message_layout.addWidget(message_label)

        top_layout.addLayout(message_layout, 1)  # Stretch to fill available space

        main_layout.addLayout(top_layout)

        # Details section (collapsible)
        if details:
            main_layout.addSpacing(10)

            # Toggle button for details
            self._toggle_button = QPushButton("▶ Show Details")
            self._toggle_button.setFlat(True)
            self._toggle_button.setStyleSheet(
                """
                QPushButton {
                    text-align: left;
                    padding: 5px;
                    border: none;
                    background: transparent;
                }
                QPushButton:hover {
                    background: rgba(0, 0, 0, 0.05);
                }
                """
            )
            self._toggle_button.clicked.connect(self._toggle_details)
            main_layout.addWidget(self._toggle_button)

            # Details widget (initially hidden)
            self._details_widget = QWidget()
            details_layout = QVBoxLayout(self._details_widget)
            details_layout.setContentsMargins(0, 5, 0, 0)

            details_text = QTextEdit()
            details_text.setReadOnly(True)
            details_text.setPlainText(details)
            details_text.setMinimumHeight(150)
            details_text.setMaximumHeight(300)
            details_text.setStyleSheet(
                """
                QTextEdit {
                    font-family: 'Courier New', Courier, monospace;
                    font-size: 10pt;
                    border: 1px solid palette(mid);
                    border-radius: 4px;
                }
                """
            )
            details_layout.addWidget(details_text)

            main_layout.addWidget(self._details_widget)
            self._details_widget.setVisible(False)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setDefault(True)
        button_layout.addWidget(close_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _get_icon_path(self) -> tuple[Path | None, str | None]:
        """Get the path to the icon and optional tooltip based on message type.

        Returns:
            Tuple of (Path to icon file or None, tooltip text or None)
        """
        resources_dir = Path(__file__).parent.parent / "resources"

        # Map message types to (icon_file, tooltip) tuples
        # tooltip is optional - use None if no tooltip desired
        icon_map = {
            MessageType.ERROR: ("fail_whale.png", "Fail Whale"),
            # MessageType.WARNING: ("icon.png", "Warning message"),
            # MessageType.INFO: ("icon.png", None),  # No tooltip example
        }

        icon_info = icon_map.get(self._message_type)
        if icon_info:
            icon_file, tooltip = icon_info
            icon_path = resources_dir / icon_file
            if icon_path.exists():
                return icon_path, tooltip

        return None, None

    def _get_fallback_icon(self) -> str:
        """Get fallback emoji icon if image not available.

        Returns:
            Emoji string for the message type
        """
        fallback_map = {
            MessageType.ERROR: "❌",
            MessageType.WARNING: "⚠️",
            MessageType.INFO: "ℹ️",  # noqa: RUF001
        }
        return fallback_map.get(self._message_type, "❌")

    def _toggle_details(self):
        """Toggle the visibility of the details section."""
        if self._details_widget is None:
            return

        is_visible = self._details_widget.isVisible()
        self._details_widget.setVisible(not is_visible)

        # Update button text
        if self._toggle_button:
            if is_visible:
                self._toggle_button.setText("▶ Show Details")
            else:
                self._toggle_button.setText("▼ Hide Details")

        # Adjust dialog size
        self.adjustSize()

    @classmethod
    def error(
        cls,
        parent,
        title: str | None = None,
        message: str = "",
        details: str | None = None,
    ) -> int:
        """Show an error dialog (Qt-style API).

        Args:
            parent: Parent widget
            title: Dialog window title (default: "Error")
            message: The error message to display
            details: Optional detailed error information (e.g., stack trace)

        Returns:
            Dialog result code (QDialog.DialogCode.Accepted or Rejected)
        """
        dialog = cls(
            message=message,
            title=title,
            details=details,
            message_type=MessageType.ERROR,
            parent=parent,
        )
        return dialog.exec()

    @classmethod
    def warning(
        cls,
        parent,
        title: str | None = None,
        message: str = "",
        details: str | None = None,
    ) -> int:
        """Show a warning dialog (Qt-style API).

        Args:
            parent: Parent widget
            title: Dialog window title (default: "Warning")
            message: The warning message to display
            details: Optional detailed warning information

        Returns:
            Dialog result code (QDialog.DialogCode.Accepted or Rejected)
        """
        dialog = cls(
            message=message,
            title=title,
            details=details,
            message_type=MessageType.WARNING,
            parent=parent,
        )
        return dialog.exec()

    @classmethod
    def information(
        cls,
        parent,
        title: str | None = None,
        message: str = "",
        details: str | None = None,
    ) -> int:
        """Show an informational dialog (Qt-style API).

        Args:
            parent: Parent widget
            title: Dialog window title (default: "Information")
            message: The information message to display
            details: Optional detailed information

        Returns:
            Dialog result code (QDialog.DialogCode.Accepted or Rejected)
        """
        dialog = cls(
            message=message,
            title=title,
            details=details,
            message_type=MessageType.INFO,
            parent=parent,
        )
        return dialog.exec()


def main():
    """Test the MessageDialog functionality.

    Note: imports that are not needed for the module itself are placed here
    to avoid unnecessary dependencies when the module is imported elsewhere.
    """
    import sys
    import traceback

    from PySide6.QtWidgets import QApplication

    class TestWindow(QWidget):
        """Test window to demonstrate MessageDialog."""

        def __init__(self):
            super().__init__()
            self.setWindowTitle("MessageDialog Test")
            self.setMinimumWidth(300)

            layout = QVBoxLayout()

            # Error dialog button
            error_button = QPushButton("Show Error Dialog")
            error_button.clicked.connect(self.show_error)
            layout.addWidget(error_button)

            # Error with details button
            error_details_button = QPushButton("Show Error with Details")
            error_details_button.clicked.connect(self.show_error_with_details)
            layout.addWidget(error_details_button)

            # Warning dialog button
            warning_button = QPushButton("Show Warning Dialog")
            warning_button.clicked.connect(self.show_warning)
            layout.addWidget(warning_button)

            # Info dialog button
            info_button = QPushButton("Show Info Dialog")
            info_button.clicked.connect(self.show_info)
            layout.addWidget(info_button)

            self.setLayout(layout)

        def show_error(self):
            """Show a simple error dialog."""
            MessageDialog.error(
                self,
                title="Processing Error",
                message="An error occurred while processing your request.",
            )

        def show_error_with_details(self):
            """Show an error dialog with expandable details."""
            try:
                # Simulate an error
                raise ValueError("Invalid value provided")
            except ValueError:
                # Capture the stack trace
                stack_trace = traceback.format_exc()

                MessageDialog.error(
                    self,
                    title="Project Load Error",
                    message="Failed to load the project file. The file may be corrupted or in an unsupported format.",
                    details=stack_trace,
                )

        def show_warning(self):
            """Show a warning dialog."""
            MessageDialog.warning(
                self,
                title="New Window Size",
                message="The selected window size has not been used before. Features will be computed on the first training run, which may take some time.",
            )

        def show_info(self):
            """Show an info dialog."""
            MessageDialog.information(
                self,
                title="Project Loaded",
                message="JABS has successfully loaded your project. You can now begin labeling behaviors or train classifiers.",
            )

    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
