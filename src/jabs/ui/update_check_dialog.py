"""Dialog for displaying JABS update check results."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from jabs.utils import is_pypi_install


class UpdateCheckDialog(QDialog):
    """Dialog that displays update check results.

    Shows the current version, latest available version on PyPI, and whether
    an update is available.
    """

    def __init__(
        self,
        current_version: str,
        latest_version: str | None,
        has_update: bool,
        parent=None,
    ):
        """Initialize the update check dialog.

        Args:
            current_version: Currently installed version
            latest_version: Latest version available on PyPI (None if check failed)
            has_update: Whether a newer version is available
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Check for Updates")
        self.setMinimumWidth(400)

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # Use the JABS application icon
        icon_label = QLabel()
        icon_path = Path(__file__).parent.parent / "resources" / "icon.png"
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path))
            # Scale the icon to a reasonable size for display in the dialog
            scaled_pixmap = pixmap.scaled(
                64,
                64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(scaled_pixmap)
        top_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignTop)

        status_layout = QVBoxLayout()
        if latest_version is None:
            # Check failed
            title_label = QLabel("Update Check Failed")
            title_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
            status_layout.addWidget(title_label)

            message_label = QLabel("Could not check for updates. Please try again later.")
            status_layout.addWidget(message_label)

        elif has_update:
            # Update available
            title_label = QLabel("Update Available")
            title_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
            status_layout.addWidget(title_label)

            message_label = QLabel("A new version of JABS is available!")
            status_layout.addWidget(message_label)

        else:
            # Up to date
            title_label = QLabel("Up to Date")
            title_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
            status_layout.addWidget(title_label)

            message_label = QLabel("You have the latest version of JABS.")
            status_layout.addWidget(message_label)

        status_layout.addSpacing(10)

        # Version information
        version_info_layout = QVBoxLayout()
        current_label = QLabel(f"<b>Current version:</b> {current_version}")
        version_info_layout.addWidget(current_label)

        if latest_version:
            latest_label = QLabel(f"<b>Latest version:</b> {latest_version}")
            version_info_layout.addWidget(latest_label)

        status_layout.addLayout(version_info_layout)

        # Add installation instructions if update available and installed from PyPI
        if has_update and latest_version and is_pypi_install():
            status_layout.addSpacing(10)
            instructions_label = QLabel(
                "To update, run:<br>"
                "<span style=\"font-family: 'Courier New', Courier, monospace;\">pip install --upgrade jabs-behavior-classifier</span>"
            )
            instructions_label.setWordWrap(True)
            instructions_label.setStyleSheet("color: #666;")
            status_layout.addWidget(instructions_label)

        top_layout.addLayout(status_layout)
        main_layout.addLayout(top_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setDefault(True)
        button_layout.addWidget(close_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
