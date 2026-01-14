from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from jabs.project.settings_manager import SettingsManager

from .cross_validation_settings_group import CrossValidationSettingsGroup


class SettingsDialog(QDialog):
    """
    Dialog for changing project settings.

    Args:
        settings_manager (SettingsManager): Project settings manager used to load and save settings.
        parent (QWidget | None, optional): Parent widget for this dialog. Defaults to None.
    """

    settings_changed = Signal()

    def __init__(self, settings_manager: SettingsManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project Settings")
        self._settings_manager = settings_manager

        # Allow resizing and show scrollbars if content overflows
        self.setSizeGripEnabled(True)

        # Scrollable page to host settings sections
        page = QWidget(self)
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        # Track all settings groups
        self._settings_groups: list = []

        # Add settings groups here
        cv_group = CrossValidationSettingsGroup(page)
        self._settings_groups.append(cv_group)
        page_layout.addWidget(cv_group)
        page_layout.setAlignment(cv_group, Qt.AlignmentFlag.AlignTop)

        # Load current settings into groups
        self._load_settings()

        page_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Keep references for width syncing
        self._scroll = scroll
        self._page = page

        # Buttons
        btn_box = QDialogButtonBox(self)
        btn_save = btn_box.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_close = btn_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        btn_save.clicked.connect(self._on_save)
        btn_close.clicked.connect(self.reject)

        # Main layout
        main = QVBoxLayout(self)
        main.addWidget(scroll, 1)
        main.addWidget(btn_box)

        self.setLayout(main)

        # Size to content initially
        self.adjustSize()
        self.resize(max(self.width(), 600), max(self.height(), 500))

    def _sync_page_width(self) -> None:
        """Ensure the inner page uses the full scroll viewport width.

        With setWidgetResizable(True), the scroll area automatically resizes the page widget
        to match the viewport width. This method is kept for compatibility but is mostly a no-op.
        """
        pass

    def showEvent(self, e: QShowEvent) -> None:
        """Handle the show event.

        Ensures the settings page width is synchronized with the viewport when the dialog is first shown.

        Args:
            e (QShowEvent): The Qt show event.
        """
        super().showEvent(e)
        self._sync_page_width()

    def resizeEvent(self, e: QResizeEvent) -> None:
        """Handle the resize event.

        Ensures the settings page width matches the viewport width when the dialog is resized.

        Args:
            e (QResizeEvent): The Qt resize event.
        """
        super().resizeEvent(e)
        self._sync_page_width()

    def _load_settings(self) -> None:
        """Load current settings from the project into all settings groups."""
        all_project_data = self._settings_manager.project_settings
        current_settings = all_project_data.get("settings", {})
        for group in self._settings_groups:
            group.set_values(current_settings)

    def _on_save(self) -> None:
        """Save settings from all groups to project and close dialog."""
        # Collect settings from all groups
        all_settings = {}
        for group in self._settings_groups:
            all_settings.update(group.get_values())

        # Save to project if there are any settings
        if all_settings:
            settings = {"settings": all_settings}
            self._settings_manager.save_project_file(settings)
            self.settings_changed.emit()

        self.accept()
