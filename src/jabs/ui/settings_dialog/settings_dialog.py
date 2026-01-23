from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from jabs.constants import APP_NAME, ORG_NAME
from jabs.project.settings_manager import SettingsManager

from .cross_validation_settings_group import CrossValidationSettingsGroup
from .session_tracking_group import SessionTrackingSettingsGroup


class BaseSettingsDialog(QDialog):
    """Base dialog for editing settings.

    Subclasses should override `_create_settings_groups()` to add their settings groups.

    Args:
        settings_manager (SettingsManager): Settings manager used to load and save settings.
        title (str): Window title.
        parent (QWidget | None, optional): Parent widget for this dialog.
    """

    settings_changed = Signal()

    def __init__(
        self,
        settings_manager: SettingsManager | None,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._settings_manager = settings_manager

        # Allow resizing and show scrollbars if content overflows
        self.setSizeGripEnabled(True)

        # Scrollable page to host settings sections
        page = QWidget(self)
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        page_layout = QVBoxLayout(page)
        page_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)

        # Track all settings groups
        self._settings_groups: list = []

        # Let subclasses add settings groups
        self._create_settings_groups(page, page_layout)

        # Load current settings into groups
        self._load_settings()

        page_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidget(page)
        page.adjustSize()
        scroll.setWidgetResizable(False)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)

        # Keep references for width syncing
        self._scroll = scroll
        self._page = page

        # Initial width sync
        self._sync_page_width()

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

    def _create_settings_groups(self, parent: QWidget, layout: QVBoxLayout) -> None:
        """Create and add settings groups to the page.

        Subclasses must override this method and append created groups to `self._settings_groups`.
        """
        raise NotImplementedError

    def _sync_page_width(self) -> None:
        """Ensure the inner page matches the scroll viewport width exactly."""
        try:
            vp = self._scroll.viewport()
            if vp is not None:
                w = vp.width()
                # Set fixed width to match viewport; this allows content to reflow when window resizes
                # or when scrollbar appears/disappears
                self._page.setFixedWidth(w)
                self._page.updateGeometry()
        except Exception:
            pass

    def showEvent(self, event: QShowEvent) -> None:
        """Handle the show event.

        Ensures the settings page width is synchronized with the viewport when the dialog is first shown.

        Args:
            event (QShowEvent): The Qt show event.
        """
        super().showEvent(event)
        self._sync_page_width()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle the resize event.

        Ensures the settings page width matches the viewport width when the dialog is resized.

        Args:
            event (QResizeEvent): The Qt resize event.
        """
        super().resizeEvent(event)
        self._sync_page_width()

    def _load_settings(self) -> None:
        """Load current settings from the project into all settings groups.

        Raises:
            RuntimeError: If the settings manager is not set. If derived classes don't
              use a settings manager, they should override this method.
        """
        if self._settings_manager is None:
            raise RuntimeError("Settings manager is not set for this dialog.")

        all_project_data = self._settings_manager.project_settings
        current_settings = all_project_data.get("settings", {})
        for group in self._settings_groups:
            group.set_values(current_settings)

    def _on_save(self) -> None:
        """Save settings from all groups to project and close dialog.

        Raises:
            RuntimeError: If the settings manager is not set. If derived classes don't
              use a settings manager, they should override this method.
        """
        if self._settings_manager is None:
            raise RuntimeError("Settings manager is not set for this dialog.")
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


class ProjectSettingsDialog(BaseSettingsDialog):
    """Dialog for changing *project* settings (previously `SettingsDialog`)."""

    def __init__(self, settings_manager: SettingsManager, parent: QWidget | None = None) -> None:
        super().__init__(
            settings_manager=settings_manager, title="Project Settings", parent=parent
        )

    def _create_settings_groups(self, parent: QWidget, layout: QVBoxLayout) -> None:
        # Add settings groups here
        cv_group = CrossValidationSettingsGroup(parent)
        self._settings_groups.append(cv_group)
        layout.addWidget(cv_group)
        layout.setAlignment(cv_group, Qt.AlignmentFlag.AlignTop)


class JabsSettingsDialog(BaseSettingsDialog):
    """Dialog for changing *JABS application* settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._settings = QtCore.QSettings(ORG_NAME, APP_NAME)
        super().__init__(settings_manager=None, title="JABS Settings", parent=parent)

    def _create_settings_groups(self, parent: QWidget, layout: QVBoxLayout) -> None:
        """Create the settings groups for the dialog.

        Args:
            parent (QWidget): Parent widget for the settings groups.
            layout (QVBoxLayout): Layout to add the settings groups to.
        """
        session_tracking_settings = SessionTrackingSettingsGroup(parent)
        self._settings_groups.append(session_tracking_settings)
        layout.addWidget(session_tracking_settings)
        layout.setAlignment(session_tracking_settings, Qt.AlignmentFlag.AlignTop)

    def _load_settings(self) -> None:
        """Load current settings from QSettings into all settings groups."""
        current_settings = {}
        for key in self._settings.allKeys():
            current_settings[key] = self._settings.value(key)
        for group in self._settings_groups:
            group.set_values(current_settings)

    def _on_save(self) -> None:
        """Save settings from all groups to QSettings and close dialog."""
        all_settings = {}
        for group in self._settings_groups:
            all_settings.update(group.get_values())
        if all_settings:
            for key, value in all_settings.items():
                self._settings.setValue(key, value)
            self.settings_changed.emit()
        self.accept()
