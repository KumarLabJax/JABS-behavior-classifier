from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from jabs.core.constants import APP_NAME, ORG_NAME
from jabs.project.settings_manager import SettingsManager

from .cross_validation_settings_group import CrossValidationSettingsGroup
from .postprocessing_settings import InterpolationStageSettingsGroup
from .postprocessing_settings.postprocessing_group import (
    DurationStageSettingsGroup,
    StitchingStageSettingsGroup,
)
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
        page_layout.setContentsMargins(16, 0, 16, 0)
        page_layout.setSpacing(10)

        # Track all settings groups
        self._settings_groups: list = []

        # Optionally add a header widget above the first settings group
        header_widget = self._create_header_widget(page)
        if header_widget is not None:
            page_layout.addWidget(header_widget)

        # Let subclasses populate self._settings_groups, and add them to the layout
        self._create_settings_groups(page)
        self._add_groups_to_layout(page_layout)

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
        # self.resize(max(self.width(), 600), max(self.height(), 500))

    def _create_settings_groups(self, parent: QWidget) -> None:
        """Create and add settings groups to the page.

        Subclasses must override this method and append created groups to `self._settings_groups`.
        """
        raise NotImplementedError

    def _create_header_widget(self, parent: QWidget) -> QWidget | None:
        """Subclasses can override to add a widget above the first settings group.

        Args:
            parent (QWidget): Parent widget for the header.

        Returns:
            QWidget | None: The header widget, or None for no header.
        """
        return None

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

    def sizeHint(self) -> QtCore.QSize:
        """Provide a size hint for the dialog based on its content.

        Returns:
            QSize: The recommended size for the dialog.
        """
        base_hint = super().sizeHint()
        target_width = max(base_hint.width(), 720)
        target_height = max(base_hint.height(), 560)
        return QtCore.QSize(target_width, target_height)

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

    def _add_groups_to_layout(self, layout: QVBoxLayout) -> None:
        """Add all settings groups to the layout."""
        for idx, group in enumerate(self._settings_groups):
            layout.addWidget(group)
            layout.setAlignment(group, Qt.AlignmentFlag.AlignTop)
            # Add spacing after each group except the last
            if idx < len(self._settings_groups) - 1:
                layout.addSpacing(24)


class ProjectSettingsDialog(BaseSettingsDialog):
    """Dialog for changing *project* settings (previously `SettingsDialog`)."""

    def __init__(self, settings_manager: SettingsManager, parent: QWidget | None = None) -> None:
        super().__init__(
            settings_manager=settings_manager, title="Project Settings", parent=parent
        )

    def _create_settings_groups(self, parent: QWidget) -> None:
        """Add dialog specific settings groups, these will be added to the main layout by the base class.

        Args:
            parent (QWidget): Parent widget for the settings groups.
        """
        cv_group = CrossValidationSettingsGroup(parent)
        self._settings_groups.append(cv_group)


class PostprocessingSettingsDialog(BaseSettingsDialog):
    """Dialog for changing *post-processing* settings."""

    def __init__(
        self, settings_manager: SettingsManager, behavior: str, parent: QWidget | None = None
    ) -> None:
        self._behavior = behavior  # needs to be set before calling super().__init__() because _load_settings() uses it
        super().__init__(
            settings_manager=settings_manager,
            title=f"{behavior} Postprocessing Settings",
            parent=parent,
        )

    def _create_settings_groups(self, parent: QWidget) -> None:
        """Create the settings groups for the dialog."""
        self._settings_groups.append(InterpolationStageSettingsGroup(parent))
        self._settings_groups.append(StitchingStageSettingsGroup(parent))
        self._settings_groups.append(DurationStageSettingsGroup(parent))

    def _create_header_widget(self, parent: QWidget) -> QWidget | None:
        """Create a header widget for the dialog."""
        header_text = QLabel(self)
        header_text.setTextFormat(Qt.TextFormat.RichText)
        header_text.setWordWrap(True)
        header_text.setText(
            f"""
            <h4>Postprocessing Settings for Behavior: {self._behavior}</h4>

            <p>Prediction postprocessing modifies the raw behavior predictions made by the model to
            smooth predictions and and reduce noise. These settings allow you to configure which filters
            to apply.</p>
            
            <p><strong>Note:</strong> Postprocessing settings are behavior-specific. If enabled, filters
            are applied in the order they are listed below.</p>
            <br/>
            """
        )
        return header_text

    def _load_settings(self) -> None:
        """Unlike the ProjectSettingsDialog, load postprocessing settings from the behavior-specific settings."""
        # load the settings for the currently selected behavior
        behavior_settings = self._settings_manager.get_behavior(self._behavior)
        settings = behavior_settings.get("postprocessing", [])

        all_settings = {}

        for stage in settings:
            all_settings[stage["stage_name"]] = stage["config"]

        # we just pass all settings dict to each group, they will pick what they need based on stage name
        for group in self._settings_groups:
            group.set_values(all_settings)

    def _on_save(self) -> None:
        """Save postprocessing settings"""
        # Order matters, since it determines the order stages are applied.
        # To preserve order they are saved as a list of dicts, each dict representing a stage with its
        # config so that order will be maintained even though the project.json file is sorted by key.
        # transform from
        #   dict {stage_name: config, stage2_name: config, ...}
        # to
        #   list [{"stage_name": stage_name, "config": config}, ...]
        ordered_stages = []
        for group in self._settings_groups:
            ordered_stages.append(group.get_values())

        # save the postprocessing settings for the current behavior
        self._settings_manager.save_behavior(self._behavior, {"postprocessing": ordered_stages})

        self.accept()


class JabsSettingsDialog(BaseSettingsDialog):
    """Dialog for changing *JABS application* settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._settings = QtCore.QSettings(ORG_NAME, APP_NAME)
        super().__init__(settings_manager=None, title="JABS Settings", parent=parent)

    def _create_settings_groups(self, parent: QWidget) -> None:
        """Create the settings groups for the dialog.

        Args:
            parent (QWidget): Parent widget for the settings groups.
        """
        session_tracking_settings = SessionTrackingSettingsGroup(parent)
        self._settings_groups.append(session_tracking_settings)

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
