import enum

from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
)

from jabs.project import Project, get_videos_to_prune
from jabs.project.project_pruning import VideoPaths


class ProjectPruningDialog(QDialog):
    """Dialog for selecting project pruning options.

    Allows the user to choose whether to remove videos with no labels for any behavior,
    or only for a specific behavior. After the user selects options and clicks "Next",
    it shows a confirmation UI listing the videos that will be pruned from the project
    and allows the user to either confirm or cancel the operation.

    The dialog does not perform the actual pruning; it only collects the videos to be
    pruned and provides a confirmation step before proceeding. The actual pruning
    operation should be handled separately, typically after the user confirms the
    selection in the confirmation UI.

    The property `videos_to_prune` can be used to access the list of videos that will
    be pruned from the project after the user has closed the dialog with an "accept"
    action.
    """

    class PruningMode(enum.IntEnum):
        """Enum to represent the pruning mode."""

        ANY = enum.auto()
        SPECIFIC = enum.auto()

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Project Pruning")
        self._behaviors = project.settings_manager.behavior_names
        self._project = project
        self._layout = QVBoxLayout(self)
        self._init_pruning_options_ui()
        self._videos_to_prune: list[VideoPaths] | None = None

    @property
    def videos_to_prune(self) -> list[VideoPaths] | None:
        """Get the set of videos that will be pruned from the project."""
        return self._videos_to_prune

    def _init_pruning_options_ui(self) -> None:
        """Set up the initial UI for pruning options.

        The initial dialog layout includes a description of the pruning
        functionality,  radio buttons for selecting the pruning mode,
        a dropdown for selecting a specific behavior (if applicable),
        and a layout for the Cancel/Next buttons. The dialog allows
        users to choose whether to remove videos with no labels for any
        behavior or only for a specific behavior.
        """
        description = QLabel(
            "Project pruning allows you to remove videos from your project that do not contain any labels. "
            "You can choose to remove videos with no labels for any behavior, or only for a specific behavior."
        )
        description.setWordWrap(True)
        self._layout.addWidget(description)
        self._layout.addSpacerItem(
            QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        # Radio buttons
        radio_group = QButtonGroup(self)
        self._radio_any = QRadioButton("Remove videos with no labels for any behavior")
        self._radio_specific = QRadioButton("Remove videos with no labels for specified behavior")
        radio_group.addButton(self._radio_any)
        radio_group.addButton(self._radio_specific)
        self._radio_any.setChecked(True)

        self._layout.addWidget(self._radio_any)
        self._layout.addWidget(self._radio_specific)

        # Behavior dropdown
        behavior_layout = QHBoxLayout()
        self._behavior_label = QLabel("Behavior:")
        self._behavior_combo = QComboBox()
        self._behavior_combo.addItems(self._behaviors)
        self._behavior_combo.setEnabled(False)
        behavior_layout.insertSpacerItem(
            0, QSpacerItem(20, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        )
        behavior_layout.addWidget(self._behavior_label)
        behavior_layout.addWidget(self._behavior_combo)
        self._layout.addLayout(behavior_layout)

        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        ok_button = QPushButton("Next")
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        self._layout.addLayout(button_layout)

        # Connections
        self._radio_any.toggled.connect(self._update_behavior_combo)
        ok_button.clicked.connect(self._show_confirmation_ui)
        cancel_button.clicked.connect(self.reject)

    def _show_confirmation_ui(self) -> None:
        """Show the confirmation UI after the user has selected pruning options and clicked "Next".

        This method retrieves the videos to be pruned based on the selected mode (any or specific behavior),
        clears the current layout, and displays a confirmation message with the list of videos to be removed
        from the project. If no videos are found to prune, it displays a message indicating that
        there are no videos to prune based on the user's selection. The user can then confirm or
        cancel the pruning operation.
        """
        mode = self.PruningMode.ANY if self._radio_any.isChecked() else self.PruningMode.SPECIFIC

        if mode == self.PruningMode.ANY:
            self._videos_to_prune = get_videos_to_prune(self._project)
        else:
            behavior = self._behavior_combo.currentText()
            self._videos_to_prune = get_videos_to_prune(self._project, behavior)

        self._clear_layout()

        if not self._videos_to_prune:
            # No videos to prune
            msg_label = QLabel("There are no videos to prune based on your selection.")
            self._layout.addWidget(msg_label)
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(self.accept)
            button_layout = QHBoxLayout()
            button_layout.addWidget(ok_button)
            self._layout.addLayout(button_layout)
            return

        # Add confirmation message
        confirm_label = QLabel(
            f"The following {len(self._videos_to_prune)} videos will be removed from the project:"
        )
        self._layout.addWidget(confirm_label)

        # Add list of videos
        video_list = QListWidget()
        for video in self._videos_to_prune:
            QListWidgetItem(video.video_path.name, video_list)
        self._layout.addWidget(video_list)

        # Add OK/Cancel buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        ok_button = QPushButton("OK")
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        self._layout.addLayout(button_layout)

        # Connect buttons
        cancel_button.clicked.connect(self.reject)
        ok_button.clicked.connect(self.accept)

    def _update_behavior_combo(self) -> None:
        """Enable or disable the behavior combo box based on the selected radio button."""
        enabled = self._radio_specific.isChecked()
        self._behavior_combo.setEnabled(enabled)
        self._behavior_label.setEnabled(enabled)

    def _clear_layout(self) -> None:
        """Remove all widgets from the main layout"""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                # Remove nested layouts
                sub_layout = item.layout()
                if sub_layout is not None:
                    while sub_layout.count():
                        sub_item = sub_layout.takeAt(0)
                        sub_widget = sub_item.widget()
                        if sub_widget is not None:
                            sub_widget.deleteLater()

    def reject(self) -> None:
        """Override reject to clear the videos to prune when the dialog is cancelled."""
        self._videos_to_prune = None
        super().reject()
