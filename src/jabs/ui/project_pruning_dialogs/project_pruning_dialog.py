import enum

from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
)

from jabs.project import Project


class ProjectPruningDialog(QDialog):
    """
    Dialog for selecting project pruning options.

    Allows the user to choose whether to remove videos with no labels for any behavior,
    or only for a specific behavior, before proceeding with the pruning operation.
    Presents radio buttons for mode selection, a dropdown for behavior selection,
    and OK/Cancel buttons to continue or abort.
    """

    class PruningMode(enum.IntEnum):
        """Enum to represent the pruning mode."""

        ANY = enum.auto()
        SPECIFIC = enum.auto()

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Project Pruning")

        layout = QVBoxLayout(self)

        description = QLabel(
            "Project pruning allows you to remove videos from your project that do not contain any labels. "
            "You can choose to remove videos with no labels for any behavior, or only for a specific behavior."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addSpacerItem(
            QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        # Radio buttons
        self._radio_group = QButtonGroup(self)
        self._radio_any = QRadioButton("Remove videos with no labels for any behavior")
        self._radio_specific = QRadioButton("Remove videos with no labels for specified behavior")
        self._radio_group.addButton(self._radio_any)
        self._radio_group.addButton(self._radio_specific)
        self._radio_any.setChecked(True)

        layout.addWidget(self._radio_any)
        layout.addWidget(self._radio_specific)

        # Behavior dropdown
        behavior_layout = QHBoxLayout()
        self.behavior_label = QLabel("Behavior:")
        self.behavior_combo = QComboBox()
        self.behavior_combo.addItems(project.settings_manager.behaviors)
        self.behavior_combo.setEnabled(False)
        behavior_layout.insertSpacerItem(
            0, QSpacerItem(20, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        )
        behavior_layout.addWidget(self.behavior_label)
        behavior_layout.addWidget(self.behavior_combo)
        layout.addLayout(behavior_layout)

        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.ok_button = QPushButton("Next")
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        layout.addLayout(button_layout)

        # Connections
        self._radio_any.toggled.connect(self._update_behavior_combo)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def _update_behavior_combo(self, checked):
        self.behavior_combo.setEnabled(self._radio_specific.isChecked())
        self.behavior_label.setEnabled(self._radio_specific.isChecked())

    def get_selection(self) -> tuple[PruningMode, str | None]:
        """Get the user's selected pruning mode and behavior."""
        if self._radio_any.isChecked():
            return self.PruningMode.ANY, None
        else:
            return self.PruningMode.SPECIFIC, self.behavior_combo.currentText()
