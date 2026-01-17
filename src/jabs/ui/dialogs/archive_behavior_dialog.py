from PySide6 import QtCore
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
)


class ArchiveBehaviorDialog(QDialog):
    """dialog to allow a user to select a behavior to archive from the project"""

    behavior_archived = QtCore.Signal(str)

    def __init__(self, behaviors: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._behavior_selection = QComboBox()
        self._behavior_selection.addItems(behaviors)
        self._behavior_selection.currentIndexChanged.connect(self.__behavior_selection_changed)

        self._confirm = QCheckBox("Confirm", self)
        self._confirm.setChecked(False)
        self._confirm.stateChanged.connect(self.__confirm_checkbox_changed)

        self._archive_button = QPushButton("Archive")
        self._archive_button.setEnabled(False)
        self._archive_button.clicked.connect(self.__archive)
        cancel_button = QPushButton("Close")
        cancel_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self._archive_button)

        layout = QVBoxLayout()
        layout.addWidget(self._behavior_selection)
        layout.addWidget(self._confirm)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def __confirm_checkbox_changed(self, state: bool):
        self._archive_button.setEnabled(state)

    def __behavior_selection_changed(self):
        self._confirm.setChecked(False)

    def __archive(self):
        # remove selected behavior from combo box
        behavior = self._behavior_selection.currentText()
        self.__remove_behavior(behavior)

        # if there are no other behaviors that can be archived then hide the dialog
        if self._behavior_selection.count() == 0:
            self.hide()

        # emit the signal to handle archiving the behavior
        self.behavior_archived.emit(behavior)

        # after emitting the signal, we can close the dialog if there are no more behaviors in the drop-down
        if self._behavior_selection.count() == 0:
            self.done(1)

    def __remove_behavior(self, behavior: str):
        idx = self._behavior_selection.findText(behavior, QtCore.Qt.MatchExactly)
        if idx != -1:
            self._behavior_selection.removeItem(idx)
