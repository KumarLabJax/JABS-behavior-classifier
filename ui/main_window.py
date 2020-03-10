from PyQt5 import QtGui, QtCore, QtWidgets
import darkdetect

from ui import PlayerWidget


class MainWindow(QtWidgets.QWidget):
    """
    QT Widget implementing our main window
    """

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # on OS X dark mode some of the colors need to be adjusted to
        # accommodate the dark window background color
        if darkdetect.theme() == 'Dark':
            self.setStyleSheet("""
                QLabel { color : #D3D3D3; }
                QGroupBox { background-color: #6C6C6C; color : #D3D3D3;}
                QToolButton { background: #6C6C6C; }
            """)

        # initial behavior labels to list in the drop down selection
        self._behaviors = [
            'Walking', 'Sleeping', 'Freezing', 'Grooming', 'Following',
            'Rearing (supported)', 'Rearing (unsupported)'
        ]

        # video player
        self.player_widget = PlayerWidget()

        # behavior selection form components
        self.behavior_selection = QtWidgets.QComboBox()
        self.behavior_selection.addItems(self._behaviors)
        self.behavior_selection.currentIndexChanged.connect(
            self.change_behavior)

        add_label_button = QtWidgets.QPushButton("New Behavior")
        add_label_button.clicked.connect(self.new_label)

        behavior_layout = QtWidgets.QVBoxLayout()
        behavior_layout.addWidget(self.behavior_selection)
        behavior_layout.addWidget(add_label_button)

        behavior_group = QtWidgets.QGroupBox("Behavior")
        behavior_group.setLayout(behavior_layout)

        # label components
        label_layout = QtWidgets.QVBoxLayout()

        self.label_behavior_button = QtWidgets.QPushButton()
        self.label_behavior_button.setText(
            self.behavior_selection.currentText())
        self.label_none_button = QtWidgets.QPushButton(
            f"Not {self.behavior_selection.currentText()}")
        label_unknown_button = QtWidgets.QPushButton("Clear Label")

        label_layout.addWidget(self.label_behavior_button)
        label_layout.addWidget(self.label_none_button)
        label_layout.addWidget(label_unknown_button)
        label_group = QtWidgets.QGroupBox("Label")
        label_group.setLayout(label_layout)

        # select mode components
        select_layout = QtWidgets.QGridLayout()

        select_button = QtWidgets.QPushButton("Select")
        select_button.setCheckable(True)

        selection_clear_button = QtWidgets.QPushButton("Clear")
        selection_play_button = QtWidgets.QPushButton("Play")

        select_layout.addWidget(select_button, 0, 0, 1, 2)
        select_layout.addWidget(selection_clear_button, 1, 0)
        select_layout.addWidget(selection_play_button, 1, 1)

        select_group = QtWidgets.QGroupBox("Selection")
        select_group.setLayout(select_layout)

        # control layout
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setSpacing(25)
        control_layout.addWidget(behavior_group)
        control_layout.addWidget(label_group)
        control_layout.addWidget(select_group)
        control_layout.addStretch()

        # main layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.player_widget, 0, 0)
        layout.addLayout(control_layout, 0, 1)

        self.setLayout(layout)

    def new_label(self):
        """
        callback for the "new behavior" button
        opens a modal dialog to allow the user to enter a new behavior label
        """
        text, ok = QtWidgets.QInputDialog.getText(self, 'New Label',
                                        'New Label Name:')
        if ok and text not in self._behaviors:
            self._behaviors.append(text)
            self.behavior_selection.addItem(text)

    def change_behavior(self):
        """
        make UI changes to reflect the currently selected behavior
        """
        self.label_behavior_button.setText(
            self.behavior_selection.currentText())
        self.label_none_button.setText(
            f"Not {self.behavior_selection.currentText()}")
