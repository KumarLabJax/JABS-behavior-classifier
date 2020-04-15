from PyQt5 import QtWidgets

from src.ui import PlayerWidget
from src.labeler import VideoLabels


class MainWindow(QtWidgets.QWidget):
    """
    QT Widget implementing our main window
    """

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # initial behavior labels to list in the drop down selection
        self._behaviors = [
            'Walking', 'Sleeping', 'Freezing', 'Grooming', 'Following',
            'Rearing (supported)', 'Rearing (unsupported)'
        ]

        # video player
        self._player_widget = PlayerWidget()
        self._player_widget.updateIdentities.connect(self._set_identities)

        self._tracks = None
        self._labels = None

        self._selection_start = 0

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

        # identity selection form components
        self.identity_selection = QtWidgets.QComboBox()
        self.identity_selection.currentIndexChanged.connect(
            self._change_identity)
        identity_layout = QtWidgets.QVBoxLayout()
        identity_layout.addWidget(self.identity_selection)
        identity_group = QtWidgets.QGroupBox("Identity")
        identity_group.setLayout(identity_layout)

        # label components
        label_layout = QtWidgets.QVBoxLayout()

        self.label_behavior_button = QtWidgets.QPushButton()
        self.label_behavior_button.setText(
            self.behavior_selection.currentText())
        self.label_behavior_button.clicked.connect(self._label_behavior)

        self.label_not_behavior_button = QtWidgets.QPushButton(
            f"Not {self.behavior_selection.currentText()}")
        self.label_not_behavior_button.clicked.connect(self._label_not_behavior)

        self.clear_label_button = QtWidgets.QPushButton("Clear Label")
        self.clear_label_button.clicked.connect(self._clear_behavior_label)

        self.select_button = QtWidgets.QPushButton("Select Frames")
        self.select_button.setCheckable(True)
        self.select_button.clicked.connect(self._start_selection)

        # label buttons are disabled unless user has a range of frames selected
        self._disable_label_buttons()

        label_layout.addWidget(self.label_behavior_button)
        label_layout.addWidget(self.label_not_behavior_button)
        label_layout.addWidget(self.clear_label_button)
        label_layout.addWidget(self.select_button)
        label_group = QtWidgets.QGroupBox("Label")
        label_group.setLayout(label_layout)

        # control layout
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setSpacing(25)
        control_layout.addWidget(behavior_group)
        control_layout.addWidget(identity_group)
        control_layout.addWidget(label_group)
        control_layout.addStretch()

        # main layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self._player_widget, 0, 0)
        layout.addLayout(control_layout, 0, 1)

        self.setLayout(layout)

    def load_video(self, path):
        """ load new avi file """
        self._player_widget.load_video(path)
        self._labels = VideoLabels(path, self._player_widget.num_frames())

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
        self.label_not_behavior_button.setText(
            f"Not {self.behavior_selection.currentText()}")

    def _start_selection(self, pressed):
        """
        handle click on "select" button. If button was previously in "unchecked"
        state, then grab the current frame to begin selecting a range. If the
        button was in the checked state, clicking cancels the current selection.

        While selection is in progress, the labeling buttons become active.
        """
        if pressed:
            self.label_behavior_button.setEnabled(True)
            self.label_not_behavior_button.setEnabled(True)
            self.clear_label_button.setEnabled(True)
            self._selection_start = self._player_widget.current_frame()
        else:
            self.label_behavior_button.setEnabled(False)
            self.label_not_behavior_button.setEnabled(False)
            self.clear_label_button.setEnabled(False)

    def _label_behavior(self):
        """ Apply behavior label to currently selected range of frames """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._labels.get_track_labels(
            self.identity_selection.currentText(),
            self.behavior_selection.currentText()
        ).label_behavior(*label_range)
        self._disable_label_buttons()

    def _label_not_behavior(self):
        """ apply _not_ behavior label to currently selected range of frames """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._labels.get_track_labels(
            self.identity_selection.currentText(),
            self.behavior_selection.currentText()
        ).label_not_behavior(*label_range)
        self._disable_label_buttons()

    def _clear_behavior_label(self):
        """ clear all behavior/not behavior labels from current selection """
        self._disable_label_buttons()

    def _set_identities(self, identities):
        """ populate the identity_selection combobox """
        self.identity_selection.clear()
        self.identity_selection.addItems([str(i) for i in identities])

    def _change_identity(self):
        """ handle changing value of identity_selection """
        self._player_widget._set_active_identity(
            self.identity_selection.currentIndex())

    def _disable_label_buttons(self):
        """ disable labeling buttons that require a selected range of frames """
        self.label_behavior_button.setEnabled(False)
        self.label_not_behavior_button.setEnabled(False)
        self.clear_label_button.setEnabled(False)
        self.select_button.setChecked(False)
