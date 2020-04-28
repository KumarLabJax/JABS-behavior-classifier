from PyQt5 import QtWidgets, QtCore

from src.ui import PlayerWidget, ManualLabelWidget, TimelineLabelWidget
from src.labeler import VideoLabels


class CentralWidget(QtWidgets.QWidget):
    """
    QT Widget implementing our main window contents
    """

    def __init__(self, *args, **kwargs):
        super(CentralWidget, self).__init__(*args, **kwargs)

        # initial behavior labels to list in the drop down selection
        self._behaviors = [
            'Walking', 'Sleeping', 'Freezing', 'Grooming', 'Following',
            'Rearing (supported)', 'Rearing (unsupported)'
        ]

        # video player
        self._player_widget = PlayerWidget()
        self._player_widget.updateIdentities.connect(self._set_identities)
        self._player_widget.updateFrameNumber.connect(self._frame_change)

        self._tracks = None
        self._labels = None

        self._selection_start = 0

        # options
        self._frame_jump = 10

        # behavior selection form components
        self.behavior_selection = QtWidgets.QComboBox()
        self.behavior_selection.addItems(self._behaviors)
        self.behavior_selection.currentIndexChanged.connect(
            self._change_behavior)

        add_label_button = QtWidgets.QPushButton("New Behavior")
        add_label_button.clicked.connect(self._new_label)

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
        self.label_behavior_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(128, 0, 0);
                border-radius: 4px;
                padding: 2px;
                color: white;
            }
            QPushButton:pressed {
                background-color: rgb(255, 0, 0);
            }
            QPushButton:disabled {
                background-color: rgb(64, 0, 0);
                color: grey;
            }
        """)

        self.label_not_behavior_button = QtWidgets.QPushButton(
            f"Not {self.behavior_selection.currentText()}")
        self.label_not_behavior_button.clicked.connect(self._label_not_behavior)
        self.label_not_behavior_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(0, 0, 128);
                border-radius: 4px;
                padding: 2px;
                color: white;
            }
            QPushButton:pressed {
                background-color: rgb(0, 0, 255);
            }
            QPushButton:disabled {
                background-color: rgb(0, 0, 64);
                color: grey;
            }
        """)

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
        control_layout.addStretch()
        control_layout.addWidget(label_group)

        # label widgets
        self.manual_labels = ManualLabelWidget()

        # timeline widget
        self.timeline_widget = TimelineLabelWidget()

        # main layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self._player_widget, 0, 0)
        layout.addLayout(control_layout, 0, 1)
        layout.addWidget(self.timeline_widget, 1, 0, 1, 2)
        layout.addWidget(self.manual_labels, 2, 0, 1, 2)

        self.setLayout(layout)

    def load_video(self, path):
        """ load new avi file """
        self._player_widget.load_video(path)
        self._labels = VideoLabels(path, self._player_widget.num_frames())
        self._set_label_track()
        self.manual_labels.set_num_frames(self._player_widget.num_frames())
        self.timeline_widget.set_num_frames(self._player_widget.num_frames())

    def keyPressEvent(self, event):
        """ handle key press events """
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            self._player_widget.previous_frame()
        elif key == QtCore.Qt.Key_Right:
            self._player_widget.next_frame()
        elif key == QtCore.Qt.Key_Up:
            self._player_widget.previous_frame(self._frame_jump)
        elif key == QtCore.Qt.Key_Down:
            self._player_widget.next_frame(self._frame_jump)
        elif key == QtCore.Qt.Key_Space:
            self.select_button.toggle()
            self._start_selection(self.select_button.isChecked())
        elif key == QtCore.Qt.Key_Z:
            if self.select_button.isChecked():
                self._label_behavior()
        elif key == QtCore.Qt.Key_X:
            if self.select_button.isChecked():
                self._clear_behavior_label()
        elif key == QtCore.Qt.Key_C:
            if self.select_button.isChecked():
                self._label_not_behavior()

    def _new_label(self):
        """
        callback for the "new behavior" button
        opens a modal dialog to allow the user to enter a new behavior label
        """
        text, ok = QtWidgets.QInputDialog.getText(self, 'New Label',
                                        'New Label Name:')
        if ok and text not in self._behaviors:
            self._behaviors.append(text)
            self.behavior_selection.addItem(text)

    def _change_behavior(self):
        """
        make UI changes to reflect the currently selected behavior
        """
        self.label_behavior_button.setText(
            self.behavior_selection.currentText())
        self.label_not_behavior_button.setText(
            f"Not {self.behavior_selection.currentText()}")
        self._set_label_track()

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
            self.manual_labels.start_selection(self._selection_start)
        else:
            self.label_behavior_button.setEnabled(False)
            self.label_not_behavior_button.setEnabled(False)
            self.clear_label_button.setEnabled(False)
            self.manual_labels.clear_selection()
        self.manual_labels.update()

    def _label_behavior(self):
        """ Apply behavior label to currently selected range of frames """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._get_label_track().label_behavior(*label_range)
        self._disable_label_buttons()
        self.manual_labels.clear_selection()
        self.manual_labels.update()
        self.timeline_widget.update_labels()

    def _label_not_behavior(self):
        """ apply _not_ behavior label to currently selected range of frames """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._get_label_track().label_not_behavior(*label_range)
        self._disable_label_buttons()
        self.manual_labels.clear_selection()
        self.manual_labels.update()
        self.timeline_widget.update_labels()

    def _clear_behavior_label(self):
        """ clear all behavior/not behavior labels from current selection """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._get_label_track().clear_labels(*label_range)
        self._disable_label_buttons()
        self.manual_labels.clear_selection()
        self.manual_labels.update()
        self.timeline_widget.update_labels()

    def _set_identities(self, identities):
        """ populate the identity_selection combobox """
        self.identity_selection.clear()
        self.identity_selection.addItems([str(i) for i in identities])

    def _change_identity(self):
        """ handle changing value of identity_selection """
        self._player_widget._set_active_identity(
            self.identity_selection.currentIndex())
        self._set_label_track()

    def _disable_label_buttons(self):
        """ disable labeling buttons that require a selected range of frames """
        self.label_behavior_button.setEnabled(False)
        self.label_not_behavior_button.setEnabled(False)
        self.clear_label_button.setEnabled(False)
        self.select_button.setChecked(False)

    def _frame_change(self, new_frame):
        """
        called when the video player widget emits its updateFrameNumber signal
        """
        self.manual_labels.set_current_frame(new_frame)
        self.timeline_widget.set_current_frame(new_frame)

    def _set_label_track(self):
        """
        loads new set of labels in self.manual_labels when the selected
        behavior or identity is changed
        """
        behavior = self.behavior_selection.currentText()
        identity = self.identity_selection.currentText()

        if identity != '' and behavior != '' and self._labels is not None:
            labels = self._labels.get_track_labels(identity, behavior)
            self.manual_labels.set_labels(labels)
            self.timeline_widget.set_labels(labels)

    def _get_label_track(self):
        """
        get the current label track for the currently selected identity and
        behavior
        """
        return self._labels.get_track_labels(
            self.identity_selection.currentText(),
            self.behavior_selection.currentText()
        )
