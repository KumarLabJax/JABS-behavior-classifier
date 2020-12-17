import sys

from PySide2 import QtWidgets, QtCore

from src.classifier import Classifier

from .colors import BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR
from .identity_combo_box import IdentityComboBox
from .k_fold_slider_widget import KFoldSliderWidget
from .label_count_widget import FrameLabelCountWidget


class MainControlWidget(QtWidgets.QWidget):

    _DEFAULT_BEHAVIORS = [
        'Walking', 'Turn left', 'Turn right', 'Sleeping', 'Freezing',
        'Grooming', 'Following', 'Rearing (supported)',
        'Rearing (unsupported)'
    ]

    label_behavior_clicked = QtCore.Signal()
    label_not_behavior_clicked = QtCore.Signal()
    clear_label_clicked = QtCore.Signal()
    start_selection = QtCore.Signal(bool)
    identity_changed = QtCore.Signal()
    train_clicked = QtCore.Signal()
    classify_clicked = QtCore.Signal()
    classifier_changed = QtCore.Signal()
    behavior_changed = QtCore.Signal(str)
    kfold_changed = QtCore.Signal()
    behavior_list_changed = QtCore.Signal(list)
    window_size_changed = QtCore.Signal(int)
    new_window_sizes = QtCore.Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initial behavior labels to list in the drop down selection
        self._behaviors = list(self._DEFAULT_BEHAVIORS)
        self._behaviors.sort()

        # behavior selection form components
        self.behavior_selection = QtWidgets.QComboBox()
        self.behavior_selection.addItems(self._behaviors)
        self.behavior_selection.currentIndexChanged.connect(
            self._behavior_changed)

        self.identity_selection = IdentityComboBox()
        self.identity_selection.currentIndexChanged.connect(
            self.identity_changed)
        self.identity_selection.setEditable(False)
        self.identity_selection.installEventFilter(self.identity_selection)

        add_label_button = QtWidgets.QToolButton()
        add_label_button.setText("+")
        add_label_button.setToolTip("Add a new behavior label")
        add_label_button.clicked.connect(self._new_label)

        behavior_layout = QtWidgets.QHBoxLayout()
        behavior_layout.addWidget(self.behavior_selection)
        behavior_layout.addWidget(add_label_button)
        behavior_layout.setContentsMargins(5, 5, 5, 5)

        behavior_group = QtWidgets.QGroupBox("Behavior")
        behavior_group.setLayout(behavior_layout)

        # identity selection form components

        identity_layout = QtWidgets.QVBoxLayout()
        identity_layout.addWidget(self.identity_selection)
        identity_layout.setContentsMargins(5, 5, 5, 5)
        identity_group = QtWidgets.QGroupBox("Subject Identity")
        identity_group.setLayout(identity_layout)

        # classifier controls
        #  buttons
        self._train_button = QtWidgets.QPushButton("Train")
        self._train_button.clicked.connect(self.train_clicked)
        self._train_button.setEnabled(False)
        self._classify_button = QtWidgets.QPushButton("Classify")
        self._classify_button.clicked.connect(self.classify_clicked)
        self._classify_button.setEnabled(False)

        # drop down to select which window size to use
        self._window_size = QtWidgets.QComboBox()
        self._window_size.currentIndexChanged.connect(
            self._window_size_changed
        )
        self._window_size.setToolTip(
            "Number of frames before and after current frame to include in "
            "sliding window used to compute window features.\n"
            "The total number of frames included in the sliding window is two "
            "times the value of this parameter plus one."
        )

        add_window_size_button = QtWidgets.QToolButton()
        add_window_size_button.setText("+")
        add_window_size_button.setToolTip("Add a new window size")
        add_window_size_button.clicked.connect(self._new_window_size)

        window_size_layout = QtWidgets.QHBoxLayout()
        window_size_layout.addWidget(self._window_size)
        window_size_layout.addWidget(add_window_size_button)

        #  drop down to select type of classifier to use
        self._classifier_selection = QtWidgets.QComboBox()
        self._classifier_selection.currentIndexChanged.connect(
            self.classifier_changed)

        classifier_types = Classifier().classifier_choices()
        for classifier, name in classifier_types.items():
            self._classifier_selection.addItem(name, userData=classifier)

        #  slider to set number of times to train/test
        self._kslider = KFoldSliderWidget()
        self._kslider.valueChanged.connect(self.kfold_changed)
        #   disabled until project loaded
        self._kslider.setEnabled(False)

        #  classifier control layout
        classifier_layout = QtWidgets.QGridLayout()
        classifier_layout.addWidget(self._train_button, 0, 0)
        classifier_layout.addWidget(self._classify_button, 0, 1)
        classifier_layout.addWidget(self._classifier_selection, 1, 0, 1, 2)
        classifier_layout.addWidget(QtWidgets.QLabel("Window Size"), 2, 0)
        classifier_layout.addLayout(window_size_layout, 2, 1)
        classifier_layout.addWidget(self._kslider, 3, 0, 1, 2)
        classifier_layout.setContentsMargins(5, 5, 5, 5)
        classifier_group = QtWidgets.QGroupBox("Classifier")
        classifier_group.setLayout(classifier_layout)

        # label components
        label_layout = QtWidgets.QGridLayout()

        self._label_behavior_button = QtWidgets.QPushButton()
        self._label_behavior_button.setText(
            self.behavior_selection.currentText())
        self._label_behavior_button.clicked.connect(self.label_behavior_clicked)
        self._label_behavior_button.setStyleSheet(f"""
                    QPushButton {{
                        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 rgb(255, 195, 77),
                                           stop: 1.0 rgb{BEHAVIOR_COLOR});
                        border-radius: 4px;
                        padding: 2px;
                        color: white;
                    }}
                    QPushButton:pressed {{
                        background-color: rgb(255, 195, 77);
                    }}
                    QPushButton:disabled {{
                        background-color: rgb(229, 143, 0);
                        color: grey;
                    }}
                """)

        self._label_not_behavior_button = QtWidgets.QPushButton(
            f"Not {self.behavior_selection.currentText()}")
        self._label_not_behavior_button.clicked.connect(self.label_not_behavior_clicked)
        self._label_not_behavior_button.setStyleSheet(f"""
                    QPushButton {{
                        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 rgb(50, 119, 234),
                                           stop: 1.0 rgb{NOT_BEHAVIOR_COLOR});
                        border-radius: 4px;
                        padding: 2px;
                        color: white;
                    }}
                    QPushButton:pressed {{
                        background-color: rgb(50, 119, 234);
                    }}
                    QPushButton:disabled {{
                        background-color: rgb(0, 77, 206);
                        color: grey;
                    }}
                """)

        self._clear_label_button = QtWidgets.QPushButton("Clear Label")
        self._clear_label_button.clicked.connect(self.clear_label_clicked)

        self._select_button = QtWidgets.QPushButton("Select Frames")
        self._select_button.setCheckable(True)
        self._select_button.clicked.connect(self.start_selection)
        # disabled until a project is loaded
        self._select_button.setEnabled(False)

        # label buttons are disabled unless user has a range of frames selected
        self.disable_label_buttons()

        label_layout.addWidget(self._label_behavior_button, 0, 0, 1, 2)
        label_layout.addWidget(self._label_not_behavior_button, 1, 0, 1, 2)
        label_layout.addWidget(self._clear_label_button, 2, 0)
        label_layout.addWidget(self._select_button, 2, 1)
        label_layout.setContentsMargins(5, 5, 5, 5)
        label_group = QtWidgets.QGroupBox("Labeling")
        label_group.setLayout(label_layout)

        # summary of number of frames / bouts for each class
        self._frame_counts = FrameLabelCountWidget()
        label_count_layout = QtWidgets.QVBoxLayout()
        label_count_layout.addWidget(self._frame_counts)
        label_count_group = QtWidgets.QGroupBox("Label Summary")
        label_count_group.setLayout(label_count_layout)

        # control layout
        control_layout = QtWidgets.QVBoxLayout()
        if sys.platform == 'darwin':
            control_layout.setSpacing(20)
        else:
            control_layout.setSpacing(10)
        control_layout.addWidget(behavior_group)
        control_layout.addWidget(identity_group)
        control_layout.addWidget(classifier_group)
        control_layout.addWidget(label_count_group)
        control_layout.addStretch()
        control_layout.addWidget(label_group)

        self.setLayout(control_layout)

    @property
    def current_behavior(self):
        return self.behavior_selection.currentText()

    @property
    def current_identity(self):
        return self.identity_selection.currentText()

    @property
    def current_identity_index(self):
        return self.identity_selection.currentIndex()

    @property
    def select_button_is_checked(self):
        return self._select_button.isChecked()

    @property
    def kfold_value(self):
        return self._kslider.value()

    @property
    def train_button_enabled(self):
        return self._train_button.isEnabled()

    @train_button_enabled.setter
    def train_button_enabled(self, enabled: bool):
        self._train_button.setEnabled(enabled)

    @property
    def classifier_type(self):
        return self._classifier_selection.currentData()

    def disable_label_buttons(self):
        """ disable labeling buttons that require a selected range of frames """
        self._label_behavior_button.setEnabled(False)
        self._label_not_behavior_button.setEnabled(False)
        self._clear_label_button.setEnabled(False)
        self._select_button.setChecked(False)

    def enable_label_buttons(self):
        self._label_behavior_button.setEnabled(True)
        self._label_not_behavior_button.setEnabled(True)
        self._clear_label_button.setEnabled(True)

    def set_classifier_selection(self, classifier_type):
        try:
            index = self._classifier_selection.findData(classifier_type)
            if index != -1:
                self._classifier_selection.setCurrentIndex(index)
        except KeyError:
            # unable to use the classifier
            pass

    def set_frame_counts(self, label_behavior_current,
                         label_not_behavior_current,
                         label_behavior_project,
                         label_not_behavior_project,
                         bout_behavior_current,
                         bout_not_behavior_current,
                         bout_behavior_project,
                         bout_not_behavior_project):
        self._frame_counts.set_counts(label_behavior_current,
                                      label_not_behavior_current,
                                      label_behavior_project,
                                      label_not_behavior_project,
                                      bout_behavior_current,
                                      bout_not_behavior_current,
                                      bout_behavior_project,
                                      bout_not_behavior_project)

    def classify_button_set_enabled(self, enabled: bool):
        self._classify_button.setEnabled(enabled)

    def select_button_set_enabled(self, enabled: bool):
        self._select_button.setEnabled(enabled)

    def select_button_set_checked(self, checked):
        self._select_button.setChecked(checked)

    def toggle_select_button(self):
        self._select_button.toggle()

    def kslider_set_enabled(self, enabled: bool):
        self._kslider.setEnabled(enabled)

    def set_identity_index(self, i: int):
        self.identity_selection.setCurrentIndex(i)

    def update_project_settings(self, project_settings: dict):
        """
        update controls from project settings
        :param project_settings: dict containing project settings
        :return: None
        """

        # update window sizes
        self._set_window_sizes(project_settings['window_sizes'])

        # update behaviors
        # reset list of behaviors, then add any from the metadata
        self._behaviors = list(self._DEFAULT_BEHAVIORS)
        self._behaviors.sort()

        # we don't need this even handler to be active while we set up the
        # project (otherwise it gets unnecessarily called multiple times)
        self.behavior_selection.currentIndexChanged.disconnect()

        behavior_index = 0
        if 'behaviors' in project_settings:
            # add behavior labels from project metadata that aren't already in
            # the app default list
            for b in project_settings['behaviors']:
                if b not in self._behaviors:
                    self._behaviors.append(b)
            self._behaviors.sort()
            self.behavior_selection.clear()
            self.behavior_selection.addItems(self._behaviors)
        if 'selected_behavior' in project_settings:
            # make sure this behavior is in the behavior selection drop down
            if project_settings['selected_behavior'] not in self._behaviors:
                self._behaviors.append(project_settings['selected_behavior'])
                self._behaviors.sort()
                self.behavior_selection.clear()
                self.behavior_selection.addItems(self._behaviors)
            behavior_index = self._behaviors.index(
                project_settings['selected_behavior'])

        # set the index to either the first behavior, or if available, the one
        # that was saved in the project metadata
        self.behavior_selection.setCurrentIndex(behavior_index)
        self._label_behavior_button.setText(self.current_behavior)
        self._label_not_behavior_button.setText(
            f"Not {self.current_behavior}")

        # use window size last
        window_settings = project_settings.get('window_size_pref', {})
        if self.current_behavior in window_settings:
            self.set_window_size(window_settings[self.current_behavior])

        # re-enable the behavior_selection change signal handler
        self.behavior_selection.currentIndexChanged.connect(
            self._behavior_changed)

    def set_identities(self, identities):
        """ populate the identity_selection combobox """
        self.identity_selection.currentIndexChanged.disconnect()
        self.identity_selection.clear()
        self.identity_selection.currentIndexChanged.connect(
            self.identity_changed)
        self.identity_selection.addItems([str(i) for i in identities])

    def _set_window_sizes(self, sizes: [int]):
        self._window_size.clear()
        for w in sizes:
            self._window_size.addItem(str(w), userData=w)

    def set_window_size(self, size: int):
        if self._window_size.findData(size) == -1:
            self._add_window_size(size)
        self._window_size.setCurrentText(str(size))

    def _new_label(self):
        """
        callback for the "new behavior" button
        opens a modal dialog to allow the user to enter a new behavior label,
        if user clicks ok, add that behavior to the combo box, and select it
        """
        text, ok = QtWidgets.QInputDialog.getText(None, 'New Behavior',
                                                  'New Behavior Name:')
        if ok and text not in self._behaviors:
            self._behaviors.append(text)
            self._behaviors.sort()
            self.behavior_selection.addItem(text)
            self.behavior_selection.setCurrentText(text)
            self.behavior_list_changed.emit(self._behaviors)

    def _new_window_size(self):
        """
        callback for the "new window size" button
        opens a modal dialog to allow the user to enter a new window size,
        if user clicks ok, add that window size and select it
        """
        val, ok = QtWidgets.QInputDialog.getInt(
            self, 'New Window Size', 'Enter a new window size:', min=1, max=20)
        if ok:
            if self._window_size.findData(val) == -1:
                self._add_window_size(val)
            self.set_window_size(val)
            QtWidgets.QMessageBox.warning(
                self, "Window Size Added",
                "Window Size Added.\n"
                "If features have not been computed for "
                "this window size, they will be computed the first time a "
                "classifier is trained using this window size.\n"
                "This may be slow.")

    def _add_window_size(self, new_size: int):
        # we clear and reset the contents of the combo box so that we
        # can re sort it with the new size

        # grab the old sizes, grabbing the data (int) instead of the
        # text
        sizes = [self._window_size.itemData(i) for i in
                 range(self._window_size.count())]

        # add our new value and sort
        sizes.append(new_size)
        sizes.sort()

        # clear and add in the new list of sizes
        self._window_size.clear()
        for s in sizes:
            self._window_size.addItem(str(s), userData=s)

        # send a signal that we have an updated list of window sizes
        self.new_window_sizes.emit(sizes)

    def _behavior_changed(self):
        self._label_behavior_button.setText(self.current_behavior)
        self._label_not_behavior_button.setText(
            f"Not {self.current_behavior}")
        self.behavior_changed.emit(self.current_behavior)

    def _window_size_changed(self):
        self.window_size_changed.emit(self._window_size.currentData())
