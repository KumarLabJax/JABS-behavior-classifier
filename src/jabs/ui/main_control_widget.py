"""Project-level controls for classifiers.

Todo:
    While this file was initially designed for controlling project settings
    This is now the primary location where project-level settings are managed
    The project class simply exposes its settings here which are modified
    The project class should be the management location of these features

Todo:
    convert many of the getter/setter methods to properties
"""

import sys

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QPainter, QPixmap

from jabs.classifier import Classifier
from jabs.ui.ear_tag_icons import EarTagIconManager

from .colors import (
    BEHAVIOR_BUTTON_COLOR_BRIGHT,
    BEHAVIOR_BUTTON_DISABLED_COLOR,
    BEHAVIOR_COLOR,
    NOT_BEHAVIOR_BUTTON_DISABLED_COLOR,
    NOT_BEHAVIOR_COLOR,
    NOT_BEHAVIOR_COLOR_BRIGHT,
)
from .k_fold_slider_widget import KFoldSliderWidget
from .label_count_widget import FrameLabelCountWidget


class MainControlWidget(QtWidgets.QWidget):
    """Controls for classifier training, labeling, and settings.

    Provides UI components and logic for managing behaviors, subject
    identities, classifier selection, window sizes, label assignment,
    and related project-level settings. Emits signals for user actions
    and updates, and synchronizes UI state with project metadata.

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.

    Signals:
        label_behavior_clicked (Signal): Emitted when the behavior label button is clicked.
        label_not_behavior_clicked (Signal): Emitted when the not-behavior label button is clicked.
        clear_label_clicked (Signal): Emitted when the clear label button is clicked.
        start_selection (Signal): Emitted when the select frames button is toggled.
        identity_changed (Signal): Emitted when the selected identity changes.
        train_clicked (Signal): Emitted when the train button is clicked.
        classify_clicked (Signal): Emitted when the classify button is clicked.
        classifier_changed (Signal): Emitted when the classifier selection changes.
        behavior_changed (Signal): Emitted when the selected behavior changes.
        kfold_changed (Signal): Emitted when the k-fold value changes.
        new_behavior_label (Signal): Emitted when a new behavior label is added.
        window_size_changed (Signal): Emitted when the window size changes.
        new_window_sizes (Signal): Emitted when the list of window sizes changes.
        use_balance_labels_changed (Signal): Emitted when the balance labels option changes.
        use_symmetric_changed (Signal): Emitted when the symmetric behavior option changes.
    """

    label_behavior_clicked = QtCore.Signal()
    label_not_behavior_clicked = QtCore.Signal()
    clear_label_clicked = QtCore.Signal()
    timeline_annotation_button_clicked = QtCore.Signal()
    start_selection = QtCore.Signal(bool)
    identity_changed = QtCore.Signal()
    train_clicked = QtCore.Signal()
    classify_clicked = QtCore.Signal()
    classifier_changed = QtCore.Signal()
    behavior_changed = QtCore.Signal(str)
    kfold_changed = QtCore.Signal()
    new_behavior_label = QtCore.Signal(list)
    window_size_changed = QtCore.Signal(int)
    new_window_sizes = QtCore.Signal(list)
    use_balance_labels_changed = QtCore.Signal(int)
    use_symmetric_changed = QtCore.Signal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initial behavior labels to list in the drop-down selection
        self._behaviors = []

        # behavior selection form components
        self.behavior_selection = QtWidgets.QComboBox()
        self.behavior_selection.addItems(self._behaviors)
        self.behavior_selection.currentIndexChanged.connect(self._behavior_changed)

        self.identity_selection = QtWidgets.QComboBox()
        self.identity_selection.currentIndexChanged.connect(self.identity_changed)
        self.identity_selection.setEditable(False)
        self.identity_selection.installEventFilter(self.identity_selection)

        self._ear_tag_icons = EarTagIconManager()
        self.identity_selection.setIconSize(QtCore.QSize(16, 16))

        self._add_label_button = QtWidgets.QToolButton()
        self._add_label_button.setText("+")
        self._add_label_button.setToolTip("Add a new behavior label")
        self._add_label_button.setEnabled(False)
        self._add_label_button.clicked.connect(self._new_label)

        # behavior selection form layout
        behavior_layout = QtWidgets.QHBoxLayout()
        behavior_layout.addWidget(self.behavior_selection)
        behavior_layout.addWidget(self._add_label_button)

        # identity selection form layout
        identity_layout = QtWidgets.QVBoxLayout()
        identity_layout.addWidget(self.identity_selection)

        # combine behavior and identity layouts into a single layout
        behavior_identity_layout = QtWidgets.QVBoxLayout()
        behavior_identity_layout.addLayout(behavior_layout)
        behavior_identity_layout.addLayout(identity_layout)
        behavior_identity_layout.setSpacing(2)
        behavior_identity_layout.setContentsMargins(5, 5, 5, 5)
        behavior_identity_group = QtWidgets.QGroupBox("Behavior && Subject")
        behavior_identity_group.setLayout(behavior_identity_layout)

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
        self._window_size.currentIndexChanged.connect(self._window_size_changed)
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
        self._classifier_selection.currentIndexChanged.connect(self.classifier_changed)

        classifier_types = Classifier().classifier_choices()
        for classifier, name in classifier_types.items():
            self._classifier_selection.addItem(name, userData=classifier)

        #  slider to set number of times to train/test
        self._kslider = KFoldSliderWidget()
        self._kslider.valueChanged.connect(self.kfold_changed)
        self._kslider.setEnabled(True)

        self._use_balace_labels_checkbox = QtWidgets.QCheckBox("Balance Training Labels")
        self._use_balace_labels_checkbox.stateChanged.connect(self.use_balance_labels_changed)

        self._symmetric_behavior_checkbox = QtWidgets.QCheckBox("Symmetric Behavior")
        self._symmetric_behavior_checkbox.stateChanged.connect(self.use_symmetric_changed)

        self._all_kfold_checkbox = QtWidgets.QCheckBox("All k-fold Cross Validation")
        self._all_kfold_checkbox.stateChanged.connect(self._all_kfold_changed)

        #  classifier control layout
        classifier_layout = QtWidgets.QGridLayout()
        classifier_layout.addWidget(self._train_button, 0, 0)
        classifier_layout.addWidget(self._classify_button, 0, 1)
        classifier_layout.addWidget(self._classifier_selection, 1, 0, 1, 2)
        classifier_layout.addWidget(QtWidgets.QLabel("Window Size"), 2, 0)
        classifier_layout.addLayout(window_size_layout, 2, 1)
        classifier_layout.addWidget(self._use_balace_labels_checkbox, 4, 0, 1, 2)
        classifier_layout.addWidget(self._symmetric_behavior_checkbox, 5, 0, 1, 2)
        classifier_layout.addWidget(self._all_kfold_checkbox, 6, 0, 1, 2)
        classifier_layout.addWidget(self._kslider, 7, 0, 1, 2)
        classifier_layout.setContentsMargins(8, 5, 5, 5)
        classifier_group = QtWidgets.QGroupBox("Classifier")
        classifier_group.setLayout(classifier_layout)

        # label components
        label_layout = QtWidgets.QGridLayout()

        self._label_behavior_button = QtWidgets.QPushButton()
        self._label_behavior_button.setToolTip("[z]")
        self._label_behavior_button.clicked.connect(self.label_behavior_clicked)
        self._label_behavior_button.setStyleSheet(
            f"""
                QPushButton {{
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 rgba{BEHAVIOR_BUTTON_COLOR_BRIGHT.getRgb()},
                                       stop: 1.0 rgba{BEHAVIOR_COLOR.getRgb()});
                    border-radius: 4px;
                    padding: 2px;
                    color: white;
                }}
                QPushButton:pressed {{
                    background-color: rgba{BEHAVIOR_BUTTON_COLOR_BRIGHT.getRgb()};
                }}
                QPushButton:disabled {{
                    background-color: rgba{BEHAVIOR_BUTTON_DISABLED_COLOR.getRgb()};
                    color: grey;
                }}
            """
        )

        self._label_not_behavior_button = QtWidgets.QPushButton()
        self._label_not_behavior_button.setToolTip("[c]")
        self._label_not_behavior_button.clicked.connect(self.label_not_behavior_clicked)
        self._label_not_behavior_button.setStyleSheet(f"""
                    QPushButton {{
                        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 rgba{NOT_BEHAVIOR_COLOR_BRIGHT.getRgb()},
                                           stop: 1.0 rgba{NOT_BEHAVIOR_COLOR.getRgb()});
                        border-radius: 4px;
                        padding: 2px;
                        color: white;
                    }}
                    QPushButton:pressed {{
                        background-color: rgba{NOT_BEHAVIOR_COLOR_BRIGHT.getRgb()};
                    }}
                    QPushButton:disabled {{
                        background-color: rgba{NOT_BEHAVIOR_BUTTON_DISABLED_COLOR.getRgb()};
                        color: grey;
                    }}
                """)

        self._timeline_annotation_button = QtWidgets.QPushButton("New Timeline Annotation")
        self._timeline_annotation_button.clicked.connect(self.timeline_annotation_button_clicked)

        self._clear_label_button = QtWidgets.QPushButton("Clear Label")
        self._clear_label_button.setToolTip("[x]")
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
        label_layout.addWidget(self._timeline_annotation_button, 2, 0, 1, 2)
        label_layout.addWidget(self._clear_label_button, 3, 0)
        label_layout.addWidget(self._select_button, 3, 1)
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
        if sys.platform == "darwin":
            control_layout.setSpacing(20)
        else:
            control_layout.setSpacing(10)
        control_layout.addWidget(behavior_identity_group)
        control_layout.addWidget(classifier_group)
        control_layout.addWidget(label_count_group)
        control_layout.addStretch()
        control_layout.addWidget(label_group)

        self.setLayout(control_layout)

    @property
    def current_behavior(self):
        """return the current behavior name"""
        return self.behavior_selection.currentText()

    @property
    def behaviors(self):
        """return a copy of the current list of behaviors"""
        return list(self._behaviors)

    @property
    def current_identity(self) -> str:
        """get the external identity

        if the pose file doesn't have external identities this will be
        the string representation of the jabs identity
        """
        return self.identity_selection.currentText()

    @property
    def current_identity_index(self) -> int:
        """identity index is the same as the JABS identity"""
        return self.identity_selection.currentIndex()

    @property
    def select_button_is_checked(self):
        """return true if the select button is checked"""
        return self._select_button.isChecked()

    @property
    def kfold_value(self):
        """return the current value of the k-fold slider"""
        return self._kslider.value()

    @property
    def train_button_enabled(self):
        """return true if the train button is enabled"""
        return self._train_button.isEnabled()

    @train_button_enabled.setter
    def train_button_enabled(self, enabled: bool):
        """set the train button to enabled or disabled"""
        self._train_button.setEnabled(enabled)

    @property
    def classify_button_enabled(self):
        """is classify button enabled?"""
        return self._classify_button.isEnabled()

    @classify_button_enabled.setter
    def classify_button_enabled(self, enabled: bool):
        """set the classify button to enabled or disabled"""
        self._classify_button.setEnabled(enabled)

    @property
    def classifier_type(self):
        """return the selected classifier type"""
        return self._classifier_selection.currentData()

    @property
    def use_balance_labels(self):
        """return true if the balance labels checkbox is checked"""
        return self._use_balace_labels_checkbox.isChecked()

    @use_balance_labels.setter
    def use_balance_labels(self, val: bool):
        """set the balance labels checkbox to the given value (only if it is enabled)"""
        if self._use_balace_labels_checkbox.isEnabled():
            self._use_balace_labels_checkbox.setChecked(val)

    @property
    def use_symmetric(self):
        """return true if the symmetric behavior checkbox is checked"""
        return self._symmetric_behavior_checkbox.isChecked()

    @use_symmetric.setter
    def use_symmetric(self, val: bool):
        """set the symmetric behavior checkbox to the given value"""
        if self._symmetric_behavior_checkbox.isEnabled():
            self._symmetric_behavior_checkbox.setChecked(val)

    @property
    def all_kfold(self):
        """return true if the all k-fold checkbox is checked"""
        return self._all_kfold_checkbox.isChecked()

    def disable_label_buttons(self):
        """disable labeling buttons"""
        self._label_behavior_button.setEnabled(False)
        self._label_not_behavior_button.setEnabled(False)
        self._clear_label_button.setEnabled(False)
        self._select_button.setChecked(False)
        self._timeline_annotation_button.setEnabled(False)

    def enable_label_buttons(self):
        """enable labeling buttons"""
        self._label_behavior_button.setEnabled(True)
        self._label_not_behavior_button.setEnabled(True)
        self._clear_label_button.setEnabled(True)
        self._timeline_annotation_button.setEnabled(True)

    def set_use_balance_labels_checkbox_enabled(self, val: bool):
        """enable or disable the balance labels checkbox"""
        self._use_balace_labels_checkbox.setEnabled(val)
        if not val:
            self._use_balace_labels_checkbox.setChecked(False)

    def set_use_symmetric_checkbox_enabled(self, val: bool):
        """enable or disable the symmetric behavior checkbox"""
        self._symmetric_behavior_checkbox.setEnabled(val)
        if not val:
            self._symmetric_behavior_checkbox.setChecked(False)

    def set_classifier_selection(self, classifier_type):
        """set the classifier selection combobox to the given classifier type"""
        try:
            index = self._classifier_selection.findData(classifier_type)
            if index != -1:
                self._classifier_selection.setCurrentIndex(index)
        except KeyError:
            # unable to use the classifier
            pass

    def set_frame_counts(
        self,
        label_behavior_current,
        label_not_behavior_current,
        label_behavior_project,
        label_not_behavior_project,
        bout_behavior_current,
        bout_not_behavior_current,
        bout_behavior_project,
        bout_not_behavior_project,
    ):
        """set the frame counts displayed by the label count widget"""
        self._frame_counts.set_counts(
            label_behavior_current,
            label_not_behavior_current,
            label_behavior_project,
            label_not_behavior_project,
            bout_behavior_current,
            bout_not_behavior_current,
            bout_behavior_project,
            bout_not_behavior_project,
        )

    @property
    def select_button_enabled(self) -> bool:
        """return true if the select button is enabled"""
        return self._select_button.isEnabled()

    @select_button_enabled.setter
    def select_button_enabled(self, enabled: bool) -> None:
        """set the select button to enabled or disabled"""
        self._select_button.setEnabled(enabled)

    def select_button_set_enabled(self, enabled: bool):
        """set the select button to enabled or disabled"""
        self._select_button.setEnabled(enabled)

    def select_button_set_checked(self, checked):
        """set the select button to checked or unchecked"""
        self._select_button.setChecked(checked)

    def toggle_select_button(self):
        """toggle the select button"""
        self._select_button.toggle()

    def kslider_set_enabled(self, enabled: bool):
        """set the k-fold slider to enabled or disabled"""
        self._kslider.setEnabled(enabled)

    def set_identity_index(self, i: int):
        """set which identity is selected in the identity selection combobox"""
        self.identity_selection.setCurrentIndex(i)

    def set_behavior(self, behavior: str):
        """set the current behavior to the given behavior

        Args:
            behavior: the name of the behavior to set as current

        Returns:
            None
        """
        if behavior in self._behaviors:
            self.behavior_selection.setCurrentText(behavior)

    def update_project_settings(self, project_settings: dict):
        """update controls from project settings

        Args:
            project_settings: dict containing project settings

        Returns:
            None

        Todo:
         - This is one of the major locations where project settings are owned by this
           widget, instead of the project class
        """
        # reset list of behaviors, then add any from the project metadata
        self._behaviors = []

        # we don't need this even handler to be active while we set up the
        # project (otherwise it gets unnecessarily called multiple times)
        self.behavior_selection.currentIndexChanged.disconnect()

        self._set_window_sizes(project_settings["window_sizes"])
        self._add_label_button.setEnabled(True)

        # select the behavior
        behavior_index = 0
        if "behavior" in project_settings:
            self._behaviors = sorted(project_settings["behavior"].keys())
        self.behavior_selection.clear()
        self.behavior_selection.addItems(self._behaviors)
        if project_settings.get("selected_behavior") in self._behaviors:
            behavior_index = self._behaviors.index(project_settings["selected_behavior"])

        if len(self._behaviors) == 0:
            self._get_first_label()

        # set the index to either the first behavior, or if available, the one
        # that was saved in the project metadata
        self.behavior_selection.setCurrentIndex(behavior_index)
        # re-enable the behavior_selection change signal handler
        self.behavior_selection.currentIndexChanged.connect(self._behavior_changed)
        # run all the updates for when a behavior changes
        self._behavior_changed()

    def set_identities(self, identities: list[str]) -> None:
        """populate the identity_selection combobox with optional SVG icons"""
        self.identity_selection.currentIndexChanged.disconnect()
        self.identity_selection.clear()
        for display_name in identities:
            if (renderer := self._ear_tag_icons.get_icon(display_name)) is not None:
                pixmap = QPixmap(16, 16)
                pixmap.fill(QtCore.Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                renderer.render(painter)
                painter.end()
                self.identity_selection.addItem(QIcon(pixmap), display_name)
            else:
                self.identity_selection.addItem(display_name)
        self.identity_selection.currentIndexChanged.connect(self.identity_changed)

    def set_window_size(self, size: int):
        """set the current window size"""
        if self._window_size.findData(size) == -1:
            self._add_window_size(size)
        self._window_size.setCurrentText(str(size))

    def remove_behavior(self, behavior: str):
        """remove a behavior from the behavior selection box"""
        idx = self.behavior_selection.findText(behavior, QtCore.Qt.MatchFlag.MatchExactly)
        if idx != -1:
            self.behavior_selection.removeItem(idx)
            self._behaviors.remove(behavior)

        if len(self._behaviors) == 0:
            self._get_first_label()

    def _set_window_sizes(self, sizes: list[int]):
        """set the list of available window sizes"""
        self._window_size.clear()
        for w in sizes:
            self._window_size.addItem(str(w), userData=w)

    def _new_label(self):
        """callback for the "new behavior" button

        opens a modal dialog to allow the user to enter a new behavior label, if user clicks ok, add that
        behavior to the combo box, and select it
        """
        text, ok = QtWidgets.QInputDialog.getText(
            self, "New Behavior", "New Behavior Name:", QtWidgets.QLineEdit.EchoMode.Normal
        )
        if ok and text not in self._behaviors:
            self._behaviors.append(text)
            self._behaviors.sort()
            self.new_behavior_label.emit(self._behaviors)
            self.behavior_selection.addItem(text)
            self.behavior_selection.setCurrentText(text)

    def _get_first_label(self):
        """Show the new label dialog.

        Used when opening a new project for the fist time or if a user archives all behaviors in a project.

        dialog is customized to hide the window close button. The only way to close the dialog is to create a new
        label or to quit jabs (the Cancel button of the dialog has been renamed "Quit JABS").
        """
        dialog = QtWidgets.QInputDialog()
        dialog.setWindowTitle("New Behavior")
        dialog.setLabelText("Please enter a behavior name to continue:")
        dialog.setOkButtonText("OK")
        dialog.setCancelButtonText("Quit JABS")
        dialog.setWindowFlags(
            dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowCloseButtonHint
            | QtCore.Qt.WindowType.CustomizeWindowHint
        )

        if dialog.exec():
            text, ok = dialog.textValue(), dialog.result()
            if ok:
                self._behaviors = [text]
                self.behavior_selection.addItem(text)
                self.behavior_selection.setCurrentText(text)
                self.new_behavior_label.emit(self._behaviors)
                self._behavior_changed()
        else:
            sys.exit(0)

    def _new_window_size(self):
        """callback for the "new window size" button

        opens a modal dialog to allow the user to enter a new window size, if user clicks ok,
        add that window size and select it
        """
        val, ok = QtWidgets.QInputDialog.getInt(
            self, "New Window Size", "Enter a new window size:", value=1, minValue=1
        )
        if ok:
            # if this window size is not already in the drop-down, add it.
            if self._window_size.findData(val) == -1:
                self._add_window_size(val)

            # select new window size
            self.set_window_size(val)
            QtWidgets.QMessageBox.warning(
                self,
                "Window Size Added",
                "Window Size Added.\n"
                "If features have not been computed for "
                "this window size, they will be computed the first time a "
                "classifier is trained using this window size.\n"
                "This may be slow.",
            )

    def _add_window_size(self, new_size: int):
        if new_size is None:
            return

        # we clear and reset the contents of the combo box so that we
        # can re sort it with the new size

        # grab the old sizes, grabbing the data (int) instead of the
        # text
        sizes = [self._window_size.itemData(i) for i in range(self._window_size.count())]

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
        self._label_behavior_button.setToolTip(f"Label frames {self.current_behavior} [z]")
        self._label_not_behavior_button.setText(f"Not {self.current_behavior}")
        self._label_not_behavior_button.setToolTip(f"Label frames Not {self.current_behavior} [c]")
        self.behavior_changed.emit(self.current_behavior)

    def _window_size_changed(self):
        self.window_size_changed.emit(self._window_size.currentData())

    def _all_kfold_changed(self):
        if self._all_kfold_checkbox.isChecked():
            self._kslider.setEnabled(False)
        else:
            self._kslider.setEnabled(True)
