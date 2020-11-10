import numpy as np
from PyQt5 import QtWidgets, QtCore

from src.classifier.skl_classifier import SklClassifier
from src.labeler.track_labels import TrackLabels
from .classification_thread import ClassifyThread
from .frame_labels_widget import FrameLabelsWidget
from .global_inference_widget import GlobalInferenceWidget
from .identity_combo_box import IdentityComboBox
from .manual_label_widget import ManualLabelWidget
from .player_widget import PlayerWidget
from .prediction_vis_widget import PredictionVisWidget
from .timeline_label_widget import TimelineLabelWidget
from .training_thread import TrainingThread
from .label_count_widget import FrameLabelCountWidget
from .k_fold_slider_widget import KFoldSliderWidget
from .colors import BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR


class CentralWidget(QtWidgets.QWidget):
    """
    QT Widget implementing our main window contents
    """

    _DEFAULT_BEHAVIORS = [
        'Walking', 'Turn left', 'Turn right', 'Sleeping', 'Freezing',
        'Grooming', 'Following', 'Rearing (supported)',
        'Rearing (unsupported)'
    ]

    # signal that we have predictions to display
    have_predictions = QtCore.pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initial behavior labels to list in the drop down selection
        self._behaviors = list(self._DEFAULT_BEHAVIORS)
        self._behaviors.sort()

        # video player
        self._player_widget = PlayerWidget()
        self._player_widget.updateIdentities.connect(self._set_identities)
        self._player_widget.updateFrameNumber.connect(self._frame_change)

        self._loaded_video = None

        self._project = None
        self._labels = None

        #  classifier
        self._classifier = SklClassifier()
        self._training_thread = None
        self._classify_thread = None

        # information about current predictions
        self._predictions = {}
        self._probabilities = {}
        self._frame_indexes = {}

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

        behavior_group = QtWidgets.QGroupBox("Behavior Selection")
        behavior_group.setLayout(behavior_layout)

        # identity selection form components
        self.identity_selection = IdentityComboBox()
        self.identity_selection.currentIndexChanged.connect(
            self._change_identity)
        self.identity_selection.pop_up_visible.connect(self._identity_popup_visibility_changed)
        self.identity_selection.setEditable(False)
        self.identity_selection.installEventFilter(self.identity_selection)
        identity_layout = QtWidgets.QVBoxLayout()
        identity_layout.addWidget(self.identity_selection)
        identity_group = QtWidgets.QGroupBox("Identity Selection")
        identity_group.setLayout(identity_layout)

        # classifier controls
        #  buttons
        self.train_button = QtWidgets.QPushButton("Train")
        self.train_button.clicked.connect(self._train_button_clicked)
        self.train_button.setEnabled(False)
        self.classify_button = QtWidgets.QPushButton("Classify")
        self.classify_button.clicked.connect(self._classify_button_clicked)
        self.classify_button.setEnabled(False)

        #  drop down to select type of classifier to use
        self._classifier_selection = QtWidgets.QComboBox()
        self._classifier_selection.currentIndexChanged.connect(self._classifier_changed)

        for classifier, name in self._classifier.classifier_choices().items():
            self._classifier_selection.addItem(name, userData=classifier)

        #  slider to set number of times to train/test
        self._kslider = KFoldSliderWidget()
        self._kslider.valueChanged.connect(self._kfold_changed)

        #  classifier control layout
        classifier_layout = QtWidgets.QGridLayout()
        classifier_layout.addWidget(self.train_button, 0, 0)
        classifier_layout.addWidget(self.classify_button, 0, 1)
        classifier_layout.addWidget(QtWidgets.QLabel("Classifier: "), 1, 0)
        classifier_layout.addWidget(self._classifier_selection, 2, 0, 1, 2)
        classifier_layout.addWidget(self._kslider, 3, 0, 1, 2)
        classifier_group = QtWidgets.QGroupBox("Classifier")
        classifier_group.setLayout(classifier_layout)

        # label components
        label_layout = QtWidgets.QVBoxLayout()

        self.label_behavior_button = QtWidgets.QPushButton()
        self.label_behavior_button.setText(
            self.behavior_selection.currentText())
        self.label_behavior_button.clicked.connect(self._label_behavior)
        self.label_behavior_button.setStyleSheet(f"""
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

        self.label_not_behavior_button = QtWidgets.QPushButton(
            f"Not {self.behavior_selection.currentText()}")
        self.label_not_behavior_button.clicked.connect(self._label_not_behavior)
        self.label_not_behavior_button.setStyleSheet(f"""
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

        # summary of number of frames / bouts for each class
        self._frame_counts = FrameLabelCountWidget()
        label_count_layout = QtWidgets.QVBoxLayout()
        label_count_layout.addWidget(self._frame_counts)
        label_count_group = QtWidgets.QGroupBox("Label Summary")
        label_count_group.setLayout(label_count_layout)

        # control layout
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setSpacing(25)
        control_layout.addWidget(behavior_group)
        control_layout.addWidget(identity_group)
        control_layout.addWidget(classifier_group)
        control_layout.addWidget(label_count_group)
        control_layout.addStretch()
        control_layout.addWidget(label_group)

        # label widgets
        self.manual_labels = ManualLabelWidget()
        self.prediction_vis = PredictionVisWidget()
        self.frame_ticks = FrameLabelsWidget()

        # timeline widgets
        self.timeline_widget = TimelineLabelWidget()
        self.inference_timeline_widget = GlobalInferenceWidget()

        # main layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self._player_widget, 0, 0)
        layout.addLayout(control_layout, 0, 1)
        layout.addWidget(self.timeline_widget, 1, 0, 1, 2)
        layout.addWidget(self.manual_labels, 2, 0, 1, 2)
        layout.addWidget(self.inference_timeline_widget, 3, 0, 1, 2)
        layout.addWidget(self.prediction_vis, 4, 0, 1, 2)
        layout.addWidget(self.frame_ticks, 5, 0, 1, 2)
        self.setLayout(layout)

        # progress bar dialog used when running the training or classify threads
        self._progress_dialog = None

        self._label_counts = None
        self._bout_counts = None

    def behavior(self):
        """
        :return: the currently selected behavior
        """
        return self.behavior_selection.currentText()

    def behavior_labels(self):
        """
        get the current contents of the behavior drop down
        :return: a copy of the list so private member can't be modified
        """
        return list(self._behaviors)

    def classifier_type(self):
        """ get the current classifier type """
        return self._classifier.classifier_type

    def set_project(self, project):
        """ set the currently opened project """
        self._project = project
        self.classify_button.setEnabled(False)

        # This will get set when the first video in the project is loaded, but
        # we need to set it to None so that we don't try to cache the current
        # labels when we do so (the current labels belong to the previous
        # project)
        self._labels = None
        self._loaded_video = None

        # get project specific settings
        settings = project.settings

        # try to select the classifier type specified in the project settings
        try:
            classifier_type = SklClassifier.ClassifierType[settings['classifier']]

            index = self._classifier_selection.findData(classifier_type)
            if index != -1:
                self._classifier_selection.setCurrentIndex(index)
        except KeyError:
            # either no classifier was specified in the settings file, or
            # unable to use the classifier specified in the settings file.
            # use the default
            pass

        # reset list of projects, then add any from the settings
        self._behaviors = list(self._DEFAULT_BEHAVIORS)

        # we don't need this even handler to be active while we set up the
        # project (otherwise it gets unnecessarily called multiple times)
        self.behavior_selection.currentIndexChanged.disconnect()

        behavior_index = 0
        if 'behaviors' in settings:
            # add behavior labels from project settings that aren't already in
            # the app default list
            for b in settings['behaviors']:
                if b not in self._behaviors:
                    self._behaviors.append(b)
            self._behaviors.sort()
            self.behavior_selection.clear()
            self.behavior_selection.addItems(self._behaviors)
        if 'selected_behavior' in settings:
            # make sure this behavior is in the behavior selection drop down
            if settings['selected_behavior'] not in self._behaviors:
                self._behaviors.append(settings['selected_behavior'])
                self._behaviors.sort()
                self.behavior_selection.clear()
                self.behavior_selection.addItems(self._behaviors)
            behavior_index = self._behaviors.index(settings['selected_behavior'])

        # set the index to either the first behavior, or if available, the one
        # that was saved in the project settings
        self.behavior_selection.setCurrentIndex(behavior_index)

        # get label/bout counts for the current project
        self._label_counts = self._project.label_counts(self.behavior())
        self._bout_counts = self._project.bout_counts(self.behavior())

        # re-enable the behavior_selection change signal handler
        self.behavior_selection.currentIndexChanged.connect(
            self._change_behavior)

    def get_labels(self):
        """
        get VideoLabels for currently opened video file
        note: the @property decorator doesn't work with QWidgets so we have
        not implemented this as a property
        """
        return self._labels

    def load_video(self, path):
        """
        load a new video file into self._player_widget
        :param path: path to video file
        :return: None
        :raises: OSError if unable to open video
        """

        # if we have labels loaded, cache them before opening labels for
        # new video
        if self._labels is not None:
            self._project.cache_annotations(self._labels)
            self._start_selection(False)
            self.select_button.setChecked(False)

        try:
            # open the video
            self._player_widget.load_video(path)

            # load labels for new video and set track for current identity
            self._labels = self._project.load_annotation_track(path)
            self._set_label_track()

            # update ui components with properties of new video
            self.manual_labels.set_num_frames(self._player_widget.num_frames())
            self.manual_labels.set_framerate(self._player_widget.stream_fps())
            self.prediction_vis.set_num_frames(self._player_widget.num_frames())
            self.frame_ticks.set_num_frames(self._player_widget.num_frames())
            self.timeline_widget.set_num_frames(
                self._player_widget.num_frames())
            self.inference_timeline_widget.set_num_frames(
                self._player_widget.num_frames()
            )

            self._loaded_video = path
            self._set_prediction_vis()
            self._update_label_counts()
            self._set_train_button_enabled_state()
        except OSError as e:
            # error loading
            self._labels = None
            self._loaded_video = None
            self._player_widget.reset()
            raise e

    def keyPressEvent(self, event):
        """ handle key press events """

        def begin_select_mode():
            if not self.select_button.isChecked():
                self.select_button.toggle()
                self._start_selection(True)

        key = event.key()
        if key == QtCore.Qt.Key_Left:
            self._player_widget.previous_frame()
        elif key == QtCore.Qt.Key_Right:
            self._player_widget.next_frame()
        elif key == QtCore.Qt.Key_Up:
            self._player_widget.next_frame(self._frame_jump)
        elif key == QtCore.Qt.Key_Down:
            self._player_widget.previous_frame(self._frame_jump)
        elif key == QtCore.Qt.Key_Space:
            self._player_widget.toggle_play()
        elif key == QtCore.Qt.Key_Z:
            if self.select_button.isChecked():
                self._label_behavior()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key_X:
            if self.select_button.isChecked():
                self._clear_behavior_label()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key_C:
            if self.select_button.isChecked():
                self._label_not_behavior()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key_Escape:
            if self.select_button.isChecked():
                self.select_button.setChecked(False)
                self._start_selection(False)

    def _new_label(self):
        """
        callback for the "new behavior" button
        opens a modal dialog to allow the user to enter a new behavior label
        """
        text, ok = QtWidgets.QInputDialog.getText(None, 'New Label',
                                                  'New Label Name:')
        if ok and text not in self._behaviors:
            self._behaviors.append(text)
            self.behavior_selection.addItem(text)
            self.behavior_selection.setCurrentIndex(self._behaviors.index(text))

    def _change_behavior(self):
        """
        make UI changes to reflect the currently selected behavior
        """
        self.label_behavior_button.setText(
            self.behavior_selection.currentText())
        self.label_not_behavior_button.setText(
            f"Not {self.behavior_selection.currentText()}")
        self._set_label_track()
        self._reset_prediction()
        self.classify_button.setEnabled(False)
        self._update_label_counts()
        self._set_train_button_enabled_state()

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
        start, end = sorted([self._selection_start,
                             self._player_widget.current_frame()])
        mask = self._player_widget.get_identity_mask()
        self._get_label_track().label_behavior(start, end, mask[start:end+1])
        self._label_button_common()

    def _label_not_behavior(self):
        """ apply _not_ behavior label to currently selected range of frames """
        start, end = sorted([self._selection_start,
                             self._player_widget.current_frame()])
        mask = self._player_widget.get_identity_mask()
        self._get_label_track().label_not_behavior(start,
                                                   end, mask[start:end+1])
        self._label_button_common()

    def _clear_behavior_label(self):
        """ clear all behavior/not behavior labels from current selection """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._get_label_track().clear_labels(*label_range)
        self._label_button_common()

    def _label_button_common(self):
        """
        functionality shared between _label_behavior(), _label_not_behavior(),
        and _clear_behavior_label(). to be called after the labels are changed
        for the current selection
        """
        self._project.save_annotations(self._labels)
        self._disable_label_buttons()
        self.manual_labels.clear_selection()
        self.manual_labels.update()
        self.timeline_widget.update_labels()
        self._update_label_counts()
        self._set_train_button_enabled_state()

    def _set_identities(self, identities):
        """ populate the identity_selection combobox """
        self.identity_selection.clear()
        self.identity_selection.addItems([str(i) for i in identities])
        self._player_widget.set_identities(identities)

    def _change_identity(self):
        """ handle changing value of identity_selection """
        self._player_widget.set_active_identity(
            self.identity_selection.currentIndex())
        self._set_label_track()
        self._update_label_counts()

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
        self.prediction_vis.set_current_frame(new_frame)
        self.timeline_widget.set_current_frame(new_frame)
        self.inference_timeline_widget.set_current_frame(new_frame)
        self.frame_ticks.set_current_frame(new_frame)

    def _set_label_track(self):
        """
        loads new set of labels in self.manual_labels when the selected
        behavior or identity is changed
        """
        behavior = self.behavior_selection.currentText()
        identity = self.identity_selection.currentText()

        if identity != '' and behavior != '' and self._labels is not None:
            labels = self._labels.get_track_labels(identity, behavior)
            self.manual_labels.set_labels(
                labels, mask=self._player_widget.get_identity_mask())
            self.timeline_widget.set_labels(labels)

        self._set_prediction_vis()

    def _get_label_track(self):
        """
        get the current label track for the currently selected identity and
        behavior
        """
        return self._labels.get_track_labels(
            self.identity_selection.currentText(),
            self.behavior_selection.currentText()
        )

    @QtCore.pyqtSlot(bool)
    def _identity_popup_visibility_changed(self, visible):
        """
        connected to the IdentityComboBox.pop_up_visible signal. When
        visible == True, we tell the player widget to overlay all identity
        labels on the frame.
        When visible == False we revert to the normal behavior of only labeling
        the currently selected identity
        """
        self._player_widget.set_identity_label_mode(visible)

    def _train_button_clicked(self):
        """ handle user click on "Train" button """
        # make sure video playback is stopped
        self._player_widget.stop()

        # setup training thread
        self._training_thread = TrainingThread(
            self._project, self._classifier,
            self.behavior_selection.currentText(),
            self._loaded_video, self._labels, self._kslider.value())
        self._training_thread.trainingComplete.connect(
            self._training_thread_complete)
        self._training_thread.update_progress.connect(
            self._update_training_progress)

        # setup progress dialog
        self._progress_dialog = QtWidgets.QProgressDialog(
            'Training', None, 0,
            self._project.total_project_identities + self._kslider.value(),
            self)
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self._progress_dialog.reset()
        self._progress_dialog.show()

        # start training thread
        self._training_thread.start()

    def _training_thread_complete(self):
        """ enable classify button once the training is complete """
        self._progress_dialog.reset()
        self.classify_button.setEnabled(True)

    def _update_training_progress(self, step):
        """ update progress bar with the number of completed tasks """
        self._progress_dialog.setValue(step)

    def _classify_button_clicked(self):
        """ handle user click on "Classify" button """
        # make sure video playback is stopped
        self._player_widget.stop()

        # setup classification thread
        self._classify_thread = ClassifyThread(
            self._classifier, self._project,
            self.behavior_selection.currentText(), self._loaded_video,
            self._labels, self._predictions, self._probabilities,
            self._frame_indexes)
        self._classify_thread.done.connect(self._classify_thread_complete)
        self._classify_thread.update_progress.connect(
            self._update_classify_progress)

        # setup progress dialog
        self._progress_dialog = QtWidgets.QProgressDialog(
            'Predicting', None, 0, self._project.total_project_identities,
            self)
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self._progress_dialog.reset()
        self._progress_dialog.show()

        # start classification thread
        self._classify_thread.start()

    def _classify_thread_complete(self):
        """ update the gui when the classification is complete """
        # display the new predictions
        self._set_prediction_vis()
        # let the MainWindow know we have predictions so it can enable the
        # file action to save the predictions to the project directory
        self.have_predictions.emit(True)

    def _update_classify_progress(self, step):
        """ update progress bar with the number of completed tasks """
        self._progress_dialog.setValue(step)

    def _set_prediction_vis(self):
        """
        update data being displayed by the prediction visualization widget
        """

        if self._loaded_video is None:
            return

        video = self._loaded_video.name
        identity = self.identity_selection.currentText()

        try:
            indexes = self._frame_indexes[video][identity]
        except KeyError:
            return

        labels = self._get_label_track().get_labels()
        prediction_labels = np.zeros((self._player_widget.num_frames()),
                                     dtype=np.uint8)
        prediction_prob = np.zeros((self._player_widget.num_frames()),
                                   dtype=np.float64)

        prediction_labels[indexes] = self._predictions[video][identity]
        prediction_prob[indexes] = self._probabilities[video][identity]
        prediction_labels[labels == TrackLabels.Label.NOT_BEHAVIOR] = TrackLabels.Label.NOT_BEHAVIOR
        prediction_prob[labels == TrackLabels.Label.NOT_BEHAVIOR] = 1.0
        prediction_labels[labels == TrackLabels.Label.BEHAVIOR] = TrackLabels.Label.BEHAVIOR
        prediction_prob[labels == TrackLabels.Label.BEHAVIOR] = 1.0

        self.prediction_vis.set_predictions(prediction_labels, prediction_prob)
        self.inference_timeline_widget.set_labels(prediction_labels)
        self.inference_timeline_widget.update_labels()

    def _reset_prediction(self):
        """ clear out the current predictions """
        if len(self._predictions) != 0:
            self._predictions = {}
            self._probabilities = {}
            self._frame_indexes = {}
            self.prediction_vis.set_predictions(None, None)
            self.inference_timeline_widget.set_labels(
                np.zeros(self._player_widget.num_frames(), dtype="uint8"))

    def _set_train_button_enabled_state(self):
        """
        set the enabled property of the train button to True or False depending
        whether the labeling meets some threshold set by the classifier module

        NOTE: must be called after _update_label_counts() so that it has the
        correct counts for the current video
        :return: None
        """

        if SklClassifier.label_threshold_met(self._label_counts,
                                             self._kslider.value()):
            self.train_button.setEnabled(True)
        else:
            self.train_button.setEnabled(False)

    def _update_label_counts(self):
        """
        update the widget with the labeled frame / bout counts

        :return: None
        """

        if self._loaded_video is None:
            return

        # update counts for the current video -- we could be more efficient
        # by only updating the current identity in the current video
        self._label_counts[self._loaded_video.name] = self._labels.label_counts(
            self.behavior())
        self._bout_counts[self._loaded_video.name] = self._labels.bout_counts(
            self.behavior())

        identity = self.identity_selection.currentText()

        label_behavior_current = 0
        label_not_behavior_current = 0
        label_behavior_project = 0
        label_not_behavior_project = 0
        bout_behavior_current = 0
        bout_not_behavior_current = 0
        bout_behavior_project = 0
        bout_not_behavior_project = 0

        for video, video_counts in self._label_counts.items():
            for identity_counts in video_counts:
                label_behavior_project += identity_counts[1][0]
                label_not_behavior_project += identity_counts[1][1]
                if video == self._loaded_video.name and identity_counts[0] == identity:
                    label_behavior_current += identity_counts[1][0]
                    label_not_behavior_current += identity_counts[1][1]

        for video, video_counts in self._bout_counts.items():
            for identity_counts in video_counts:
                bout_behavior_project += identity_counts[1][0]
                bout_not_behavior_project += identity_counts[1][1]
                if video == self._loaded_video.name and identity_counts[0] == identity:
                    bout_behavior_current += identity_counts[1][0]
                    bout_not_behavior_current += identity_counts[1][1]

        self._frame_counts.set_counts(label_behavior_current,
                                      label_not_behavior_current,
                                      label_behavior_project,
                                      label_not_behavior_project,
                                      bout_behavior_current,
                                      bout_not_behavior_current,
                                      bout_behavior_project,
                                      bout_not_behavior_project)

    def _kfold_changed(self):
        """ handle kfold slider change event """
        self._set_train_button_enabled_state()

    def _classifier_changed(self):
        """ handle classifier selection change """
        self._classifier.set_classifier(self._classifier_selection.currentData())

    def save_predictions(self):
        """ save predictions (if the classifier has been run) """
        if not self._predictions:
            return
        self._project.save_predictions(self._predictions, self._probabilities,
                                       self._frame_indexes,
                                       self.behavior_selection.currentText())
