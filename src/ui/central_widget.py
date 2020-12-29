import sys

import numpy as np
from PySide2 import QtWidgets, QtCore
from shapely.geometry import Point

from src.classifier.classifier import Classifier
from src.project.track_labels import TrackLabels
import src.feature_extraction
from .classification_thread import ClassifyThread

from .frame_labels_widget import FrameLabelsWidget
from .global_inference_widget import GlobalInferenceWidget
from .manual_label_widget import ManualLabelWidget
from .player_widget import PlayerWidget
from .prediction_vis_widget import PredictionVisWidget
from .timeline_label_widget import TimelineLabelWidget
from .training_thread import TrainingThread
from .main_control_widget import MainControlWidget


class CentralWidget(QtWidgets.QWidget):
    """
    QT Widget implementing our main window contents
    """

    export_training_status_change = QtCore.Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # video player
        self._player_widget = PlayerWidget()
        self._player_widget.updateIdentities.connect(self._set_identities)
        self._player_widget.updateFrameNumber.connect(self._frame_change)
        self._player_widget.pixmap_clicked.connect(self._pixmap_clicked)
        self._curr_frame_index = 0

        self._loaded_video = None
        self._project = None
        self._labels = None
        self._pose_est = None

        #  classifier
        self._classifier = Classifier(n_jobs=4)
        self._training_thread = None
        self._classify_thread = None

        # information about current predictions
        self._predictions = {}
        self._probabilities = {}
        self._frame_indexes = {}

        self._selection_start = 0

        # options
        self._frame_jump = 10
        self._window_size = src.feature_extraction.DEFAULT_WINDOW_SIZE

        # main controls
        self._controls = MainControlWidget()
        self._controls.identity_changed.connect(self._change_identity)
        self._controls.label_behavior_clicked.connect(self._label_behavior)
        self._controls.label_not_behavior_clicked.connect(self._label_not_behavior)
        self._controls.clear_label_clicked.connect(self._clear_behavior_label)
        self._controls.start_selection.connect(self._start_selection)
        self._controls.train_clicked.connect(self._train_button_clicked)
        self._controls.classify_clicked.connect(self._classify_button_clicked)
        self._controls.classifier_changed.connect(self._classifier_changed)
        self._controls.behavior_changed.connect(self._change_behavior)
        self._controls.kfold_changed.connect(
            self._set_train_button_enabled_state)
        self._controls.behavior_list_changed.connect(
            lambda b: self._project.save_metadata({'behaviors': b}))
        self._controls.window_size_changed.connect(self._window_feature_size_changed)
        self._controls.new_window_sizes.connect(self._save_window_sizes)

        # label & prediction vis widgets
        self.manual_labels = ManualLabelWidget()
        self.prediction_vis = PredictionVisWidget()
        self.frame_ticks = FrameLabelsWidget()

        # timeline widgets
        self.timeline_widget = TimelineLabelWidget()
        self.inference_timeline_widget = GlobalInferenceWidget()

        # main layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self._player_widget, 0, 0)
        layout.addWidget(self._controls, 0, 1, 5, 1)
        layout.addWidget(self.timeline_widget, 1, 0)
        layout.addWidget(self.manual_labels, 2, 0)
        layout.addWidget(self.inference_timeline_widget, 3, 0)
        layout.addWidget(self.prediction_vis, 4, 0)
        layout.addWidget(self.frame_ticks, 5, 0)
        self.setLayout(layout)

        # progress bar dialog used when running the training or classify threads
        self._progress_dialog = None

        self._counts = None

        # set focus policy of all children widgets, needed to keep controls
        # from grabbing focus on Windows (which breaks arrow key video nav)
        for child in self.findChildren(QtWidgets.QWidget):
            child.setFocusPolicy(QtCore.Qt.NoFocus)

    @property
    def behavior(self):
        """ get the currently selected behavior """
        return self._controls.current_behavior

    @property
    def classifier_type(self):
        """ get the current classifier type """
        return self._classifier.classifier_type

    @property
    def window_size(self):
        """ get current window size """
        return self._window_size

    @property
    def classify_button_enabled(self):
        """
        return true if the classify button is currently enabled, false otherwise
        """
        return self._controls.classify_button_enabled

    def set_project(self, project):
        """ set the currently opened project """
        self._project = project

        # This will get set when the first video in the project is loaded, but
        # we need to set it to None so that we don't try to cache the current
        # labels when we do so (the current labels belong to the previous
        # project)
        self._labels = None
        self._loaded_video = None

        self._controls.classify_button_set_enabled(False)
        self._controls.update_project_settings(project.metadata)

        classifier_loaded = False
        try:
            classifier_loaded = self._project.load_classifier(
                self._classifier, self.behavior)
        except Exception as e:
            print('failed to load classifier', file=sys.stderr)
            print(e, file=sys.stderr)

        # if a classifier was loaded, set the drop down to match the type
        if classifier_loaded:
            self._update_classifier_selection()

        # set classify button state
        # it will only get enabled if we loaded a classifier and we know
        # what window size was used to train it
        window_settings = self._project.metadata.get('window_size_pref', {})
        # do we have a window size for this behavior?
        if self.behavior in window_settings:
            # yes, does it match current window size?
            if self.window_size == window_settings[self.behavior]:
                # yes, classify button can be enabled if we loaded a classifier
                self._controls.classify_button_set_enabled(classifier_loaded)

        # get label/bout counts for the current project
        self._counts = self._project.counts(self.behavior)

        self._controls.select_button_set_enabled(True)
        self._controls.kslider_set_enabled(True)
        self._set_train_button_enabled_state()

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
            self._controls.select_button_set_checked(False)

        try:
            # open the video
            self._pose_est = self._project.load_pose_est(path)
            self._player_widget.load_video(path, self._pose_est)

            # load labels for new video and set track for current identity
            self._labels = self._project.load_video_labels(path)

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

            # load saved predictions for this video
            self._predictions, self._probabilities, self._frame_indexes = \
                self._project.load_predictions(path.name,
                                               self.behavior)

            self._loaded_video = path

            # update display with labels/predictions for this video
            self._set_label_track()
            self._update_label_counts()
            self._set_prediction_vis()
        except OSError as e:
            # error loading
            self._labels = None
            self._loaded_video = None
            self._pose_est = None
            self._set_identities([])
            self._player_widget.reset()
            raise e

    def keyPressEvent(self, event):
        """ handle key press events """

        def begin_select_mode():
            if not self._controls.select_button_is_checked:
                self._controls.toggle_select_button()
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
            if self._controls.select_button_is_checked:
                self._label_behavior()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key_X:
            if self._controls.select_button_is_checked:
                self._clear_behavior_label()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key_C:
            if self._controls.select_button_is_checked:
                self._label_not_behavior()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key_Escape:
            if self._controls.select_button_is_checked:
                self._controls.select_button_set_checked(False)
                self._start_selection(False)
        elif key == QtCore.Qt.Key_L:
            # show closest with no argument toggles the setting
            self._player_widget.show_closest()

    def show_track(self, show: bool):
        self._player_widget.show_track(show)

    def overlay_pose(self, new_val: bool):
        self._player_widget.overlay_pose(new_val)

    def _change_behavior(self):
        """
        make UI changes to reflect the currently selected behavior
        """
        if self._project is None:
            return

        self._controls.classify_button_set_enabled(False)

        classifier_loaded = False
        try:
            classifier_loaded = self._project.load_classifier(
                self._classifier, self.behavior)
        except Exception as e:
            print('failed to load classifier', file=sys.stderr)
            print(e, file=sys.stderr)

        if classifier_loaded:
            self._update_classifier_selection()

        # get label/bout counts for the current project
        self._counts = self._project.counts(self.behavior)
        self._update_label_counts()

        # load saved predictions
        if self._loaded_video:
            self._predictions, self._probabilities, self._frame_indexes = \
                self._project.load_predictions(self._loaded_video.name, self.behavior)

        # display labels and predictions for new behavior
        self._set_label_track()
        self._set_train_button_enabled_state()
        self._project.save_metadata({'selected_behavior': self.behavior})

        # try to set the window size to the last one used to train this
        # behavior
        window_settings = self._project.metadata.get('window_size_pref', {})
        # do we have a window size in the project settings for this behavior?
        if self.behavior in window_settings:
            # yes, try to set the window size to match
            self._controls.set_window_size(window_settings[self.behavior])
            if self.window_size == window_settings[self.behavior]:
                # success: enable classify button if a classifier was loaded
                self._controls.classify_button_set_enabled(classifier_loaded)

    def _start_selection(self, pressed):
        """
        handle click on "select" button. If button was previously "unchecked"
        then grab the current frame to begin selecting a range. If the
        button was in the checked state, clicking cancels the current selection.

        While selection is in progress, the labeling buttons become active.
        """
        if pressed:
            self._controls.enable_label_buttons()
            self._selection_start = self._player_widget.current_frame()
            self.manual_labels.start_selection(self._selection_start)
        else:
            self._controls.disable_label_buttons()
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
        self._controls.disable_label_buttons()
        self.manual_labels.clear_selection()
        self.manual_labels.update()
        self.timeline_widget.update_labels()
        self._update_label_counts()
        self._set_train_button_enabled_state()

    def _set_identities(self, identities):
        """ populate the identity_selection combobox """
        self._controls.set_identities(identities)
        self._player_widget.set_identities(identities)

    def _change_identity(self):
        """ handle changing value of identity_selection """
        self._player_widget.set_active_identity(
            self._controls.current_identity_index)
        self._set_label_track()
        self._update_label_counts()

    def _frame_change(self, new_frame):
        """
        called when the video player widget emits its updateFrameNumber signal
        """
        self._curr_frame_index = new_frame
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
        behavior = self._controls.current_behavior
        identity = self._controls.current_identity

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
            self._controls.current_identity,
            self._controls.current_behavior
        )

    def _update_classifier_selection(self):
        """
        Called when the classifier selection widget should be updated
        """
        self._controls.set_classifier_selection(self._classifier.classifier_type)

    def _train_button_clicked(self):
        """ handle user click on "Train" button """
        # make sure video playback is stopped
        self._player_widget.stop()

        # setup training thread
        self._training_thread = TrainingThread(
            self._project, self._classifier,
            self._controls.current_behavior,
            self._window_size,
            self._controls.kfold_value)
        self._training_thread.training_complete.connect(
            self._training_thread_complete)
        self._training_thread.update_progress.connect(
            self._update_training_progress)
        self._training_thread.current_status.connect(
            lambda m: self.parent().display_status_message(m, 0))

        # setup progress dialog
        self._progress_dialog = QtWidgets.QProgressDialog(
            'Training', None, 0,
            self._project.total_project_identities + self._controls.kfold_value + 1,
            self)
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self._progress_dialog.reset()
        self._progress_dialog.show()

        # start training thread
        self._training_thread.start()

        # save the window size used for this behavior
        window_settings = self._project.metadata.get('window_size_pref', {})
        window_settings[self.behavior] = self._window_size
        self._project.save_metadata({'window_size_pref': window_settings})

    def _training_thread_complete(self):
        """ enable classify button once the training is complete """
        self._progress_dialog.reset()
        self.parent().display_status_message("Training Complete", 3000)
        self._controls.classify_button_set_enabled(True)

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
            self._controls.current_behavior, self._predictions,
            self._probabilities, self._frame_indexes, self._loaded_video.name,
            self._window_size)
        self._classify_thread.done.connect(self._classify_thread_complete)
        self._classify_thread.update_progress.connect(
            self._update_classify_progress)
        self._classify_thread.current_status.connect(
            lambda m: self.parent().display_status_message(m, 0))

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
        self.parent().display_status_message("Classification Complete")
        self._set_prediction_vis()

    def _update_classify_progress(self, step):
        """ update progress bar with the number of completed tasks """
        self._progress_dialog.setValue(step)

    def _set_prediction_vis(self):
        """
        update data being displayed by the prediction visualization widget
        """

        if self._loaded_video is None:
            return

        identity = self._controls.current_identity

        try:
            indexes = self._frame_indexes[identity]
        except KeyError:
            self.prediction_vis.set_predictions(None, None)
            self.inference_timeline_widget.set_labels(
                np.full(self._player_widget.num_frames(),
                        TrackLabels.Label.NONE.value, dtype=np.byte))
            return

        labels = self._get_label_track().get_labels()
        prediction_labels = np.full((self._player_widget.num_frames()),
                                    TrackLabels.Label.NONE.value,
                                    dtype=np.byte)
        prediction_prob = np.zeros((self._player_widget.num_frames()),
                                   dtype=np.float64)

        prediction_labels[indexes] = self._predictions[identity]
        prediction_prob[indexes] = self._probabilities[identity]
        prediction_labels[labels == TrackLabels.Label.NOT_BEHAVIOR] = TrackLabels.Label.NOT_BEHAVIOR
        prediction_prob[labels == TrackLabels.Label.NOT_BEHAVIOR] = 1.0
        prediction_labels[labels == TrackLabels.Label.BEHAVIOR] = TrackLabels.Label.BEHAVIOR
        prediction_prob[labels == TrackLabels.Label.BEHAVIOR] = 1.0

        self.prediction_vis.set_predictions(prediction_labels, prediction_prob)
        self.inference_timeline_widget.set_labels(prediction_labels)
        self.inference_timeline_widget.update_labels()

    def _set_train_button_enabled_state(self):
        """
        set the enabled property of the train button to True or False depending
        whether the labeling meets some threshold set by the classifier module

        NOTE: must be called after _update_label_counts() so that it has the
        correct counts for the current video
        :return: None
        """

        if Classifier.label_threshold_met(self._counts,
                                          self._controls.kfold_value):
            self._controls.train_button_enabled = True
            self.export_training_status_change.emit(True)
        else:
            self._controls.train_button_enabled = False
            self.export_training_status_change.emit(False)

    def _update_label_counts(self):
        """
        update the widget with the labeled frame / bout counts

        :return: None
        """

        if self._loaded_video is None:
            return

        # update counts for the current video -- we could be more efficient
        # by only updating the current identity in the current video
        self._counts[self._loaded_video.name] = self._labels.counts(self.behavior)

        identity = self._controls.current_identity

        label_behavior_current = 0
        label_not_behavior_current = 0
        label_behavior_project = 0
        label_not_behavior_project = 0
        bout_behavior_current = 0
        bout_not_behavior_current = 0
        bout_behavior_project = 0
        bout_not_behavior_project = 0

        for video, video_counts in self._counts.items():
            for identity_counts in video_counts:
                label_behavior_project += identity_counts[1][0]
                label_not_behavior_project += identity_counts[1][1]
                bout_behavior_project += identity_counts[2][0]
                bout_not_behavior_project += identity_counts[2][1]
                if video == self._loaded_video.name and identity_counts[0] == identity:
                    label_behavior_current += identity_counts[1][0]
                    label_not_behavior_current += identity_counts[1][1]
                    bout_behavior_current += identity_counts[2][0]
                    bout_not_behavior_current += identity_counts[2][1]

        self._controls.set_frame_counts(label_behavior_current,
                                        label_not_behavior_current,
                                        label_behavior_project,
                                        label_not_behavior_project,
                                        bout_behavior_current,
                                        bout_not_behavior_current,
                                        bout_behavior_project,
                                        bout_not_behavior_project)

    def _classifier_changed(self):
        """ handle classifier selection change """
        if self._classifier.classifier_type != self._controls.classifier_type:
            # changing classifier type, disable until retrained
            self._controls.classify_button_set_enabled(False)
            self._classifier.set_classifier(self._controls.classifier_type)

    def _pixmap_clicked(self, event):
        """
        handle event where user clicked on the video -- if they click
        on one of the mice, make that one active
        """
        if self._pose_est is not None:
            # since convex hulls are represented as y, x we need to maintain
            # this ordering
            pt = Point(event['y'], event['x'])
            for i, ident in enumerate(self._pose_est.identities):
                c_hulls = self._pose_est.get_identity_convex_hulls(ident)
                curr_c_hull = c_hulls[self._curr_frame_index]
                if curr_c_hull is not None and curr_c_hull.contains(pt):
                    self._controls.set_identity_index(i)
                    break

    def _window_feature_size_changed(self, new_size):
        """ handle window feature size change """
        if new_size is not None and new_size != self._window_size:
            self._window_size = new_size
            # if we change the window size disable the classify button until
            # the classifier is retrained
            self._controls.classify_button_set_enabled(False)

    def _save_window_sizes(self, window_sizes):
        """ save the window sizes to the project settings """
        self._project.save_metadata({'window_sizes': window_sizes})
