import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from shapely.geometry import Point

import jabs.feature_extraction
from jabs.classifier import Classifier
from jabs.project import VideoLabels
from jabs.project.track_labels import TrackLabels
from jabs.video_reader.utilities import get_frame_count

from .classification_thread import ClassifyThread
from .main_control_widget import MainControlWidget
from .player_widget import PlayerWidget
from .stacked_timeline_widget import StackedTimelineWidget
from .training_thread import TrainingThread

_CLICK_THRESHOLD = 20


class CentralWidget(QtWidgets.QWidget):
    """QT Widget implementing our main window contents"""

    export_training_status_change = QtCore.Signal(bool)
    status_message = QtCore.Signal(str, int)  # message, timeout (ms)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # timeline widgets
        self._stacked_timeline = StackedTimelineWidget(self)

        # video player
        self._player_widget = PlayerWidget(self)
        self._player_widget.updateFrameNumber.connect(self._frame_change)
        self._player_widget.updateFrameNumber.connect(self._stacked_timeline.set_current_frame)
        self._player_widget.pixmap_clicked.connect(self._pixmap_clicked)
        self._curr_frame_index = 0

        self._loaded_video = None
        self._project = None
        self._labels = None
        self._pose_est = None
        self._label_overlay_mode = PlayerWidget.LabelOverlay.NONE
        self._suppress_label_track_update = False

        #  classifier
        self._classifier = Classifier(n_jobs=-1)
        self._training_thread = None
        self._classify_thread = None

        # information about current predictions
        self._predictions = {}
        self._probabilities = {}
        self._frame_indexes = {}

        self._selection_start = 0

        # options
        self._frame_jump = 10
        self._window_size = jabs.feature_extraction.DEFAULT_WINDOW_SIZE

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
        self._controls.kfold_changed.connect(self._set_train_button_enabled_state)
        self._controls.window_size_changed.connect(self._window_feature_size_changed)
        self._controls.new_window_sizes.connect(self._save_window_sizes)
        self._controls.use_balance_labels_changed.connect(self._use_balance_labels_changed)
        self._controls.use_symmetric_changed.connect(self._use_symmetric_changed)

        # main layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self._player_widget, 0, 0)
        layout.addWidget(self._controls, 0, 1, 2, 1)
        layout.addWidget(self._stacked_timeline, 1, 0)

        # set row stretch to allow player to expand vertically but not other rows
        layout.setRowStretch(0, 1)  # Player row expands
        layout.setRowStretch(1, 0)  # Label overview

        self.setLayout(layout)

        # progress bar dialog used when running the training or classify threads
        self._progress_dialog = None

        self._counts = None

        # set focus policy of all children widgets, needed to keep controls
        # from grabbing focus on Windows (which breaks arrow key video nav)
        for child in self.findChildren(QtWidgets.QWidget):
            child.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def eventFilter(self, source, event):
        """filter events emitted by progress dialog

        The main purpose of this is to prevent the progress dialog from closing if the user presses the escape key.
        """
        if source == self._progress_dialog and (
            event.type() == QtCore.QEvent.Type.Close
            or (
                event.type() == QtCore.QEvent.Type.KeyPress
                and isinstance(event, QtGui.QKeyEvent)
                and event.key() == Qt.Key.Key_Escape
            )
        ):
            event.accept()
            return True
        return super().eventFilter(source, event)

    @property
    def behavior(self):
        """get the currently selected behavior"""
        return self._controls.current_behavior

    @property
    def classifier_type(self):
        """get the current classifier type"""
        return self._classifier.classifier_type

    @property
    def window_size(self):
        """get current window size"""
        return self._window_size

    @property
    def uses_balance(self):
        """return true if the controls widget is set to use balanced labels, false otherwise"""
        return self._controls.use_balance_labels

    @property
    def uses_symmetric(self):
        """return true if the controls widget is set to use symmetric behavior, false otherwise"""
        return self._controls.use_symmetric

    @property
    def all_kfold(self):
        """return true if all kfold is selected in the controls widget, false otherwise"""
        return self._controls.all_kfold

    @property
    def classify_button_enabled(self):
        """return true if the classify button is currently enabled, false otherwise"""
        return self._controls.classify_button_enabled

    @property
    def behaviors(self):
        """return the behaviors from the controls widget"""
        return self._controls.behaviors

    @property
    def label_overlay_mode(self) -> PlayerWidget.LabelOverlay:
        """return the current label overlay mode of the player widget"""
        return self._label_overlay_mode

    @label_overlay_mode.setter
    def label_overlay_mode(self, mode: PlayerWidget.LabelOverlay) -> None:
        """set the label overlay mode of the player widget

        If the mode is changed, update the player widget labels with
        either the current labels or predictions based on the mode.

        Args:
            mode (PlayerWidget.LabelOverlay): The new label overlay mode to set.
        """
        if mode != self._label_overlay_mode:
            self._label_overlay_mode = mode
            # also update self._player_widget labels
            if mode == PlayerWidget.LabelOverlay.LABEL:
                self._player_widget.set_labels(
                    [labels.get_labels() for labels in self._get_label_list()]
                )
            elif mode == PlayerWidget.LabelOverlay.PREDICTION:
                prediction_list, _ = self._get_prediction_list()
                self._player_widget.set_labels(prediction_list)
            else:
                # if the player is set to show nothing, clear the labels
                self._player_widget.set_labels(None)

    def set_project(self, project):
        """set the currently opened project"""
        self._project = project

        # This will get set when the first video in the project is loaded, but
        # we need to set it to None so that we don't try to cache the current
        # labels when we do so (the current labels belong to the previous
        # project)
        self._labels = None
        self._loaded_video = None

        self._controls.update_project_settings(project.settings)

    def load_video(self, path):
        """load a new video file into self._player_widget

        Args:
            path: path to video file

        Returns:
            None
        """
        self._suppress_label_track_update = True
        if self._labels is not None:
            self._start_selection(False)
            self._controls.select_button_set_checked(False)

        try:
            self._loaded_video = path

            # open poses and any labels that might exist for this video
            self._pose_est = self._project.load_pose_est(path)
            self._labels = self._project.video_manager.load_video_labels(path)
            self._stacked_timeline.num_identities = self._pose_est.num_identities
            self._stacked_timeline.num_frames = self._pose_est.num_frames

            # if no saved labels exist, initialize a new VideoLabels object
            if self._labels is None:
                nframes = get_frame_count(str(path))
                self._labels = VideoLabels(path.name, nframes, self._pose_est.external_identities)

            # load saved predictions for this video
            self._predictions, self._probabilities, self._frame_indexes = (
                self._project.prediction_manager.load_predictions(path.name, self.behavior)
            )

            # load video into player
            self._player_widget.load_video(path, self._pose_est)

            # update ui components with properties of new video
            if self._pose_est.external_identities:
                self._set_identities(self._pose_est.external_identities)
            else:
                self._set_identities(self._pose_est.identities)

            self._stacked_timeline.framerate = self._player_widget.stream_fps()
            self._suppress_label_track_update = False
            self._set_label_track()
            self._update_select_button_state()

        except OSError as e:
            # error loading
            self._labels = None
            self._loaded_video = None
            self._pose_est = None
            self._set_identities([])
            self._player_widget.reset()
            raise e

    def keyPressEvent(self, event):
        """handle key press events"""

        def begin_select_mode():
            if (
                not self._controls.select_button_is_checked
                and self._controls.select_button_enabled
            ):
                self._controls.toggle_select_button()
                self._start_selection(True)

        key = event.key()
        shift_pressed = event.modifiers() & Qt.KeyboardModifier.ShiftModifier

        if key == QtCore.Qt.Key.Key_Left:
            self._player_widget.previous_frame()
        elif key == QtCore.Qt.Key.Key_Right:
            self._player_widget.next_frame()
        elif key == QtCore.Qt.Key.Key_Up:
            if shift_pressed:
                self._increment_identity_index()
            else:
                self._player_widget.next_frame(self._frame_jump)
        elif key == QtCore.Qt.Key.Key_Down:
            if shift_pressed:
                self._decrement_identity_index()
            else:
                self._player_widget.previous_frame(self._frame_jump)
        elif key == QtCore.Qt.Key.Key_Space:
            self._player_widget.toggle_play()
        elif key == QtCore.Qt.Key.Key_Z:
            if self._controls.select_button_is_checked:
                self._label_behavior()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key.Key_X:
            if self._controls.select_button_is_checked:
                self._clear_behavior_label()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key.Key_C:
            if self._controls.select_button_is_checked:
                self._label_not_behavior()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key.Key_Escape:
            if self._controls.select_button_is_checked:
                self._controls.select_button_set_checked(False)
                self._start_selection(False)
        elif key == QtCore.Qt.Key.Key_Question:
            # show closest with no argument toggles the setting
            self._player_widget.show_closest()

    def show_track(self, show: bool):
        """set the show track property of the player widget"""
        self._player_widget.show_track(show)

    def overlay_pose(self, checked: bool):
        """set the overlay pose property of the player widget"""
        self._player_widget.overlay_pose(checked)

    def overlay_landmarks(self, checked: bool):
        """set the overlay landmarks property of the player widget"""
        self._player_widget.overlay_landmarks(checked)

    def overlay_segmentation(self, checked: bool):
        """set the overlay segmentation property of the player widget"""
        self._player_widget.overlay_segmentation(checked)

    def remove_behavior(self, behavior: str):
        """remove a behavior from the list of behaviors"""
        self._controls.remove_behavior(behavior)

    @property
    def controls(self):
        """return the controls widget"""
        return self._controls

    @property
    def timeline_view_mode(self):
        """return the timeline view mode"""
        return self._stacked_timeline.view_mode

    @timeline_view_mode.setter
    def timeline_view_mode(self, view_mode: StackedTimelineWidget.ViewMode):
        """set the timeline view mode"""
        self._stacked_timeline.view_mode = view_mode
        self._update_select_button_state()

    @property
    def timeline_identity_mode(self):
        """return the timeline identity mode"""
        return self._stacked_timeline.identity_mode

    @timeline_identity_mode.setter
    def timeline_identity_mode(self, identity_mode: StackedTimelineWidget.IdentityMode):
        """set the timeline view mode"""
        self._stacked_timeline.identity_mode = identity_mode

    def _change_behavior(self, new_behavior):
        """make UI changes to reflect the currently selected behavior"""
        if self._project is None:
            return

        # load up settings for new behavior
        self._update_controls_from_project_settings()
        self._load_cached_classifier()

        # get label/bout counts for the current project
        self._counts = self._project.counts(self.behavior)
        self._update_label_counts()

        # load saved predictions
        if self._loaded_video:
            self._predictions, self._probabilities, self._frame_indexes = (
                self._project.prediction_manager.load_predictions(
                    self._loaded_video.name, self.behavior
                )
            )

        # display labels and predictions for new behavior
        self._set_label_track()
        self._set_train_button_enabled_state()

        self._project.settings_manager.save_project_file({"selected_behavior": self.behavior})

    def _start_selection(self, pressed):
        """Handle a click on "select" button.

        If button was previously "unchecked" then enter "select mode". If the button was in the checked state,
        clicking cancels the current selection. While selection is in progress, the labeling buttons become active.
        """
        if pressed:
            self._controls.enable_label_buttons()
            self._selection_start = self._player_widget.current_frame()
            self._stacked_timeline.start_selection(self._selection_start)
        else:
            self._controls.disable_label_buttons()
            self._stacked_timeline.clear_selection()

    def _label_behavior(self):
        """Apply behavior label to currently selected range of frames"""
        start, end = sorted([self._selection_start, self._player_widget.current_frame()])
        mask = self._pose_est.identity_mask(self._controls.current_identity_index)
        self._get_label_track().label_behavior(start, end, mask[start : end + 1])
        self._label_button_common()

    def _label_not_behavior(self):
        """apply _not_ behavior label to currently selected range of frames"""
        start, end = sorted([self._selection_start, self._player_widget.current_frame()])
        mask = self._pose_est.identity_mask(self._controls.current_identity_index)
        self._get_label_track().label_not_behavior(start, end, mask[start : end + 1])
        self._label_button_common()

    def _clear_behavior_label(self):
        """clear all behavior/not behavior labels from current selection"""
        label_range = sorted([self._selection_start, self._player_widget.current_frame()])
        self._get_label_track().clear_labels(*label_range)
        self._label_button_common()

    def _label_button_common(self) -> None:
        """common label button functionality

        functionality shared between _label_behavior(), _label_not_behavior(),
        and _clear_behavior_label(). To be called after the labels are changed
        for the current selection.
        """
        self._project.save_annotations(self._labels)
        self._controls.disable_label_buttons()
        self._stacked_timeline.clear_selection()
        self._update_label_counts()
        self._set_train_button_enabled_state()
        self._player_widget.reload_frame()

    def _set_identities(self, identities: list) -> None:
        """populate the identity_selection combobox"""
        self._controls.set_identities(identities)

    def _change_identity(self) -> None:
        """handle changing value of identity_selection"""
        self._player_widget.set_active_identity(self._controls.current_identity_index)
        self._update_label_counts()
        self._stacked_timeline.active_identity_index = self._controls.current_identity_index

    def _frame_change(self, new_frame: int) -> None:
        """called when the video player widget emits its updateFrameNumber signal"""
        self._curr_frame_index = new_frame

    def _set_label_track(self) -> None:
        """loads new set of labels in self.manual_labels when the selected behavior or identity is changed"""
        if self._suppress_label_track_update:
            return

        behavior = self._controls.current_behavior
        identity = self._controls.current_identity_index

        if identity != -1 and behavior != "" and self._labels is not None:
            label_list = self._get_label_list()
            mask_list = [
                self._pose_est.identity_mask(i) for i in range(self._pose_est.num_identities)
            ]
            self._stacked_timeline.set_labels(label_list, mask_list)

            if self._label_overlay_mode == PlayerWidget.LabelOverlay.LABEL:
                # if configured to show labels, update the player widget with the new labels
                self._player_widget.set_labels([labels.get_labels() for labels in label_list])

        self._set_prediction_vis()

    def _get_label_list(self):
        """get a list of np.ndarray containing labels, one for each identity"""
        behavior = self._controls.current_behavior
        identity = self._controls.current_identity_index
        if identity != -1 and behavior != "" and self._labels is not None:
            return [
                self._labels.get_track_labels(str(i), behavior)
                for i in range(self._pose_est.num_identities)
            ]
        return []

    def _get_label_track(self) -> TrackLabels:
        """get the current label track for the currently selected identity and behavior"""
        return self._labels.get_track_labels(
            str(self._controls.current_identity_index), self._controls.current_behavior
        )

    def _update_classifier_controls(self) -> None:
        """Called when settings related to a loaded classifier should be updated"""
        self._controls.set_classifier_selection(self._classifier.classifier_type)

        # does the classifier match the current settings?
        classifier_settings = self._classifier.project_settings
        if (
            classifier_settings is not None
            and classifier_settings.get("window_size", None) == self.window_size
            and classifier_settings.get("balance_labels", None)
            == self._controls.use_balance_labels
            and classifier_settings.get("symmetric_behavior", None) == self._controls.use_symmetric
        ):
            # if yes, we can enable the classify button
            self._controls.classify_button_set_enabled(True)
        else:
            # if not, the classify button needs to be disabled until the
            # user retrains
            self._controls.classify_button_set_enabled(False)

    def _train_button_clicked(self) -> None:
        """handle user click on "Train" button"""
        # make sure video playback is stopped
        self._player_widget.stop()

        # setup training thread
        self._training_thread = TrainingThread(
            self._project,
            self._classifier,
            self._controls.current_behavior,
            np.inf if self._controls.all_kfold else self._controls.kfold_value,
            parent=self,
        )
        self._training_thread.training_complete.connect(self._training_thread_complete)
        self._training_thread.update_progress.connect(self._update_training_progress)
        self._training_thread.current_status.connect(lambda m: self.status_message.emit(m, 0))

        # setup progress dialog
        # adds 2 for final training
        total_steps = self._project.total_project_identities + 2
        if self._controls.all_kfold:
            project_counts = self._project.counts(self._controls.current_behavior)
            total_steps += self._classifier.count_label_threshold(project_counts)
        else:
            total_steps += self._controls.kfold_value
        self._progress_dialog = QtWidgets.QProgressDialog("Training", None, 0, total_steps, self)
        self._progress_dialog.installEventFilter(self)
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress_dialog.reset()
        self._progress_dialog.show()

        # start training thread
        self._training_thread.start()

    def _training_thread_complete(self) -> None:
        """enable classify button once the training is complete"""
        self._progress_dialog.reset()
        self.status_message.emit("Training Complete", 3000)
        self._controls.classify_button_set_enabled(True)

    def _update_training_progress(self, step: int) -> None:
        """update progress bar with the number of completed tasks"""
        self._progress_dialog.setValue(step)

    def _classify_button_clicked(self) -> None:
        """handle user click on "Classify" button"""
        # make sure video playback is stopped
        self._player_widget.stop()

        # setup classification thread
        self._classify_thread = ClassifyThread(
            self._classifier,
            self._project,
            self._controls.current_behavior,
            self._loaded_video.name,
            parent=self,
        )
        self._classify_thread.done.connect(self._classify_thread_complete)
        self._classify_thread.update_progress.connect(self._update_classify_progress)
        self._classify_thread.current_status.connect(lambda m: self.status_message.emit(m, 0))

        # setup progress dialog
        self._progress_dialog = QtWidgets.QProgressDialog(
            "Predicting", None, 0, self._project.total_project_identities + 1, self
        )
        self._progress_dialog.installEventFilter(self)
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress_dialog.reset()
        self._progress_dialog.show()

        # start classification thread
        self._classify_thread.start()

    def _classify_thread_complete(self, output: dict) -> None:
        """update the gui when the classification is complete"""
        # display the new predictions
        self._predictions = output["predictions"]
        self._probabilities = output["probabilities"]
        self._frame_indexes = output["frame_indexes"]
        self.status_message.emit("Classification Complete", 3000)
        self._set_prediction_vis()

    def _update_classify_progress(self, step: int) -> None:
        """update progress bar with the number of completed tasks"""
        self._progress_dialog.setValue(step)

    def _set_prediction_vis(self) -> None:
        """update data being displayed by the prediction visualization widget"""
        if self._loaded_video is None:
            return

        prediction_list, probability_list = self._get_prediction_list()
        self._stacked_timeline.set_predictions(prediction_list, probability_list)
        if self._label_overlay_mode == PlayerWidget.LabelOverlay.PREDICTION:
            # if the player is set to show predictions, update the player widget
            self._player_widget.set_labels(prediction_list)

    def _get_prediction_list(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """get the prediction and probability list for each identity in the current video"""
        prediction_list = []
        probability_list = []

        for i in range(self._pose_est.num_identities):
            prediction_labels = np.full(
                (self._player_widget.num_frames()),
                TrackLabels.Label.NONE.value,
                dtype=np.byte,
            )

            prediction_prob = np.zeros((self._player_widget.num_frames()), dtype=np.float64)

            try:
                indexes = self._frame_indexes[i]
            except KeyError:
                prediction_list.append(prediction_labels)
                probability_list.append(prediction_prob)
                continue

            prediction_labels[indexes] = self._predictions[i][indexes]
            prediction_prob[indexes] = self._probabilities[i][indexes]
            prediction_list.append(prediction_labels)
            probability_list.append(prediction_prob)
        return prediction_list, probability_list

    def _set_train_button_enabled_state(self) -> None:
        """set the enabled property of the train button

        Sets enabled state of the train button to True or False depending on
        whether the labeling meets some threshold set by the classifier module

        NOTE: must be called after _update_label_counts() so that it has the
        correct counts for the current video

        Returns:
            None
        """
        # kfold slider is enabled before a project is loaded and moving it will trigger this function
        # in that case we don't want to do anything
        if self._project is None:
            return

        if Classifier.label_threshold_met(self._counts, self._controls.kfold_value):
            self._controls.train_button_enabled = True
            self.export_training_status_change.emit(True)
        else:
            self._controls.train_button_enabled = False
            self.export_training_status_change.emit(False)

    def _update_label_counts(self) -> None:
        """update the widget with the labeled frame / bout counts

        Returns:
            None
        """
        if self._loaded_video is None:
            return

        # update counts for the current video -- we could be more efficient
        # by only updating the current identity in the current video
        self._counts[self._loaded_video.name] = self._labels.counts(self.behavior)

        # TODO fix so we're not using the identity index as a string for keys in the label counts
        identity = str(self._controls.current_identity_index)

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

        self._controls.set_frame_counts(
            label_behavior_current,
            label_not_behavior_current,
            label_behavior_project,
            label_not_behavior_project,
            bout_behavior_current,
            bout_not_behavior_current,
            bout_behavior_project,
            bout_not_behavior_project,
        )

    def _classifier_changed(self) -> None:
        """handle classifier selection change"""
        if self._classifier.classifier_type != self._controls.classifier_type:
            # changing classifier type, disable until retrained
            self._controls.classify_button_set_enabled(False)
            self._classifier.set_classifier(self._controls.classifier_type)

    def _pixmap_clicked(self, event: dict[str, int]) -> None:
        """handle event where user clicked on the video

        if user clicks on one of the mice, make that one active
        """
        clicked_identity = None
        if self._pose_est is not None:
            pt = Point(event["x"], event["y"])
            for i, ident in enumerate(self._pose_est.identities):
                c_hulls = self._pose_est.get_identity_convex_hulls(ident)
                curr_c_hull = c_hulls[self._curr_frame_index]

                # if the click is in a convex hull, set that identity as active
                if curr_c_hull is not None and curr_c_hull.contains(pt):
                    clicked_identity = i
                    break

            # if the click was not on a convex hull, check to see if it was close to one
            # with few keypoints, sometimes the convex hull is thin and easy to miss when clicking on the mouse
            min_distance = float("inf")
            closest_identity = None
            for i, ident in enumerate(self._pose_est.identities):
                c_hulls = self._pose_est.get_identity_convex_hulls(ident)
                curr_c_hull = c_hulls[self._curr_frame_index]

                # if the click is in a convex hull, set that identity as active
                if curr_c_hull is not None and curr_c_hull.distance(pt) < min_distance:
                    closest_identity = i
                    min_distance = curr_c_hull.distance(pt)

            if clicked_identity is None and min_distance < _CLICK_THRESHOLD:
                clicked_identity = closest_identity

        if clicked_identity is not None:
            self._controls.set_identity_index(clicked_identity)

    def _window_feature_size_changed(self, new_size: int) -> None:
        """handle window feature size change"""
        if new_size is not None and new_size != self._window_size:
            self._window_size = new_size
            self.update_behavior_settings("window_size", new_size)
            self._update_classifier_controls()

    def _save_window_sizes(self, window_sizes: list[int]) -> None:
        """save the window sizes to the project settings"""
        self._project.settings_manager.save_project_file({"window_sizes": window_sizes})

    def update_behavior_settings(self, key: str, val: any) -> None:
        """propagates an updated setting to the project"""
        # early exit if no behavior selected
        if self.behavior == "":
            return

        self._project.settings_manager.save_behavior(self.behavior, {key: val})

    def _use_balance_labels_changed(self) -> None:
        if self.behavior == "":
            # don't do anything if behavior is not set, this means we're
            # the project isn't fully loaded and we just reset the
            # checkbox
            return

        self.update_behavior_settings("balance_labels", self._controls.use_balance_labels)
        self._update_classifier_controls()

    def _use_symmetric_changed(self) -> None:
        if self.behavior == "":
            # Copy behavior of use_balance_labels_changed
            return

        self.update_behavior_settings("symmetric_behavior", self._controls.use_symmetric)
        self._update_classifier_controls()

    def _update_controls_from_project_settings(self) -> None:
        if self._project is None or self.behavior is None:
            return

        behavior_metadata = self._project.settings_manager.get_behavior(self.behavior)
        self._controls.set_window_size(behavior_metadata["window_size"])
        self._controls.use_balance_labels = behavior_metadata["balance_labels"]
        self._controls.use_symmetric = behavior_metadata["symmetric_behavior"]

    def _load_cached_classifier(self) -> None:
        classifier_loaded = False
        try:
            classifier_loaded = self._project.load_classifier(self._classifier, self.behavior)
        except Exception as e:
            print("failed to load classifier:", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            print("classifier will need to be retrained")

        if classifier_loaded:
            self._update_classifier_controls()
        else:
            self._controls.classify_button_set_enabled(False)

    def _increment_identity_index(self) -> None:
        """Increment the identity selection index, rolling over if necessary."""
        if self._pose_est is None:
            return

        num_identities = self._pose_est.num_identities
        if num_identities == 0:
            return

        current_index = self._controls.current_identity_index
        next_index = (current_index + 1) % num_identities
        self._controls.set_identity_index(next_index)

    def _decrement_identity_index(self) -> None:
        """Decrement the identity selection index, rolling over if necessary."""
        if self._pose_est is None:
            return

        num_identities = self._pose_est.num_identities
        if num_identities == 0:
            return

        current_index = self._controls.current_identity_index
        prev_index = (current_index - 1) % num_identities
        self._controls.set_identity_index(prev_index)

    def _update_select_button_state(self) -> None:
        """Update the state of the select button based on multiple factors.

        Considers if a video is loaded, if the video has identities to
        label, and if the current view mode is appropriate for selection
        (i.e. is the label timeline being displayed).
        """

        def disable_select_button() -> None:
            """Disable the select button and uncheck it."""
            self._start_selection(False)
            self._controls.select_button_enabled = False
            self._controls.select_button_set_checked(False)
            self._stacked_timeline.clear_selection()

        # if no video is loaded, or if there are no identities, disable the select button
        if self._loaded_video is None or (
            self._pose_est is not None and self._pose_est.num_identities == 0
        ):
            disable_select_button()

        if self._stacked_timeline.view_mode == self._stacked_timeline.view_mode.PREDICTIONS:
            # If the view mode is predictions, disable the select button because there is
            # no way for the user to see what they are labeling
            disable_select_button()
        else:
            self._controls.select_button_enabled = True
