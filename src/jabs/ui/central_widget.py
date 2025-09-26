import sys
import traceback
from pathlib import Path
from typing import cast

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog
from shapely.geometry import Point

import jabs.feature_extraction
from jabs.behavior_search import SearchHit
from jabs.classifier import Classifier
from jabs.pose_estimation import PoseEstimation, PoseEstimationV8
from jabs.project import Project, TimelineAnnotations, TrackLabels, VideoLabels
from jabs.types import ClassifierType
from jabs.ui.search_bar_widget import SearchBarWidget

from .annotation_edit_dialog import AnnotationEditDialog
from .classification_thread import ClassifyThread
from .exceptions import ThreadTerminatedError
from .main_control_widget import MainControlWidget
from .player_widget import PlayerWidget
from .progress_dialog import create_cancelable_progress_dialog
from .stacked_timeline_widget import StackedTimelineWidget
from .training_thread import TrainingThread

_CLICK_THRESHOLD = 20
_DEBOUNCE_SEARCH_DELAY_MS = 100


class CentralWidget(QtWidgets.QWidget):
    """QT Widget implementing our main window contents"""

    export_training_status_change = QtCore.Signal(bool)
    status_message = QtCore.Signal(str, int)  # message, timeout (ms)
    search_hit_loaded = QtCore.Signal(SearchHit)
    bbox_overlay_supported = QtCore.Signal(bool)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # behavior search bar
        self._search_bar_widget = SearchBarWidget(self)
        self._search_bar_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._search_bar_widget.current_search_hit_changed.connect(
            self._on_search_hit_changed_later
        )
        self._search_bar_widget.search_results_changed.connect(
            lambda _: self._update_timeline_search_results()
        )
        self._debounce_search_hit_timer = QtCore.QTimer(self)
        self._debounce_search_hit_timer.setSingleShot(True)
        self._debounce_search_hit_timer.setInterval(_DEBOUNCE_SEARCH_DELAY_MS)
        self._debounce_search_hit_timer.timeout.connect(self._on_search_hit_changed)

        # timeline widgets
        self._stacked_timeline = StackedTimelineWidget(self)

        # video player
        self._player_widget = PlayerWidget(self)
        self._player_widget.update_frame_number.connect(self._on_frame_changed)
        self._player_widget.update_frame_number.connect(self._stacked_timeline.set_current_frame)
        self._player_widget.pixmap_clicked.connect(self._on_pixmap_clicked)
        self._player_widget.id_label_clicked.connect(self._on_id_label_clicked)
        self._curr_frame_index = 0

        self._loaded_video = None
        self._project: Project | None = None
        self._labels: VideoLabels | None = None
        self._prediction_list = None
        self._probability_list = None
        self._pose_est: PoseEstimation | None = None
        self._label_overlay_mode = PlayerWidget.LabelOverlayMode.NONE
        self._suppress_label_track_update = False

        #  classifier
        self._classifier = Classifier(n_jobs=-1)
        self._training_thread: TrainingThread | None = None
        self._classify_thread: ClassifyThread | None = None

        # information about current predictions
        self._predictions = {}
        self._probabilities = {}
        self._frame_indexes = {}

        self._selection_start = None
        self._selection_end = None

        # options
        self._frame_jump = 10
        self._window_size = jabs.feature_extraction.DEFAULT_WINDOW_SIZE

        # main controls
        self._controls = MainControlWidget()
        self._controls.identity_changed.connect(self._on_identity_changed)
        self._controls.label_behavior_clicked.connect(self._label_behavior)
        self._controls.label_not_behavior_clicked.connect(self._label_not_behavior)
        self._controls.clear_label_clicked.connect(self._clear_behavior_label)
        self._controls.start_selection.connect(self._start_selection)
        self._controls.train_clicked.connect(self._train_button_clicked)
        self._controls.classify_clicked.connect(self._classify_button_clicked)
        self._controls.classifier_changed.connect(self._classifier_changed)
        self._controls.behavior_changed.connect(self._on_behavior_changed)
        self._controls.kfold_changed.connect(self._set_train_button_enabled_state)
        self._controls.window_size_changed.connect(self._on_window_size_changed)
        self._controls.new_window_sizes.connect(self._save_window_sizes)
        self._controls.use_balance_labels_changed.connect(self._on_use_balance_labels_changed)
        self._controls.use_symmetric_changed.connect(self._on_use_symmetric_changed)
        self._controls.timeline_annotation_button_clicked.connect(
            self._on_timeline_annotation_button_clicked
        )

        # main grid layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self._player_widget, 0, 0)
        layout.addWidget(self._controls, 0, 1, 2, 1)
        layout.addWidget(self._stacked_timeline, 1, 0)

        # set row stretch to allow player to expand vertically but not other rows
        layout.setRowStretch(0, 1)  # Player row expands
        layout.setRowStretch(1, 0)  # Label overview

        # container for the grid layout
        grid_container = QtWidgets.QWidget()
        grid_container.setLayout(layout)

        # main vertical layout to hold the search bar and grid container
        main_vbox = QtWidgets.QVBoxLayout()
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(0)
        main_vbox.addWidget(self._search_bar_widget)
        main_vbox.addWidget(grid_container, 1)

        self.setLayout(main_vbox)

        # progress bar dialog used when running the training or classify threads
        self._progress_dialog = None

        self._counts = None

        # set focus policy of all children widgets, needed to keep controls
        # from grabbing focus on Windows (which breaks arrow key video nav)
        for child in self.findChildren(QtWidgets.QWidget):
            child.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def update_behavior_search_query(self, search_query) -> None:
        """Update the search query for the search bar widget"""
        self._search_bar_widget.update_search(search_query)

    @property
    def loaded_video(self) -> Path | None:
        """get the currently loaded video path"""
        return self._loaded_video

    @property
    def overlay_annotations_enabled(self) -> bool:
        """get the annotation overlay enabled status from player widget."""
        return self._player_widget.overlay_annotations_enabled

    @overlay_annotations_enabled.setter
    def overlay_annotations_enabled(self, enabled: bool) -> None:
        """Set the annotation overlay enabled status on the player widget."""
        self._player_widget.overlay_annotations_enabled = enabled

    @property
    def behavior(self) -> str:
        """get the currently selected behavior"""
        return self._controls.current_behavior

    @property
    def classifier_type(self) -> ClassifierType:
        """get the current classifier type"""
        return self._classifier.classifier_type

    @property
    def window_size(self) -> int:
        """get current window size"""
        return self._window_size

    @property
    def uses_balance(self) -> bool:
        """return true if the controls widget is set to use balanced labels, false otherwise"""
        return self._controls.use_balance_labels

    @property
    def uses_symmetric(self) -> bool:
        """return true if the controls widget is set to use symmetric behavior, false otherwise"""
        return self._controls.use_symmetric

    @property
    def all_kfold(self) -> bool:
        """return true if all kfold is selected in the controls widget, false otherwise"""
        return self._controls.all_kfold

    @property
    def classify_button_enabled(self) -> bool:
        """return true if the classify button is currently enabled, false otherwise"""
        return self._controls.classify_button_enabled

    @property
    def behaviors(self) -> list[str]:
        """return the behaviors from the controls widget"""
        return self._controls.behaviors

    @property
    def search_results_changed(self) -> QtCore.Signal:
        """Signal emitted when search results change."""
        return self._search_bar_widget.search_results_changed

    def _update_timeline_search_results(self) -> None:
        """Update the timeline with search results."""
        search_results = self._search_bar_widget.search_results
        behavior_search_query = self._search_bar_widget.behavior_search_query
        video_name = self._loaded_video.name if self._loaded_video else None
        if not self._loaded_video or not video_name:
            self._stacked_timeline.set_search_results(None, [])
        else:
            # Filter search results for the currently loaded video
            filtered_results = [hit for hit in search_results if hit.file == video_name]
            self._stacked_timeline.set_search_results(behavior_search_query, filtered_results)

    @property
    def label_overlay_mode(self) -> PlayerWidget.LabelOverlayMode:
        """return the current label overlay mode of the player widget"""
        return self._label_overlay_mode

    @label_overlay_mode.setter
    def label_overlay_mode(self, mode: PlayerWidget.LabelOverlayMode) -> None:
        """set the label overlay mode of the player widget

        If the mode is changed, update the player widget labels with
        either the current labels or predictions based on the mode.

        Args:
            mode (PlayerWidget.LabelOverlayMode): The new label overlay mode to set.
        """
        if mode != self._label_overlay_mode:
            self._label_overlay_mode = mode
            # also update self._player_widget labels
            if mode == PlayerWidget.LabelOverlayMode.LABEL:
                self._player_widget.set_labels(
                    [labels.get_labels() for labels in self._get_label_list()]
                )
            elif mode == PlayerWidget.LabelOverlayMode.PREDICTION:
                # prediction_list, _ = self._get_prediction_list()
                self._player_widget.set_labels(self._prediction_list)
            else:
                # if the player is set to show nothing, clear the labels
                self._player_widget.set_labels(None)

    @property
    def id_overlay_mode(self) -> PlayerWidget.IdentityOverlayMode:
        """return the current identity overlay mode of the player widget"""
        return self._player_widget.id_overlay_mode

    @id_overlay_mode.setter
    def id_overlay_mode(self, mode: PlayerWidget.IdentityOverlayMode) -> None:
        """set the identity overlay mode of the player widget

        Args:
            mode (PlayerWidget.IdentityOverlayMode): The new identity overlay mode to set.
        """
        self._player_widget.id_overlay_mode = mode

    def set_project(self, project: Project) -> None:
        """set the currently opened project"""
        self._project = project

        # This will get set when the first video in the project is loaded, but
        # we need to set it to None so that we don't try to cache the current
        # labels when we do so (the current labels belong to the previous
        # project)
        self._labels = None
        self._loaded_video = None

        self._controls.update_project_settings(project.settings)
        self._search_bar_widget.update_project(project)
        self._update_timeline_search_results()

    def load_video(self, path: Path) -> None:
        """load a new video file into self._player_widget

        Args:
            path: path to video file

        Returns:
            None
        """
        previous_video = self._loaded_video

        self._suppress_label_track_update = True
        self._search_bar_widget.video_frame_position_changed(path.name, 0)
        if self._labels is not None:
            self._start_selection(False)
            self._controls.select_button_set_checked(False)

        try:
            self._loaded_video = path

            # open poses and any labels that might exist for this video
            self._pose_est = self._project.load_pose_est(path)
            self._labels = self._project.video_manager.load_video_labels(path, self._pose_est)
            self._stacked_timeline.pose = self._pose_est

            # if no saved labels exist, initialize a new VideoLabels object
            if self._labels is None:
                self._labels = VideoLabels(path.name, self._pose_est.num_frames)

            # load saved predictions for this video
            self._predictions, self._probabilities, self._frame_indexes = (
                self._project.prediction_manager.load_predictions(path.name, self.behavior)
            )

            # load video into player
            self._player_widget.load_video(path, self._pose_est, self._labels)

            # update ui components with properties of new video
            display_identities = [
                self._pose_est.identity_index_to_display(i) for i in self._pose_est.identities
            ]
            self._set_identities(display_identities)
            self._player_widget.set_active_identity(self._controls.current_identity_index)

            self._stacked_timeline.framerate = self._player_widget.stream_fps
            self._suppress_label_track_update = False
            self._set_label_track()
            self._update_select_button_state()
            self._update_timeline_search_results()
            self._update_label_counts()

            # check if bbox overlay is supported by the pose file
            if self._pose_est.format_major_version < 8:
                self.bbox_overlay_supported.emit(False)
            else:
                # bboxes are optional in v8+, check if they exist. cast to a PoseEstimationV8 to access bbox methods
                pose_v8 = cast(PoseEstimationV8, self._pose_est)
                if pose_v8.has_bounding_boxes:
                    self.bbox_overlay_supported.emit(True)
                else:
                    self.bbox_overlay_supported.emit(False)

            if previous_video is not None:
                self._project.session_tracker.video_closed(previous_video)
            self._project.session_tracker.video_opened(path)

        except OSError as e:
            # error loading
            self._labels = None
            self._loaded_video = None
            self._pose_est = None
            self._set_identities([])
            self._player_widget.reset()
            raise e

    def keyPressEvent(self, event) -> None:
        """handle key press events"""

        def begin_select_mode() -> None:
            if (
                not self._controls.select_button_is_checked
                and self._controls.select_button_enabled
            ):
                self._controls.toggle_select_button()
                self._start_selection(True)

        key = event.key()
        shift_pressed = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        alt_pressed = event.modifiers() & Qt.KeyboardModifier.AltModifier

        match key:
            case QtCore.Qt.Key.Key_Left:
                self._player_widget.previous_frame()
            case QtCore.Qt.Key.Key_Right:
                self._player_widget.next_frame()
            case QtCore.Qt.Key.Key_Up:
                if shift_pressed:
                    self._increment_identity_index()
                else:
                    self._player_widget.next_frame(self._frame_jump)
            case QtCore.Qt.Key.Key_Down:
                if shift_pressed:
                    self._decrement_identity_index()
                else:
                    self._player_widget.previous_frame(self._frame_jump)
            case QtCore.Qt.Key.Key_Space:
                if alt_pressed:
                    self._play_current_bout(True)
                elif shift_pressed:
                    self._play_current_bout()
                else:
                    self._player_widget.toggle_play()
            case QtCore.Qt.Key.Key_Z:
                if self._controls.select_button_is_checked:
                    self._label_behavior()
                else:
                    begin_select_mode()
            case QtCore.Qt.Key.Key_X:
                if self._controls.select_button_is_checked:
                    self._clear_behavior_label()
                else:
                    begin_select_mode()
            case QtCore.Qt.Key.Key_C:
                if self._controls.select_button_is_checked:
                    self._label_not_behavior()
                else:
                    begin_select_mode()
            case QtCore.Qt.Key.Key_Escape:
                if self._controls.select_button_is_checked:
                    self._controls.select_button_set_checked(False)
                    self._start_selection(False)
            case QtCore.Qt.Key.Key_Question:
                self._player_widget.show_closest()
            case _:
                pass

    def show_track(self, show: bool) -> None:
        """set the show track property of the player widget"""
        self._player_widget.show_track(show)

    def overlay_pose(self, checked: bool) -> None:
        """set the overlay pose property of the player widget"""
        if checked:
            self._player_widget.pose_overlay_mode = PlayerWidget.PoseOverlayMode.ALL
        else:
            self._player_widget.pose_overlay_mode = PlayerWidget.PoseOverlayMode.NONE

    def overlay_landmarks(self, checked: bool) -> None:
        """set the overlay landmarks property of the player widget"""
        self._player_widget.overlay_landmarks(checked)

    def overlay_segmentation(self, checked: bool) -> None:
        """set the overlay segmentation property of the player widget"""
        self._player_widget.overlay_segmentation(checked)

    def remove_behavior(self, behavior: str) -> None:
        """remove a behavior from the list of behaviors"""
        self._controls.remove_behavior(behavior)

    @property
    def controls(self) -> MainControlWidget:
        """return the controls widget"""
        return self._controls

    @property
    def timeline_view_mode(self) -> StackedTimelineWidget.ViewMode:
        """return the timeline view mode"""
        return self._stacked_timeline.view_mode

    @timeline_view_mode.setter
    def timeline_view_mode(self, view_mode: StackedTimelineWidget.ViewMode) -> None:
        """set the timeline view mode"""
        self._stacked_timeline.view_mode = view_mode
        self._update_select_button_state()

    @property
    def timeline_identity_mode(self) -> StackedTimelineWidget.IdentityMode:
        """return the timeline identity mode"""
        return self._stacked_timeline.identity_mode

    @timeline_identity_mode.setter
    def timeline_identity_mode(self, identity_mode: StackedTimelineWidget.IdentityMode) -> None:
        """set the timeline view mode"""
        self._stacked_timeline.identity_mode = identity_mode

    def _on_behavior_changed(self) -> None:
        """make UI changes to reflect the currently selected behavior"""
        if self._project is None:
            return

        self._project.session_tracker.behavior_selected(self.behavior)

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

    def _start_selection(self, pressed: bool) -> None:
        """Handle a click on "select" button.

        If button was previously "unchecked" then enter "select mode". If the button was in the checked state,
        clicking cancels the current selection. While selection is in progress, the labeling buttons become active.
        """
        if pressed:
            self._controls.enable_label_buttons()
            self._selection_start = self._player_widget.current_frame
            self._selection_end = None
            self._stacked_timeline.start_selection(self._selection_start)
        else:
            self._controls.disable_label_buttons()
            self._stacked_timeline.clear_selection()

    def select_all(self) -> None:
        """Select all frames in the current video for the current identity and behavior."""
        if not self._controls.select_button_is_checked and self._controls.select_button_enabled:
            self._controls.toggle_select_button()

        if self._controls.select_button_enabled:
            num_frames = self._player_widget.num_frames
            if num_frames > 0:
                self._controls.enable_label_buttons()
                self._selection_start = 0
                self._selection_end = num_frames - 1
                self._stacked_timeline.start_selection(self._selection_start, self._selection_end)

    @property
    def _curr_selection_end(self) -> int:
        """Get the end of the current selection.

        If no selection end is set, return the current frame index.
        """
        return (
            self._selection_end
            if self._selection_end is not None
            else self._player_widget.current_frame
        )

    def _label_behavior(self) -> None:
        """Apply behavior label to currently selected range of frames"""
        start, end = sorted([self._selection_start, self._curr_selection_end])
        self._project.session_tracker.label_created(
            self._loaded_video,
            self._controls.current_identity_index,
            self._controls.current_behavior,
            True,
            start,
            end,
        )
        self._get_label_track().label_behavior(start, end)
        self._label_button_common()

    def _label_not_behavior(self) -> None:
        """apply _not_ behavior label to currently selected range of frames"""
        start, end = sorted([self._selection_start, self._curr_selection_end])
        self._project.session_tracker.label_created(
            self._loaded_video,
            self._controls.current_identity_index,
            self._controls.current_behavior,
            False,
            start,
            end,
        )
        self._get_label_track().label_not_behavior(start, end)
        self._label_button_common()

    def _clear_behavior_label(self) -> None:
        """clear all behavior/not behavior labels from current selection"""
        start, end = sorted([self._selection_start, self._curr_selection_end])
        self._project.session_tracker.label_deleted(
            self._loaded_video,
            self._controls.current_identity_index,
            self._controls.current_behavior,
            start,
            end,
        )
        self._get_label_track().clear_labels(start, end)
        self._label_button_common()

    def _label_button_common(self) -> None:
        """common label button functionality

        functionality shared between _label_behavior(), _label_not_behavior(),
        and _clear_behavior_label(). To be called after the labels are changed
        for the current selection.
        """
        self._project.save_annotations(self._labels, self._pose_est)
        self._controls.disable_label_buttons()
        self._stacked_timeline.clear_selection()
        self._update_label_counts()
        self._set_train_button_enabled_state()
        self._player_widget.reload_frame()

    def _set_identities(self, identities: list[str]) -> None:
        """populate the identity_selection combobox"""
        self._controls.set_identities(identities)

    def _on_identity_changed(self) -> None:
        """handle changing value of identity_selection"""
        self._player_widget.set_active_identity(self._controls.current_identity_index)
        self._update_label_counts()
        self._stacked_timeline.active_identity_index = self._controls.current_identity_index

    def _on_frame_changed(self, new_frame: int) -> None:
        """called when the video player widget emits its updateFrameNumber signal"""
        self._curr_frame_index = new_frame

        # if there is a pending "next" or "previous" search hit, we don't
        # update the search bar with the new frame
        if not self._debounce_search_hit_timer.isActive() and self._loaded_video is not None:
            self._search_bar_widget.video_frame_position_changed(
                self._loaded_video.name,
                new_frame,
            )

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

            if self._label_overlay_mode == PlayerWidget.LabelOverlayMode.LABEL:
                # if configured to show labels, update the player widget with the new labels
                self._player_widget.set_labels([labels.get_labels() for labels in label_list])

        self._set_prediction_vis()

    def _get_label_list(self) -> list[TrackLabels]:
        """get a list of TrackLabels, one for each identity"""
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
            self._controls.classify_button_enabled = True
        else:
            # if not, the classify button needs to be disabled until the
            # user retrains
            self._controls.classify_button_enabled = False

    def _train_button_clicked(self) -> None:
        """handle user click on "Train" button"""
        # make sure video playback is stopped
        self._player_widget.stop()

        # setup training thread
        self._training_thread = TrainingThread(
            self._classifier,
            self._project,
            self._controls.current_behavior,
            np.inf if self._controls.all_kfold else self._controls.kfold_value,
            parent=self,
        )
        self._training_thread.training_complete.connect(self._training_thread_complete)
        self._training_thread.error_callback.connect(self._training_thread_error_callback)
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
        self._progress_dialog = create_cancelable_progress_dialog(self, "Training", total_steps)
        self._progress_dialog.show()
        self._progress_dialog.canceled.connect(self._training_thread.request_termination)

        # start training thread
        self._training_thread.start()

    def _training_thread_complete(self) -> None:
        """enable classify button once the training is complete"""
        self._cleanup_training_thread()
        self._cleanup_progress_dialog()
        self.status_message.emit("Training Complete", 3000)
        self._controls.classify_button_enabled = True

    def _training_thread_error_callback(self, error: Exception) -> None:
        """handle an error in the training thread"""
        self._cleanup_training_thread()
        self._cleanup_progress_dialog()

        if isinstance(error, ThreadTerminatedError):
            self.status_message.emit("Training Canceled", 3000)
        else:
            self._print_exception(error)
            self.status_message.emit("Training Failed", 3000)
            QtWidgets.QMessageBox.critical(
                self, "Error", f"An exception occurred during training:\n{error}"
            )
            self._controls.classify_button_enabled = False

    def _classify_thread_error_callback(self, error: Exception) -> None:
        """handle an error in the classification thread"""
        self._cleanup_classify_thread()
        self._cleanup_progress_dialog()

        if isinstance(error, ThreadTerminatedError):
            self.status_message.emit("Classification Canceled", 3000)
        else:
            self._print_exception(error)
            self.status_message.emit("Classification Failed", 3000)
            QtWidgets.QMessageBox.critical(
                self, "Error", f"An exception occurred during classification:\n{error}"
            )

    @staticmethod
    def _print_exception(e: Exception) -> None:
        """Print a formatted traceback for the given exception to the terminal.

        This method outputs the full stack trace and exception details to help with debugging.
        It can be extended in the future to support additional logging or error handling mechanisms.

        Args:
            e (Exception): The exception instance to print.
        """
        traceback.print_exception(e)

    def _cleanup_progress_dialog(self) -> None:
        """clean up the progress dialog"""
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None

    def _cleanup_training_thread(self) -> None:
        """clean up the training thread"""
        if self._training_thread:
            self._training_thread.deleteLater()
            self._training_thread = None

    def _cleanup_classify_thread(self) -> None:
        """clean up the training thread"""
        if self._classify_thread:
            self._classify_thread.deleteLater()
            self._classify_thread = None

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
        self._classify_thread.classification_complete.connect(self._classify_thread_complete)
        self._classify_thread.error_callback.connect(self._classify_thread_error_callback)
        self._classify_thread.update_progress.connect(self._update_classify_progress)
        self._classify_thread.current_status.connect(lambda m: self.status_message.emit(m, 0))
        self._progress_dialog = create_cancelable_progress_dialog(
            self, "Predicting", self._project.total_project_identities + 1
        )
        self._progress_dialog.show()
        self._progress_dialog.canceled.connect(self._classify_thread.request_termination)

        # start classification thread
        self._classify_thread.start()

    def _classify_thread_complete(self, output: dict) -> None:
        """update the gui when the classification is complete"""
        # display the new predictions
        self._predictions = output["predictions"]
        self._probabilities = output["probabilities"]
        self._frame_indexes = output["frame_indexes"]
        self._cleanup_progress_dialog()
        self._cleanup_classify_thread()
        self.status_message.emit("Classification Complete", 3000)
        self._set_prediction_vis()

    def _update_classify_progress(self, step: int) -> None:
        """update progress bar with the number of completed tasks"""
        self._progress_dialog.setValue(step)

    def _set_prediction_vis(self) -> None:
        """update data being displayed by the prediction visualization widget"""
        if self._loaded_video is None:
            return

        self._prediction_list, self._probability_list = self._get_prediction_list()
        self._stacked_timeline.set_predictions(self._prediction_list, self._probability_list)
        if self._label_overlay_mode == PlayerWidget.LabelOverlayMode.PREDICTION:
            # if the player is set to show predictions, update the player widget
            self._player_widget.set_labels(self._prediction_list)

    def _get_prediction_list(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """get the prediction and probability list for each identity in the current video"""
        prediction_list = []
        probability_list = []

        for i in range(self._pose_est.num_identities):
            prediction_labels = np.full(
                self._player_widget.num_frames,
                TrackLabels.Label.NONE.value,
                dtype=np.byte,
            )

            prediction_prob = np.zeros(self._player_widget.num_frames, dtype=np.float64)

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

        # update counts for the current video
        self._counts[self._loaded_video.name] = self._project.load_counts(
            self._loaded_video.name, self.behavior
        )

        current_identity = self._controls.current_identity_index

        label_behavior_current = 0
        label_not_behavior_current = 0
        label_behavior_project = 0
        label_not_behavior_project = 0
        bout_behavior_current = 0
        bout_not_behavior_current = 0
        bout_behavior_project = 0
        bout_not_behavior_project = 0

        for video, video_counts in self._counts.items():
            for identity, counts in video_counts.items():
                label_behavior_project += counts["unfragmented_frame_counts"][0]
                label_not_behavior_project += counts["unfragmented_frame_counts"][1]
                bout_behavior_project += counts["unfragmented_bout_counts"][0]
                bout_not_behavior_project += counts["unfragmented_bout_counts"][1]

                if video == self._loaded_video.name and identity == current_identity:
                    label_behavior_current = counts["unfragmented_frame_counts"][0]
                    label_not_behavior_current = counts["unfragmented_frame_counts"][1]
                    bout_behavior_current = counts["unfragmented_bout_counts"][0]
                    bout_not_behavior_current = counts["unfragmented_bout_counts"][1]

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
            self._controls.classify_button_enabled = False
            self._classifier.set_classifier(self._controls.classifier_type)

    def _on_pixmap_clicked(self, event: dict[str, int]) -> None:
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

    def _on_id_label_clicked(self, id_clicked: int) -> None:
        """handle event where use clicked a floating identity label"""
        if self._pose_est is not None and id_clicked < self._pose_est.num_identities:
            self._controls.set_identity_index(id_clicked)

    def _on_window_size_changed(self, new_size: int) -> None:
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

    def _on_use_balance_labels_changed(self) -> None:
        if self.behavior == "":
            # don't do anything if behavior is not set, this means we're
            # the project isn't fully loaded and we just reset the
            # checkbox
            return

        self.update_behavior_settings("balance_labels", self._controls.use_balance_labels)
        self._update_classifier_controls()

    def _on_use_symmetric_changed(self) -> None:
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
            self._controls.classify_button_enabled = False

    def _on_search_hit_changed_later(self, _: SearchHit | None) -> None:
        """Update the search hit after a short delay to allow UI updates to complete."""
        self._debounce_search_hit_timer.start()

    def _on_search_hit_changed(self) -> None:
        """Handle updates when the current search hit changes."""
        search_hit = self._search_bar_widget.current_search_hit
        if search_hit is not None and self._project is not None:
            # load the video and seek to frame for the search hit
            video_to_load = self._project.video_manager.video_path(search_hit.file)
            if video_to_load != self._loaded_video:
                self.load_video(video_to_load)
            self._player_widget.seek_to_frame(search_hit.start_frame)

            # set the current identity based on the search hit
            selected_id = search_hit.identity
            if selected_id is not None:
                try:
                    selected_id = int(selected_id)
                except ValueError:
                    selected_id = None

            num_identities = self._pose_est.num_identities
            if selected_id is not None and selected_id < num_identities:
                self._controls.set_identity_index(selected_id)

            # update the behavior in the controls to match the search hit
            if search_hit.behavior is not None:
                self._controls.set_behavior(search_hit.behavior)

            self.search_hit_loaded.emit(search_hit)

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

        # disable select frames button if no video is loaded, there are
        # no identities to label, or the current view mode is predictions
        # only (which does not allow selection of frames)
        if (
            self._loaded_video is None
            or (self._pose_est is not None and self._pose_est.num_identities == 0)
            or self._stacked_timeline.view_mode == self._stacked_timeline.view_mode.PREDICTIONS
        ):
            disable_select_button()
        else:
            self._controls.select_button_enabled = True

    def _play_current_bout(self, use_predictions: bool = False) -> None:
        """Play the current bout: contiguous frames with the same label as the current frame.

        Args:
            use_predictions (bool): If True, use prediction labels instead of manual labels.
        """
        if self._labels is None or self._pose_est is None:
            return

        # stop if we are currently playing
        self._player_widget.stop()

        current_frame = self._player_widget.current_frame
        identity = self._controls.current_identity_index
        behavior = self._controls.current_behavior

        if identity == -1 or behavior == "":
            return

        if use_predictions:
            if not self._prediction_list or identity >= len(self._prediction_list):
                return
            labels = self._prediction_list[identity]
        else:
            track_labels = self._labels.get_track_labels(str(identity), behavior)
            labels = track_labels.get_labels()

        if labels is None or len(labels) == 0:
            return

        current_label = labels[current_frame]
        # Only play if current label is BEHAVIOR or NOT_BEHAVIOR
        if current_label not in (
            TrackLabels.Label.BEHAVIOR.value,
            TrackLabels.Label.NOT_BEHAVIOR.value,
        ):
            return

        # Find start of bout
        start = current_frame
        while start > 0 and labels[start - 1] == current_label:
            start -= 1

        # Find end of bout (inclusive)
        num_frames = len(labels)
        end = current_frame
        while end < num_frames and labels[end] == current_label:
            end += 1
        end -= 1

        self._player_widget.play_range(start, end)

    def _on_timeline_annotation_button_clicked(self) -> None:
        """Handle the event when the button to create a new timeline annotation is clicked.

        Note: start, end, tag value, and identity uniquely identify an annotation. If one already exists
        with the same values, do not create a duplicate and show a warning message instead.
        """
        identity_index = self._controls.current_identity_index
        display_identity = self._pose_est.identity_index_to_display(identity_index)
        start = min(self._selection_start, self._curr_frame_index)
        end = max(self._selection_start, self._curr_frame_index)

        dialog = AnnotationEditDialog(
            start,
            end,
            identity_index=identity_index,
            display_identity=display_identity,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.get_annotation()
            # result keys: tag, color, description, identity_scoped
            tag = result["tag"]
            identity_index = (
                self._controls.current_identity_index if result["identity_scoped"] else None
            )
            display_identity = (
                self._pose_est.identity_index_to_display(self._controls.current_identity_index)
                if identity_index is not None
                else None
            )

            # Duplicate check: look for an interval with the same (start, end, tag, identity_index)
            if self._labels.timeline_annotations.annotation_exists(
                start=start, end=end, tag=tag, identity_index=identity_index
            ):
                if identity_index is None:
                    message = f"A video-level annotation with tag '{tag}' already exists at frames {start}-{end}."
                else:
                    message = f"An annotation with tag '{tag}' for this identity already exists at frames {start}-{end}."
                QtWidgets.QMessageBox.warning(self, "Duplicate annotation", message)
                return

            # No duplicate found; create and insert the annotation
            annotation = TimelineAnnotations.Annotation(
                tag=tag,
                color=result["color"],
                description=result["description"],
                identity_index=identity_index,
                display_identity=display_identity,
                start=start,
                end=end,
            )
            self._labels.add_annotation(annotation)
            self._project.save_annotations(self._labels, self._pose_est)
            self._label_button_common()
            self._player_widget.update()

    def on_annotation_edited(self, key: dict, updated: dict) -> None:
        """Handle the event when an existing annotation is edited.

        Args:
            key (dict): Original annotation key with start, end, tag, identity (uniquely identifies annotation).
            updated (dict): Updated annotation details from the dialog.

        This is an external API called by the AnnotationEditDialog when an annotation is edited and not a signal
        handler only used internally in CentralWidget.

        Note: start, end, tag value, and identity uniquely identify an annotation. If the updated annotation
        would create a duplicate with the same values, do not apply the update and show a warning message instead.
        """
        start = key["start"]
        end = key["end"]
        tag = key["tag"]
        old_identity = key["identity"]

        # Determine the new identity based on scope toggle
        new_tag = updated["tag"]
        new_identity = old_identity if updated["identity_scoped"] else None
        display_identity = (
            self._pose_est.identity_index_to_display(new_identity)
            if new_identity is not None
            else None
        )
        new_data = TimelineAnnotations.Annotation(
            start=start,
            end=end,
            tag=new_tag,
            color=updated["color"],
            description=updated["description"],
            identity_index=new_identity,
            display_identity=display_identity,
        )

        # Only treat as duplicate if the new key (start,end,tag,identity) collides with a *different* annotation
        # i.e., if the tag/identity are unchanged, it's the same annotation and should be allowed.
        if not (
            new_tag == tag and new_identity == old_identity
        ) and self._labels.timeline_annotations.annotation_exists(
            start=start, end=end, tag=new_tag, identity_index=new_identity
        ):
            message = f"An annotation with tag '{new_tag}' for this identity already exists at frames {start}-{end}."
            QtWidgets.QMessageBox.warning(self, "Duplicate annotation", message)
            return

        # modification is handled by removing the old annotation and inserting a new one with the updated properties
        self._labels.timeline_annotations.remove_annotation_by_key(
            start=start, end=end, tag=tag, identity_index=old_identity
        )
        self._labels.timeline_annotations.add_annotation(new_data)

        # make sure the changes are saved
        self._project.save_annotations(self._labels, self._pose_est)

        # refresh video player to show updated annotation
        self._player_widget.update()

    def on_annotation_deleted(self, payload: dict) -> None:
        """Handle the event when an existing annotation is deleted.

        Args:
            payload (dict): Payload from the deleted annotation.

        This is an external API called by the AnnotationEditDialog when an annotation is deleted and not a signal
        handler only used internally in CentralWidget.
        """
        start = payload.get("start")
        end = payload.get("end")
        tag = payload.get("tag")
        identity = payload.get("identity_index")

        if None in (start, end, tag):
            raise RuntimeWarning("Invalid annotation delete payload")

        self._labels.timeline_annotations.remove_annotation_by_key(
            start=start, end=end, tag=tag, identity_index=identity
        )

        # make sure the changes are saved
        self._project.save_annotations(self._labels, self._pose_est)

        # refresh video player to show updated annotation
        self._player_widget.update()

    def _save_and_refresh_annotations(self) -> None:
        """Persist annotation changes and update UI widgets."""
        self._project.save_annotations(self._labels, self._pose_est)
        self._player_widget.update()
