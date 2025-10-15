import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction

from jabs.constants import ORG_NAME, RECENT_PROJECTS_MAX
from jabs.feature_extraction.landmark_features import LandmarkFeatureGroup
from jabs.project import export_training_data
from jabs.ui.behavior_search_dialog import BehaviorSearchDialog
from jabs.utils import FINAL_TRAIN_SEED, get_bool_env_var, hide_stderr
from jabs.version import version_str

from .about_dialog import AboutDialog
from .archive_behavior_dialog import ArchiveBehaviorDialog
from .central_widget import CentralWidget
from .license_dialog import LicenseAgreementDialog
from .player_widget import PlayerWidget
from .progress_dialog import create_progress_dialog
from .project_loader_thread import ProjectLoaderThread
from .project_pruning_dialog import ProjectPruningDialog
from .stacked_timeline_widget import StackedTimelineWidget
from .user_guide_dialog import UserGuideDialog
from .util import send_file_to_recycle_bin
from .video_list_widget import VideoListDockWidget

USE_NATIVE_FILE_DIALOG = get_bool_env_var("JABS_NATIVE_FILE_DIALOG", True)

RECENT_PROJECTS_KEY = "recent_projects"
LICENSE_ACCEPTED_KEY = "license_accepted"
LICENSE_VERSION_KEY = "license_version"
WINDOW_SIZE_KEY = "main_window_size"
SESSION_TRACKING_ENABLED_KEY = "session_tracking_enabled"


class MainWindow(QtWidgets.QMainWindow):
    """Main application window for the JABS UI.

    Handles the setup and management of the main user interface, including menus, status bar,
    central widget, and dock widgets. Manages project loading, user actions, and feature toggles.

    Args:
        app_name (str): Short application name.
        app_name_long (str): Full application name.
        *args: Additional positional arguments for QMainWindow.
        **kwargs: Additional keyword arguments for QMainWindow.
    """

    def __init__(self, app_name: str, app_name_long: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.setWindowTitle(f"{app_name_long} {version_str()}")
        self._central_widget = CentralWidget(self)
        self._central_widget.status_message.connect(self.display_status_message)
        self._central_widget.search_hit_loaded.connect(self._search_hit_loaded)
        self.setCentralWidget(self._central_widget)
        self.setStatusBar(QtWidgets.QStatusBar(self))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setUnifiedTitleAndToolBarOnMac(True)

        self._app_name = app_name
        self._app_name_long = app_name_long
        self._project = None
        self._project_loader_thread = None
        self._progress_dialog = None
        self._user_guide_window = None
        self._settings = QtCore.QSettings(ORG_NAME, app_name)
        self._previous_identity_overlay_mode = PlayerWidget.IdentityOverlayMode.FLOATING

        size = self._settings.value("main_window_size", None, type=QtCore.QSize)
        if size and isinstance(size, QtCore.QSize):
            self.resize(size)
        else:
            self.resize(1280, 720)

        # setup menu bar
        menu = self.menuBar()

        app_menu = menu.addMenu(self._app_name)
        file_menu = menu.addMenu("File")
        view_menu = menu.addMenu("View")
        feature_menu = menu.addMenu("Features")

        # Setup App Menu
        # about app
        about_action = QtGui.QAction(f" &About {self._app_name}", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self._show_about_dialog)
        app_menu.addAction(about_action)

        # user guide
        user_guide_action = QtGui.QAction(" &User Guide", self)
        user_guide_action.setStatusTip("Open User Guide")
        user_guide_action.setShortcut(QtGui.QKeySequence("Ctrl+U"))
        user_guide_action.triggered.connect(self._open_user_guide)
        app_menu.addAction(user_guide_action)

        # license action
        license_action = QtGui.QAction("View License Agreement", self)
        license_action.setStatusTip("View License Agreement")
        license_action.triggered.connect(self._view_license)
        app_menu.addAction(license_action)

        # enable/disable session tracking
        session_tracking_action = QtGui.QAction("Enable Session Tracking", self)
        session_tracking_action.setStatusTip("Enable or disable session tracking")
        session_tracking_action.setCheckable(True)
        session_tracking_action.triggered.connect(self._on_session_tracking_triggered)
        session_tracking_action.setChecked(
            self._settings.value(SESSION_TRACKING_ENABLED_KEY, False, type=bool)
        )
        app_menu.addAction(session_tracking_action)

        # clear cache action
        self._clear_cache = QtGui.QAction("Clear Project Cache", self)
        self._clear_cache.setStatusTip("Clear Project Cache")
        self._clear_cache.setEnabled(False)
        self._clear_cache.triggered.connect(self._clear_cache_action)
        app_menu.addAction(self._clear_cache)

        # exit action
        exit_action = QtGui.QAction(f" &Quit {self._app_name}", self)
        exit_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(QtCore.QCoreApplication.quit)
        app_menu.addAction(exit_action)

        # Setup File Menu
        # open action
        open_action = QtGui.QAction("&Open Project", self)
        open_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        open_action.setStatusTip("Open Project")
        open_action.triggered.connect(self._show_project_open_dialog)
        file_menu.addAction(open_action)

        # open recent
        self._open_recent_menu = QtWidgets.QMenu("Open Recent", self)
        file_menu.addMenu(self._open_recent_menu)
        self._update_recent_projects()

        # export training data action
        self._export_training = QtGui.QAction("Export Training Data", self)
        self._export_training.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        self._export_training.setStatusTip("Export training data for this classifier")
        self._export_training.setEnabled(False)
        self._export_training.triggered.connect(self._export_training_data)
        file_menu.addAction(self._export_training)

        # archive behavior action
        self._archive_behavior = QtGui.QAction("Archive Behavior", self)
        self._archive_behavior.setStatusTip("Open Archive Behavior Dialog")
        self._archive_behavior.setEnabled(False)
        self._archive_behavior.triggered.connect(self._open_archive_behavior_dialog)
        file_menu.addAction(self._archive_behavior)

        # prune project action
        self._prune_action = QtGui.QAction("Prune Project", self)
        self._prune_action.setStatusTip("Remove videos with no labels")
        self._prune_action.setEnabled(False)
        self._prune_action.triggered.connect(self._show_project_pruning_dialog)
        file_menu.addAction(self._prune_action)

        # Setup View Menu

        # video playlist menu item
        self.view_playlist = QtGui.QAction("View Playlist", self)
        self.view_playlist.setCheckable(True)
        self.view_playlist.triggered.connect(self._set_video_list_visibility)
        view_menu.addAction(self.view_playlist)

        # Timeline submenu
        timeline_menu = QtWidgets.QMenu("Timeline", self)
        view_menu.addMenu(timeline_menu)

        # First mutually exclusive group: Labels & Predictions, Labels, Predictions
        timeline_group = QtGui.QActionGroup(self)
        timeline_group.setExclusive(True)

        self._timeline_labels_preds = QtGui.QAction("Labels && Predictions", self, checkable=True)
        self._timeline_labels = QtGui.QAction("Labels", self, checkable=True)
        self._timeline_preds = QtGui.QAction("Predictions", self, checkable=True)

        timeline_group.addAction(self._timeline_labels_preds)
        timeline_group.addAction(self._timeline_labels)
        timeline_group.addAction(self._timeline_preds)

        timeline_menu.addAction(self._timeline_labels_preds)
        timeline_menu.addAction(self._timeline_labels)
        timeline_menu.addAction(self._timeline_preds)

        self._timeline_labels_preds.triggered.connect(self._on_timeline_view_mode_changed)
        self._timeline_labels.triggered.connect(self._on_timeline_view_mode_changed)
        self._timeline_preds.triggered.connect(self._on_timeline_view_mode_changed)

        # Separator
        timeline_menu.addSeparator()

        # Second mutually exclusive group: All Animals, Selected Animals
        animal_group = QtGui.QActionGroup(self)
        animal_group.setExclusive(True)

        self._timeline_all_animals = QtGui.QAction("All Animals", self, checkable=True)
        self._timeline_selected_animal = QtGui.QAction("Selected Animal", self, checkable=True)

        animal_group.addAction(self._timeline_all_animals)
        animal_group.addAction(self._timeline_selected_animal)

        timeline_menu.addAction(self._timeline_all_animals)
        timeline_menu.addAction(self._timeline_selected_animal)

        self._timeline_all_animals.triggered.connect(self._on_timeline_identity_mode_changed)
        self._timeline_selected_animal.triggered.connect(self._on_timeline_identity_mode_changed)

        # Set default checked actions
        self._timeline_labels_preds.setChecked(True)
        self._timeline_selected_animal.setChecked(True)

        # label overlay menu
        label_overlay_menu = QtWidgets.QMenu("Label Overlay", self)
        view_menu.addMenu(label_overlay_menu)
        # mutually exclusive group: None, Labels, Predictions
        label_overlay_group = QtGui.QActionGroup(self)
        label_overlay_group.setExclusive(True)

        self._label_overlay_none = QtGui.QAction("No Overlay", self, checkable=True, checked=True)
        self._label_overlay_labels = QtGui.QAction("Labels", self, checkable=True)
        self._label_overlay_preds = QtGui.QAction("Predictions", self, checkable=True)
        label_overlay_group.addAction(self._label_overlay_none)
        label_overlay_group.addAction(self._label_overlay_labels)
        label_overlay_group.addAction(self._label_overlay_preds)
        label_overlay_menu.addAction(self._label_overlay_none)
        label_overlay_menu.addAction(self._label_overlay_labels)
        label_overlay_menu.addAction(self._label_overlay_preds)

        self._label_overlay_none.triggered.connect(self._on_label_overlay_mode_changed)
        self._label_overlay_labels.triggered.connect(self._on_label_overlay_mode_changed)
        self._label_overlay_preds.triggered.connect(self._on_label_overlay_mode_changed)

        # Identity Overlay submenu
        identity_overlay_menu = QtWidgets.QMenu("Identity Overlay", self)
        view_menu.addMenu(identity_overlay_menu)

        identity_overlay_group = QtGui.QActionGroup(self)
        identity_overlay_group.setExclusive(True)

        self._identity_overlay_centroid = QtGui.QAction("Centroid", self, checkable=True)
        self._identity_overlay_floating = QtGui.QAction("Floating", self, checkable=True)
        self._identity_overlay_minimal = QtGui.QAction("Minimalist", self, checkable=True)
        self._identity_overlay_bbox = QtGui.QAction("Bounding Box", self, checkable=True)

        identity_overlay_group.addAction(self._identity_overlay_centroid)
        identity_overlay_group.addAction(self._identity_overlay_floating)
        identity_overlay_group.addAction(self._identity_overlay_minimal)
        identity_overlay_group.addAction(self._identity_overlay_bbox)

        identity_overlay_menu.addAction(self._identity_overlay_centroid)
        identity_overlay_menu.addAction(self._identity_overlay_floating)
        identity_overlay_menu.addAction(self._identity_overlay_minimal)
        identity_overlay_menu.addAction(self._identity_overlay_bbox)

        # set the checked state based on the current identity overlay mode
        match self._central_widget.id_overlay_mode:
            case PlayerWidget.IdentityOverlayMode.CENTROID:
                self._identity_overlay_centroid.setChecked(True)
            case PlayerWidget.IdentityOverlayMode.FLOATING:
                self._identity_overlay_floating.setChecked(True)
            case PlayerWidget.IdentityOverlayMode.BBOX:
                self._identity_overlay_bbox.setChecked(True)
            case _:
                self._identity_overlay_minimal.setChecked(True)

        self._identity_overlay_centroid.triggered.connect(
            lambda: setattr(
                self._central_widget, "id_overlay_mode", PlayerWidget.IdentityOverlayMode.CENTROID
            )
        )
        self._identity_overlay_floating.triggered.connect(
            lambda: setattr(
                self._central_widget, "id_overlay_mode", PlayerWidget.IdentityOverlayMode.FLOATING
            )
        )
        self._identity_overlay_minimal.triggered.connect(
            lambda: setattr(
                self._central_widget, "id_overlay_mode", PlayerWidget.IdentityOverlayMode.MINIMAL
            )
        )
        self._identity_overlay_bbox.triggered.connect(
            lambda: setattr(
                self._central_widget, "id_overlay_mode", PlayerWidget.IdentityOverlayMode.BBOX
            )
        )

        overlay_annotations = QtGui.QAction("Overlay Annotations", self)
        overlay_annotations.setCheckable(True)
        overlay_annotations.setChecked(self._central_widget.overlay_annotations_enabled)
        overlay_annotations.triggered.connect(
            lambda checked: setattr(self._central_widget, "overlay_annotations_enabled", checked)
        )
        view_menu.addAction(overlay_annotations)

        self.show_track = QtGui.QAction("Show Track", self)
        self.show_track.setCheckable(True)
        self.show_track.triggered.connect(self._set_animal_track_visibility)
        view_menu.addAction(self.show_track)

        self.overlay_pose = QtGui.QAction("Overlay Pose", self)
        self.overlay_pose.setCheckable(True)
        self.overlay_pose.triggered.connect(self._set_pose_overlay_visibility)
        view_menu.addAction(self.overlay_pose)

        self.overlay_landmark = QtGui.QAction("Overlay Landmarks", self)
        self.overlay_landmark.setCheckable(True)
        self.overlay_landmark.triggered.connect(self._set_landmark_overlay_visibility)
        view_menu.addAction(self.overlay_landmark)

        # Test add check mark for overlay segmentation
        self.overlay_segmentation = QtGui.QAction("Overlay Segmentation", self)
        self.overlay_segmentation.setCheckable(True)
        self.overlay_segmentation.triggered.connect(self._set_segmentation_overlay_visibility)
        view_menu.addAction(self.overlay_segmentation)

        # add behavior search
        self.behavior_search = QtGui.QAction("Search Behaviors", self)
        self.behavior_search.setShortcut(QtGui.QKeySequence.StandardKey.Find)
        self.behavior_search.setStatusTip("Search for behaviors")
        self.behavior_search.setEnabled(False)
        self.behavior_search.triggered.connect(self._search_behaviors)
        view_menu.addAction(self.behavior_search)

        # Feature subset actions
        # All these settings should be updated whenever the behavior_changed event occurs
        self._central_widget.controls.behavior_changed.connect(self.behavior_changed_event)
        self._central_widget.controls.new_behavior_label.connect(self.behavior_label_add_event)

        self.enable_cm_units = QtGui.QAction("CM Units", self)
        self.enable_cm_units.setCheckable(True)
        self.enable_cm_units.triggered.connect(self._set_use_cm_units)
        feature_menu.addAction(self.enable_cm_units)

        self.enable_window_features = QtGui.QAction("Enable Window Features", self)
        self.enable_window_features.setCheckable(True)
        self.enable_window_features.triggered.connect(self._set_window_features_enabled)
        feature_menu.addAction(self.enable_window_features)

        self.enable_fft_features = QtGui.QAction("Enable Signal Features", self)
        self.enable_fft_features.setCheckable(True)
        self.enable_fft_features.triggered.connect(self._set_fft_features_enabled)
        feature_menu.addAction(self.enable_fft_features)

        self.enable_social_features = QtGui.QAction("Enable Social Features", self)
        self.enable_social_features.setCheckable(True)
        self.enable_social_features.triggered.connect(self._set_social_features_enabled)
        feature_menu.addAction(self.enable_social_features)

        # Static objects
        enable_landmark_features = {}
        for landmark_name in LandmarkFeatureGroup.feature_map:
            landmark_action = QtGui.QAction(f"Enable {landmark_name.capitalize()} Features", self)
            landmark_action.setCheckable(True)
            landmark_action.triggered.connect(self._set_static_object_features_enabled)
            feature_menu.addAction(landmark_action)
            enable_landmark_features[landmark_name] = landmark_action
        self.enable_landmark_features = enable_landmark_features

        self.enable_segmentation_features = QtGui.QAction("Enable Segmentation Features", self)
        self.enable_segmentation_features.setCheckable(True)
        self.enable_segmentation_features.triggered.connect(
            self._set_segmentation_features_enabled
        )
        feature_menu.addAction(self.enable_segmentation_features)

        # select all action
        select_all_action = QtGui.QAction(self)
        select_all_action.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self._handle_select_all)
        self.addAction(select_all_action)

        # playlist widget added to dock on left side of main window
        self.video_list = VideoListDockWidget(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.video_list)
        self.video_list.setFloating(False)
        self.video_list.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # if the playlist visibility changes, make sure the view_playlists
        # check mark is set correctly
        self.video_list.visibilityChanged.connect(self.view_playlist.setChecked)

        # handle event where user selects a different video in the playlist
        self.video_list.selectionChanged.connect(self._video_list_selection)

        # handle event to set status of File-Export Training Data action
        self._central_widget.export_training_status_change.connect(
            self._export_training.setEnabled
        )

        # the video list needs to show search hit counts
        self._central_widget.search_results_changed.connect(self.video_list.show_search_results)

        # enable/disable the bounding box overlay menu item based
        self._central_widget.bbox_overlay_supported.connect(self._on_bbox_overlay_support_changed)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """override keyPressEvent so we can pass some key press events on to the centralWidget"""
        key = event.key()

        # pass along some of the key press events to the central widget
        if key in [
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Down,
            Qt.Key.Key_Up,
            Qt.Key.Key_Space,
            Qt.Key.Key_Z,
            Qt.Key.Key_X,
            Qt.Key.Key_C,
            Qt.Key.Key_Escape,
            Qt.Key.Key_Question,
            Qt.Key.Key_Shift,
        ]:
            self.centralWidget().keyPressEvent(event)
            return

        match key:
            case Qt.Key.Key_T:
                self.show_track.trigger()
            case Qt.Key.Key_P:
                self.overlay_pose.trigger()
            case Qt.Key.Key_L:
                self.overlay_landmark.trigger()
            case Qt.Key.Key_Comma:
                self.video_list.select_previous_video()
            case Qt.Key.Key_Period:
                self.video_list.select_next_video()
            case Qt.Key.Key_I if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self._toggle_identity_overlay_minimalist()
            case _:
                # anything else pass on to the super class keyPressEvent
                super().keyPressEvent(event)

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:
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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handle the resize event of the main window.

        This method saves the current size of the main window to the settings so the size can be restored next time
        the application is run.
        """
        super().resizeEvent(event)
        self._settings.setValue("main_window_size", self.size())

    def open_project(self, project_path: str) -> None:
        """open a new project directory"""
        self._progress_dialog = create_progress_dialog(self, "Loading Project...", 0)
        self._progress_dialog.show()

        session_tracking_enabled = bool(
            self._settings.value(SESSION_TRACKING_ENABLED_KEY, False, type=bool)
        )
        self._project_loader_thread = ProjectLoaderThread(
            project_path, parent=self, session_tracking_enabled=session_tracking_enabled
        )
        self._project_loader_thread.project_loaded.connect(self._project_loaded_callback)
        self._project_loader_thread.load_error.connect(self._project_load_error_callback)
        self._project_loader_thread.start()

    def changeEvent(self, event: QEvent) -> None:
        """Handle change events for the main window.

        If a project is open, minimizing the window will pause the session tracker
        """
        if event.type() == QEvent.Type.WindowStateChange and self._project:
            old_state = event.oldState()
            new_state = self.windowState()

            was_minimized = bool(old_state & Qt.WindowState.WindowMinimized)
            is_minimized = bool(new_state & Qt.WindowState.WindowMinimized)

            if not was_minimized and is_minimized:
                # just entered minimized state
                self._project.session_tracker.pause_session()
            elif was_minimized and not is_minimized:
                self._project.session_tracker.resume_session()

        super().changeEvent(event)

    def behavior_changed_event(self, new_behavior: str) -> None:
        """menu items to change when a new behavior is selected."""
        # skip if no behavior assigned (only should occur during new project)
        if new_behavior is None or new_behavior == "":
            return

        # Populate settings based project data
        behavior_metadata = self._project.settings_manager.get_behavior(new_behavior)
        self.enable_cm_units.setChecked(behavior_metadata.get("cm_units", False))
        self.enable_window_features.setChecked(behavior_metadata.get("window", False))
        self.enable_fft_features.setChecked(behavior_metadata.get("fft", False))
        self.enable_social_features.setChecked(behavior_metadata.get("social", False))
        self.enable_segmentation_features.setChecked(behavior_metadata.get("segmentation", False))
        static_settings = behavior_metadata.get("static_objects", {})
        for static_object, menu_item in self.enable_landmark_features.items():
            menu_item.setChecked(static_settings.get(static_object, False))

    def behavior_label_add_event(self, behaviors: list[str]) -> None:
        """handle project updates required when user adds new behavior labels"""
        # check for new behaviors
        for behavior in behaviors:
            if behavior not in self._project.settings_manager.project_settings["behavior"]:
                # save new behavior with default settings
                self._project.settings_manager.save_behavior(behavior, {})

    def display_status_message(self, message: str, duration: int = 3000) -> None:
        """display a message in the main window status bar

        Args:
            message: message to display
            duration: duration of the message in milliseconds. Use 0 to
                display the message until clear_status_bar() is called

        Returns:
            None
        """
        if duration < 0:
            raise ValueError("duration must be >= 0")
        self.statusBar().showMessage(message, duration)

    def clear_status_bar(self) -> None:
        """clear the status bar message

        Returns:
            None
        """
        self.statusBar().clearMessage()

    def _show_project_open_dialog(self) -> None:
        """prompt the user to select a project directory and open it"""
        options = QtWidgets.QFileDialog.Option(0)
        if not USE_NATIVE_FILE_DIALOG:
            options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog

        # on macOS QFileDialog can cause some error messages to be written to stderr but the dialog still works
        # so hide anything written to stderr while we're showing the dialog. we can use a Qt based file dialog instead
        # of the native one by setting the env variable USE_NATIVE_FILE_DIALOG to a non-true value (default is True)
        with hide_stderr():
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Project Directory", options=options
            )

        if directory:
            self.open_project(directory)

    def _show_about_dialog(self) -> None:
        dialog = AboutDialog(f"{self._app_name_long} ({self._app_name})", self)
        dialog.exec_()

    def _open_user_guide(self) -> None:
        """show the user guide document in a separate window"""
        if self._user_guide_window is None:
            self._user_guide_window = UserGuideDialog(
                f"{self._app_name_long} ({self._app_name})", parent=None
            )
        self._user_guide_window.show()

    def _export_training_data(self) -> None:
        if not self._central_widget.classify_button_enabled:
            # classify button disabled, don't allow exporting training data
            QtWidgets.QMessageBox.warning(
                self,
                "Unable to export training data",
                "Classifier has not been trained, or classifier parameters "
                "have changed.\n\n"
                "You must train the classifier before export.",
            )
            return

        try:
            out_path = export_training_data(
                self._project,
                self._central_widget.behavior,
                self._project.feature_manager.min_pose_version,
                self._central_widget.classifier_type,
                FINAL_TRAIN_SEED,
            )
            self.display_status_message(f"Training data exported: {out_path}", 5000)
        except OSError as e:
            print(f"Unable to export training data: {e}", file=sys.stderr)

    def _set_video_list_visibility(self, checked: bool) -> None:
        """show/hide video list"""
        if not checked:
            # user unchecked
            self.video_list.hide()
        else:
            # user checked
            self.video_list.show()

    def _set_animal_track_visibility(self, checked: bool) -> None:
        """show/hide track overlay for subject."""
        self._central_widget.show_track(checked)

    def _set_pose_overlay_visibility(self, checked: bool) -> None:
        """show/hide pose overlay for subject."""
        self._central_widget.overlay_pose(checked)

    def _set_landmark_overlay_visibility(self, checked: bool) -> None:
        """show/hide landmark features."""
        self._central_widget.overlay_landmarks(checked)

    def _set_segmentation_overlay_visibility(self, checked: bool) -> None:
        """show/hide segmentation overlay for subject."""
        self._central_widget.overlay_segmentation(checked)

    def _search_behaviors(self) -> None:
        """open a dialog to search for behaviors if a project is loaded."""
        if self._project is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Project Loaded",
                "Please load a project before searching for behaviors.",
            )
            return

        # open the behavior search dialog
        dialog = BehaviorSearchDialog(self._project, self)
        if dialog.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            search_query = dialog.behavior_search_query
            self._central_widget.update_behavior_search_query(search_query)

    def _set_use_cm_units(self, checked: bool) -> None:
        """toggle project to use pixel units."""
        # TODO: Warn the user that features may need to be re-calculated
        self._project.settings_manager.save_behavior(
            self._central_widget.behavior, {"cm_units": checked}
        )

    def _set_social_features_enabled(self, checked: bool) -> None:
        """toggle project to use social features."""
        self._project.settings_manager.save_behavior(
            self._central_widget.behavior, {"social": checked}
        )

    def _set_window_features_enabled(self, checked: bool) -> None:
        """toggle project to use window features."""
        self._project.settings_manager.save_behavior(
            self._central_widget.behavior, {"window": checked}
        )

    def _set_fft_features_enabled(self, checked: bool) -> None:
        """toggle project to use fft features."""
        self._project.settings_manager.save_behavior(
            self._central_widget.behavior, {"fft": checked}
        )

    def _set_segmentation_features_enabled(self, checked: bool) -> None:
        """toggle project to use segmentation features."""
        self._project.settings_manager.save_behavior(
            self._central_widget.behavior, {"segmentation": checked}
        )

    def _set_static_object_features_enabled(self, checked: bool) -> None:
        """toggle project to use a specific static object feature set."""
        # get the key from the caller
        key = self.sender().text().split(" ")[1].lower()
        all_object_settings = self._project.settings_manager.get_behavior(
            self._central_widget.behavior
        ).get("static_objects", {})
        all_object_settings[key] = checked
        self._project.settings_manager.save_behavior(
            self._central_widget.behavior, {"static_objects": all_object_settings}
        )

    def _video_list_selection(self, filename: str) -> None:
        """handle a click on a new video in the list loaded into the main window dock"""
        try:
            self._central_widget.load_video(self._project.video_manager.video_path(filename))
        except OSError as e:
            self.display_status_message(f"Unable to load video: {e}")
            self._project_load_error_callback(e)

    def _open_archive_behavior_dialog(self) -> None:
        dialog = ArchiveBehaviorDialog(self._central_widget.behaviors, self)
        dialog.behavior_archived.connect(self._archive_behavior_callback)
        dialog.exec_()

    def _archive_behavior_callback(self, behavior: str) -> None:
        self._project.archive_behavior(behavior)
        self._central_widget.remove_behavior(behavior)

    def _project_loaded_callback(self) -> None:
        """Callback function to be called when the project is loaded."""
        self._project = self._project_loader_thread.project
        self._project_loader_thread = None

        # The central_widget updates main_control_widget
        self._central_widget.set_project(self._project)
        self.video_list.set_project(self._project)

        # Update which controls should be available
        self._archive_behavior.setEnabled(True)
        self._prune_action.setEnabled(True)
        self.enable_cm_units.setEnabled(self._project.feature_manager.is_cm_unit)
        self.enable_social_features.setEnabled(
            self._project.feature_manager.can_use_social_features
        )
        self.enable_segmentation_features.setEnabled(
            self._project.feature_manager.can_use_segmentation_features
        )
        self._clear_cache.setEnabled(True)
        available_objects = self._project.feature_manager.static_objects
        for static_object, menu_item in self.enable_landmark_features.items():
            if static_object in available_objects:
                menu_item.setEnabled(True)
            else:
                menu_item.setEnabled(False)
        self.behavior_search.setEnabled(True)

        # update the recent project menu
        self._add_recent_project(self._project.project_paths.project_dir)
        self._progress_dialog.close()
        self._progress_dialog.deleteLater()
        self._progress_dialog = None

    def _project_load_error_callback(self, error: Exception) -> None:
        """Callback function to be called when the project fails to load."""
        self._project_loader_thread.deleteLater()
        self._project_loader_thread = None
        self._progress_dialog.close()
        self._progress_dialog.deleteLater()
        self._progress_dialog = None
        QtWidgets.QMessageBox.critical(self, "Error loading project", str(error))

    def show_license_dialog(self) -> QtWidgets.QDialog.DialogCode:
        """prompt the user to accept the license agreement if they haven't already"""
        # check to see if user already accepted the license
        if self._settings.value(LICENSE_ACCEPTED_KEY, False, type=bool):
            return QtWidgets.QDialog.DialogCode.Accepted

        # show dialog
        dialog = LicenseAgreementDialog(self)
        result = dialog.exec_()

        # persist the license acceptance
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self._settings.setValue(LICENSE_ACCEPTED_KEY, True)
            self._settings.setValue(LICENSE_VERSION_KEY, version_str())
            self._settings.sync()

        return QtWidgets.QDialog.DialogCode(result)

        # show dialog

    def _clear_cache_action(self):
        """Clear the cache for the current project. Opens a dialog to get user confirmation first."""
        app = QtWidgets.QApplication.instance()
        dont_use_native_dialogs = QtWidgets.QApplication.instance().testAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs
        )

        # if app is currently set to use native dialogs, we will temporarily set it to use Qt dialogs
        # the native style, at least on macOS, is not ideal so we'll force the Qt dialog instead
        if not dont_use_native_dialogs:
            app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)

        result = QtWidgets.QMessageBox.warning(
            self,
            "Clear Cache",
            "Are you sure you want to clear the project cache?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        # restore the original setting
        app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, dont_use_native_dialogs)

        if result == QtWidgets.QMessageBox.StandardButton.Yes:
            self._project.clear_cache()
            # need to reload the current video to force the pose file to reload
            if self._central_widget.loaded_video:
                self._central_widget.load_video(self._central_widget.loaded_video)
            self.display_status_message("Cache cleared", 3000)

    def _update_recent_projects(self) -> None:
        """update the contents of the Recent Projects menu"""
        self._open_recent_menu.clear()
        recent_projects = self._settings.value(RECENT_PROJECTS_KEY, [], type=list)

        # add menu action for each of the recent projects
        for project_path in recent_projects:
            action = self._open_recent_menu.addAction(project_path)
            action.setData(project_path)
            action.triggered.connect(self._open_recent_project)

    def _add_recent_project(self, project_path: Path) -> None:
        """add a project to the recent projects list"""
        # project path in the _project_loaded_callback is a Path object, Qt needs a string to add to the menu
        path_str = str(project_path.absolute())

        recent_projects = self._settings.value(RECENT_PROJECTS_KEY, [], type=list)

        # remove the project if it already exists in the list since we're going to add it to the front of the list
        # this keeps the list sorted with the most recent project at the top
        if path_str in recent_projects:
            recent_projects.remove(path_str)

        # add the project to the front of the list and truncate the list to the max size
        recent_projects.insert(0, path_str)
        recent_projects = recent_projects[:RECENT_PROJECTS_MAX]

        # persist updated recent projects list
        self._settings.setValue(RECENT_PROJECTS_KEY, recent_projects)
        self._settings.sync()

        # update the menu
        self._update_recent_projects()

    def _open_recent_project(self) -> None:
        """open a recent project"""
        action = self.sender()
        if isinstance(action, QAction):
            project_path = action.data()
            if project_path:
                self.open_project(project_path)

    def _on_timeline_view_mode_changed(self) -> None:
        if self._timeline_labels_preds.isChecked():
            self._central_widget.timeline_view_mode = (
                StackedTimelineWidget.ViewMode.LABELS_AND_PREDICTIONS
            )
        elif self._timeline_labels.isChecked():
            self._central_widget.timeline_view_mode = StackedTimelineWidget.ViewMode.LABELS
        elif self._timeline_preds.isChecked():
            self._central_widget.timeline_view_mode = StackedTimelineWidget.ViewMode.PREDICTIONS

    def _on_timeline_identity_mode_changed(self) -> None:
        if self._timeline_all_animals.isChecked():
            self._central_widget.timeline_identity_mode = StackedTimelineWidget.IdentityMode.ALL
        elif self._timeline_selected_animal.isChecked():
            self._central_widget.timeline_identity_mode = StackedTimelineWidget.IdentityMode.ACTIVE

    def _search_hit_loaded(self, search_hit) -> None:
        """Update the selected video in the video list when a search hit is loaded."""
        if search_hit is not None:
            self.video_list.select_video(search_hit.file, suppress_event=True)

    def _on_label_overlay_mode_changed(self) -> None:
        if self._label_overlay_none.isChecked():
            self._central_widget.label_overlay_mode = PlayerWidget.LabelOverlayMode.NONE
        elif self._label_overlay_labels.isChecked():
            self._central_widget.label_overlay_mode = PlayerWidget.LabelOverlayMode.LABEL
        elif self._label_overlay_preds.isChecked():
            self._central_widget.label_overlay_mode = PlayerWidget.LabelOverlayMode.PREDICTION

    def _on_session_tracking_triggered(self, checked: bool) -> None:
        """Handle the session tracking toggle action."""
        self._settings.setValue(SESSION_TRACKING_ENABLED_KEY, checked)
        if self._project:
            if checked:
                # if a project is already loaded and user is enabling session tracking,
                # they need to open the project again to enable session tracking.
                QtWidgets.QMessageBox.warning(
                    self,
                    "Session Tracking Enabled",
                    "Session Tracking Enabled: Please reopen the project to start tracking.",
                )
            else:
                # if session tracking was just disabled, we stop logging new events
                self._project.session_tracker.enabled = False

    def _handle_select_all(self) -> None:
        """Handle the Select All event"""
        self._central_widget.select_all()

    def _show_project_pruning_dialog(self) -> None:
        """Handle the Prune Project menu action."""
        dialog = ProjectPruningDialog(self._project, parent=self)
        if dialog.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            videos_to_prune = dialog.videos_to_prune

            # there were no videos selected for pruning, nothing to do
            if not videos_to_prune:
                return

            # don't let the user remove all videos from the project
            if len(videos_to_prune) == len(self._project.video_manager.videos):
                QtWidgets.QMessageBox.critical(
                    self,
                    "All Videos Selected",
                    "ERROR: This action would remove all videos from the project.",
                )
                return

            # User confirmed to prune videos. Create a Set of all files to delete.
            files_to_delete = {video.video_path for video in videos_to_prune}
            files_to_delete.update(video.pose_path for video in videos_to_prune)
            files_to_delete.update(video.annotation_path for video in videos_to_prune)
            self._move_files_to_recycle_bin_with_delete_fallback(files_to_delete)

            # Remove videos from the project video manager, which keeps track of the project's videos.
            for video in videos_to_prune:
                self._project.video_manager.remove_video(video.video_path.name)

            # Force the video list to update its contents
            self.video_list.set_project(self._project)

    def _move_files_to_recycle_bin_with_delete_fallback(self, files: set[Path]) -> None:
        """Attempt to move files to recycle bin, fallback to permanently delete if user agrees."""
        delete_on_failure: bool | None = None
        for file in files:
            removed = send_file_to_recycle_bin(file)
            if not removed:
                if delete_on_failure is None:
                    delete_on_failure = (
                        QtWidgets.QMessageBox.question(
                            self,
                            "Delete Failed",
                            f"Unable to move file to the recycle bin. Delete permanently?\n{file}",
                            QtWidgets.QMessageBox.StandardButton.Yes
                            | QtWidgets.QMessageBox.StandardButton.No,
                        )
                        == QtWidgets.QMessageBox.StandardButton.Yes
                    )
                if delete_on_failure:
                    try:
                        file.unlink(missing_ok=True)
                        self.statusBar().showMessage(f"{file} permanently deleted", 3000)
                    except OSError as e:
                        self.statusBar().showMessage(f"Unable to delete {file}", 3000)
                        print(f"Error deleting file {file}: {e}", file=sys.stderr)
            else:
                self.statusBar().showMessage(f"Moved {file} to recycle bin", 3000)

    def _toggle_identity_overlay_minimalist(self) -> None:
        checked = self._identity_overlay_minimal.isChecked()

        if checked:
            self._central_widget.id_overlay_mode = self._previous_identity_overlay_mode
            match self._previous_identity_overlay_mode:
                case PlayerWidget.IdentityOverlayMode.CENTROID:
                    self._identity_overlay_centroid.setChecked(True)
                case PlayerWidget.IdentityOverlayMode.FLOATING:
                    self._identity_overlay_floating.setChecked(True)
                case PlayerWidget.IdentityOverlayMode.MINIMAL:
                    self._identity_overlay_minimal.setChecked(True)
                case PlayerWidget.IdentityOverlayMode.BBOX:
                    self._identity_overlay_bbox.setChecked(True)
                case _:
                    # default to floating if previous_mode is not recognized
                    self._central_widget.id_overlay_mode = (
                        PlayerWidget.IdentityOverlayMode.FLOATING
                    )
                    self._identity_overlay_floating.setChecked(True)
        else:
            self._previous_identity_overlay_mode = self._central_widget.id_overlay_mode
            self._central_widget.id_overlay_mode = PlayerWidget.IdentityOverlayMode.MINIMAL
            self._identity_overlay_minimal.setChecked(True)

    def _on_bbox_overlay_support_changed(self, supported: bool) -> None:
        """Enable/disable the bounding box overlay menu item based on whether the current pose supports it."""
        self._identity_overlay_bbox.setEnabled(supported)
        if (
            not supported
            and self._central_widget.id_overlay_mode == PlayerWidget.IdentityOverlayMode.BBOX
        ):
            # if the user had bbox overlay selected but the new video doesn't support it, switch to floating
            self._central_widget.id_overlay_mode = PlayerWidget.IdentityOverlayMode.FLOATING
            self._identity_overlay_floating.setChecked(True)

        self._identity_overlay_bbox.setEnabled(supported)

    def _view_license(self) -> None:
        """View the license agreement (JABS->View License Agreement menu action)"""
        dialog = LicenseAgreementDialog(self, view_only=True)
        dialog.exec_()
