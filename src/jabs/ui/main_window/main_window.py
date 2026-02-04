import logging
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QEvent, Qt

from jabs.core.constants import ORG_NAME
from jabs.core.utils.process_pool_manager import ProcessPoolManager
from jabs.version import version_str

from ..constants import (
    LICENSE_ACCEPTED_KEY,
    LICENSE_VERSION_KEY,
    RECENT_PROJECTS_KEY,
    SESSION_TRACKING_ENABLED_KEY,
    WINDOW_SIZE_KEY,
)
from ..dialogs import LicenseAgreementDialog, MessageDialog
from ..dialogs.progress_dialog import create_progress_dialog
from ..player_widget import PlayerWidget
from ..project_loader_thread import ProjectLoaderThread
from .central_widget import CentralWidget
from .constants import DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH, RECENT_PROJECTS_MAX
from .menu_builder import MenuBuilder
from .menu_handlers import MenuHandlers
from .video_list_widget import VideoListDockWidget

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        app_name: str,
        app_name_long: str,
        process_pool: ProcessPoolManager = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._settings = QtCore.QSettings(ORG_NAME, app_name)

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
        self._pool_warm_thread = None
        self._user_guide_window = None
        self._previous_identity_overlay_mode = PlayerWidget.IdentityOverlayMode.FLOATING

        # Use the pre-created process pool (created before Qt to avoid fork+threads issues)
        # If no pool provided (e.g., in tests), create one lazily
        if process_pool is not None:
            self._process_pool = process_pool
            logger.debug(
                f"[MainWindow] Using pre-created ProcessPoolManager id={id(self._process_pool)}"
            )
        else:
            # Fallback for tests or if called without pool
            logger.warning(
                "[MainWindow] No process pool provided, creating lazily (not recommended in production)"
            )
            self._process_pool = ProcessPoolManager(name="JABS-AppProcessPool")

        self._pool_warm_thread = None

        # Load window size from settings
        size = self._settings.value(WINDOW_SIZE_KEY, None, type=QtCore.QSize)
        if size and isinstance(size, QtCore.QSize):
            self.resize(size)
        else:
            self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        # Create menu handlers (must be before MenuBuilder)
        self.menu_handlers = MenuHandlers(self)

        # Build all menus using MenuBuilder
        self.menu_builder = MenuBuilder(self, app_name, app_name_long)
        menu_refs = self.menu_builder.build_menus()

        # Store references to menus and actions for later use
        self._window_menu = menu_refs.window_menu
        self._open_recent_menu = menu_refs.open_recent_menu
        self._export_training = menu_refs.export_training
        self._archive_behavior = menu_refs.archive_behavior
        self._prune_action = menu_refs.prune_action
        self._clear_cache = menu_refs.clear_cache
        self.view_playlist = menu_refs.view_playlist
        self.show_track = menu_refs.show_track
        self.overlay_pose = menu_refs.overlay_pose
        self.overlay_landmark = menu_refs.overlay_landmark
        self.overlay_segmentation = menu_refs.overlay_segmentation
        self.behavior_search = menu_refs.behavior_search
        self._timeline_labels_preds = menu_refs.timeline_labels_preds
        self._timeline_labels = menu_refs.timeline_labels
        self._timeline_preds = menu_refs.timeline_preds
        self._timeline_all_animals = menu_refs.timeline_all_animals
        self._timeline_selected_animal = menu_refs.timeline_selected_animal
        self._label_overlay_none = menu_refs.label_overlay_none
        self._label_overlay_labels = menu_refs.label_overlay_labels
        self._label_overlay_preds = menu_refs.label_overlay_preds
        self._identity_overlay_centroid = menu_refs.identity_overlay_centroid
        self._identity_overlay_floating = menu_refs.identity_overlay_floating
        self._identity_overlay_minimal = menu_refs.identity_overlay_minimal
        self._identity_overlay_bbox = menu_refs.identity_overlay_bbox
        self.enable_cm_units = menu_refs.enable_cm_units
        self.enable_window_features = menu_refs.enable_window_features
        self.enable_fft_features = menu_refs.enable_fft_features
        self.enable_social_features = menu_refs.enable_social_features
        self.enable_landmark_features = menu_refs.enable_landmark_features
        self.enable_segmentation_features = menu_refs.enable_segmentation_features
        self._settings_action = menu_refs.settings_action

        # Update recent projects menu
        self._update_recent_projects()

        # Connect central widget signals to behavior events
        self._central_widget.controls.behavior_changed.connect(self.behavior_changed_event)
        self._central_widget.controls.new_behavior_label.connect(self.behavior_label_add_event)

        # Setup dock widgets
        self._setup_dock_widgets()

        # Connect central widget signals for menu state updates
        self._central_widget.export_training_status_change.connect(
            self._export_training.setEnabled
        )
        self._central_widget.search_results_changed.connect(self.video_list.show_search_results)
        self._central_widget.bbox_overlay_supported.connect(
            self.menu_handlers.on_bbox_overlay_support_changed
        )

    def _setup_dock_widgets(self) -> None:
        """Setup playlist and other dock widgets."""
        # Playlist widget added to dock on left side of main window
        self.video_list = VideoListDockWidget(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.video_list)
        self.video_list.setFloating(False)
        self.video_list.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # If the playlist visibility changes, make sure the view_playlists check mark is set correctly
        self.video_list.visibilityChanged.connect(self.view_playlist.setChecked)

        # Handle event where user selects a different video in the playlist
        self.video_list.selectionChanged.connect(self._video_list_selection)

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
        self._settings.setValue(WINDOW_SIZE_KEY, self.size())

    def open_project(self, project_path: str) -> None:
        """open a new project directory"""
        # Note: Process pool warming happens on startup and is shared across all projects
        # No need to interrupt or restart warming when opening projects
        self._progress_dialog = create_progress_dialog(self, "Loading Project...", 0)
        self._progress_dialog.show()

        # Clear the current project reference (no need to shut down executor - it's shared)
        self._project = None

        session_tracking_enabled = bool(
            self._settings.value(SESSION_TRACKING_ENABLED_KEY, False, type=bool)
        )
        self._project_loader_thread = ProjectLoaderThread(
            project_path,
            process_pool=self._process_pool,
            parent=self,
            session_tracking_enabled=session_tracking_enabled,
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

    def _video_list_selection(self, filename: str) -> None:
        """handle a click on a new video in the list loaded into the main window dock"""
        try:
            self._central_widget.load_video(self._project.video_manager.video_path(filename))
        except OSError as e:
            self.display_status_message(f"Unable to load video: {e}")
            self._project_load_error_callback(e)

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
        self._settings_action.setEnabled(True)
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

        # Note: Process pool is warmed on MainWindow startup, no need to warm here

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

    def _update_recent_projects(self) -> None:
        """update the contents of the Recent Projects menu"""
        self._open_recent_menu.clear()
        recent_projects = self._settings.value(RECENT_PROJECTS_KEY, [], type=list)

        # add menu action for each of the recent projects
        for project_path in recent_projects:
            action = self._open_recent_menu.addAction(project_path)
            # Use lambda to pass project_path explicitly instead of using sender()
            action.triggered.connect(
                lambda checked=False, path=project_path: self.open_project(path)
            )

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

    def _search_hit_loaded(self, search_hit) -> None:
        """Update the selected video in the video list when a search hit is loaded."""
        if search_hit is not None:
            self.video_list.select_video(search_hit.file, suppress_event=True)

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle the close event for the main window.

        Ensures proper cleanup of the process pool before the application exits.
        Note: The warm-up thread is a daemon, so it won't block exit.
        """
        logger.debug("[MainWindow] closeEvent: shutting down process pool")
        # Shutdown the process pool with wait=False and cancel_futures=True
        # The daemon warm-up thread (if still running) will be killed automatically
        self._process_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def on_project_settings_changed(self) -> None:
        """Slot called when project settings are changed via ProjectSettingsDialog.

        Called when settings are changed, in case any UI updates are needed.
        """
        # changing the settings can affect training thresholds, so the train button state needs to be updated
        self._central_widget.set_train_button_enabled_state()

    def on_app_settings_changed(self) -> None:
        """Slot called when application settings are changed via JabsSettingsDialog.

        Called when settings are saved, in case any UI updates are needed.
        """
        if self._project is not None:
            # check to see if the session tracking setting has changed
            session_tracking_enabled = bool(
                self._settings.value(SESSION_TRACKING_ENABLED_KEY, False, type=bool)
            )
            if self._project.session_tracker.enabled != session_tracking_enabled:
                if session_tracking_enabled:
                    # if a project is already loaded and user is enabling session tracking,
                    # they need to open the project again to enable session tracking.
                    MessageDialog.warning(
                        self,
                        "Session Tracking Enabled",
                        "Session Tracking Enabled: Please reopen the project to start tracking for the current project.",
                    )
                else:
                    # if session tracking was just disabled, we stop logging new events
                    # unlike starting session tracking, we can just disable it on the fly
                    self._project.session_tracker.enabled = False
