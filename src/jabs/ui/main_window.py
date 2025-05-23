import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeyEvent

from jabs.constants import ORG_NAME, RECENT_PROJECTS_MAX
from jabs.feature_extraction.landmark_features import LandmarkFeatureGroup
from jabs.project import export_training_data
from jabs.utils import FINAL_TRAIN_SEED, get_bool_env_var, hide_stderr
from jabs.version import version_str

from .about_dialog import AboutDialog
from .archive_behavior_dialog import ArchiveBehaviorDialog
from .central_widget import CentralWidget
from .license_dialog import LicenseAgreementDialog
from .project_loader_thread import ProjectLoaderThread
from .user_guide_viewer_widget import UserGuideDialog
from .video_list_widget import VideoListDockWidget

USE_NATIVE_FILE_DIALOG = get_bool_env_var("JABS_NATIVE_FILE_DIALOG", True)

RECENT_PROJECTS_KEY = "recent_projects"
LICENSE_ACCEPTED_KEY = "license_accepted"
LICENSE_VERSION_KEY = "license_version"


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

    def __init__(self, app_name: str, app_name_long: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle(f"{app_name_long} {version_str()}")
        self._central_widget = CentralWidget()
        self.setCentralWidget(self._central_widget)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setUnifiedTitleAndToolBarOnMac(True)

        self._app_name = app_name
        self._app_name_long = app_name_long
        self._project = None
        self._project_loader_thread = None
        self._progress_dialog = None

        self._status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self._status_bar)

        self._user_guide_window = None

        self._settings = QtCore.QSettings(ORG_NAME, app_name)

        # setup menu bar
        menu = self.menuBar()

        app_menu = menu.addMenu(self._app_name)
        file_menu = menu.addMenu("File")
        view_menu = menu.addMenu("View")
        feature_menu = menu.addMenu("Features")

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

        # exit action
        exit_action = QtGui.QAction(f" &Quit {self._app_name}", self)
        exit_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(QtCore.QCoreApplication.quit)
        app_menu.addAction(exit_action)

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

        # video playlist menu item
        self.view_playlist = QtGui.QAction("View Playlist", self)
        self.view_playlist.setCheckable(True)
        self.view_playlist.triggered.connect(self._toggle_video_list)
        view_menu.addAction(self.view_playlist)

        self.show_track = QtGui.QAction("Show Track", self)
        self.show_track.setCheckable(True)
        self.show_track.triggered.connect(self._toggle_track)
        view_menu.addAction(self.show_track)

        self.overlay_pose = QtGui.QAction("Overlay Pose", self)
        self.overlay_pose.setCheckable(True)
        self.overlay_pose.triggered.connect(self._toggle_pose_overlay)
        view_menu.addAction(self.overlay_pose)

        self.overlay_landmark = QtGui.QAction("Overlay Landmarks", self)
        self.overlay_landmark.setCheckable(True)
        self.overlay_landmark.triggered.connect(self._toggle_landmark_overlay)
        view_menu.addAction(self.overlay_landmark)

        # Test add check mark for overlay segmentation
        self.overlay_segmentation = QtGui.QAction("Overlay Segmentation", self)
        self.overlay_segmentation.setCheckable(True)
        self.overlay_segmentation.triggered.connect(self._toggle_segmentation_overlay)
        view_menu.addAction(self.overlay_segmentation)

        # Feature subset actions
        # All these settings should be updated whenever the behavior_changed event occurs
        self._central_widget.controls.behavior_changed.connect(
            self.behavior_changed_event
        )
        self._central_widget.controls.new_behavior_label.connect(
            self.behavior_label_add_event
        )

        self.enable_cm_units = QtGui.QAction("CM Units", self)
        self.enable_cm_units.setCheckable(True)
        self.enable_cm_units.triggered.connect(self._toggle_cm_units)
        feature_menu.addAction(self.enable_cm_units)

        self.enable_window_features = QtGui.QAction("Enable Window Features", self)
        self.enable_window_features.setCheckable(True)
        self.enable_window_features.triggered.connect(self._toggle_window_features)
        feature_menu.addAction(self.enable_window_features)

        self.enable_fft_features = QtGui.QAction("Enable Signal Features", self)
        self.enable_fft_features.setCheckable(True)
        self.enable_fft_features.triggered.connect(self._toggle_fft_features)
        feature_menu.addAction(self.enable_fft_features)

        self.enable_social_features = QtGui.QAction("Enable Social Features", self)
        self.enable_social_features.setCheckable(True)
        self.enable_social_features.triggered.connect(self._toggle_social_features)
        feature_menu.addAction(self.enable_social_features)

        # Static objects
        enable_landmark_features = {}
        for landmark_name in LandmarkFeatureGroup.feature_map:
            landmark_action = QtGui.QAction(
                f"Enable {landmark_name.capitalize()} Features", self
            )
            landmark_action.setCheckable(True)
            landmark_action.triggered.connect(self._toggle_static_object_feature)
            feature_menu.addAction(landmark_action)
            enable_landmark_features[landmark_name] = landmark_action
        self.enable_landmark_features = enable_landmark_features

        self.enable_segmentation_features = QtGui.QAction(
            "Enable Segmentation Features", self
        )
        self.enable_segmentation_features.setCheckable(True)
        self.enable_segmentation_features.triggered.connect(
            self._toggle_segmentation_features
        )
        feature_menu.addAction(self.enable_segmentation_features)

        # playlist widget added to dock on left side of main window
        self.video_list = VideoListDockWidget()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.video_list)
        self.video_list.setFloating(False)
        self.video_list.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
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

    def keyPressEvent(self, event: QKeyEvent):
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
        ]:
            self.centralWidget().keyPressEvent(event)
        elif key == Qt.Key.Key_T:
            self.show_track.trigger()
        elif key == Qt.Key.Key_P:
            self.overlay_pose.trigger()
        elif key == Qt.Key.Key_L:
            self.overlay_landmark.trigger()
        else:
            # anything else pass on to the super class keyPressEvent
            super().keyPressEvent(event)

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

    def open_project(self, project_path: str):
        """open a new project directory"""
        self._progress_dialog = QtWidgets.QProgressDialog(
            "Loading project...", None, 0, 0, self
        )
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.installEventFilter(self)
        self._progress_dialog.show()
        self._project_loader_thread = ProjectLoaderThread(project_path)
        self._project_loader_thread.project_loaded.connect(
            self._project_loaded_callback
        )
        self._project_loader_thread.load_error.connect(
            self._project_load_error_callback
        )
        self._project_loader_thread.start()

    def behavior_changed_event(self, new_behavior: str):
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
        self.enable_segmentation_features.setChecked(
            behavior_metadata.get("segmentation", False)
        )
        static_settings = behavior_metadata.get("static_objects", {})
        for static_object, menu_item in self.enable_landmark_features.items():
            menu_item.setChecked(static_settings.get(static_object, False))

    def behavior_label_add_event(self, behaviors: list[str]):
        """handle project updates required when user adds new behavior labels"""
        # check for new behaviors
        for behavior in behaviors:
            if (
                behavior
                not in self._project.settings_manager.project_settings["behavior"]
            ):
                # save new behavior with default settings
                self._project.settings_manager.save_behavior(behavior, {})

    def display_status_message(self, message: str, duration: int = 3000):
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
        self._status_bar.showMessage(message, duration)

    def clear_status_bar(self):
        """clear the status bar message

        Returns:
            None
        """
        self._status_bar.clearMessage()

    def _show_project_open_dialog(self):
        """prompt the user to select a project directory and open it"""
        options = QtWidgets.QFileDialog.Options()
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

    def _show_about_dialog(self):
        dialog = AboutDialog(f"{self._app_name_long} ({self._app_name})")
        dialog.exec_()

    def _open_user_guide(self):
        """show the user guide document in a separate window"""
        if self._user_guide_window is None:
            self._user_guide_window = UserGuideDialog(
                f"{self._app_name_long} ({self._app_name})"
            )
        self._user_guide_window.show()

    def _export_training_data(self):
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

    def _toggle_video_list(self, checked: bool):
        """show/hide video list"""
        if not checked:
            # user unchecked
            self.video_list.hide()
        else:
            # user checked
            self.video_list.show()

    def _toggle_track(self, checked: bool):
        """show/hide track overlay for subject."""
        self._central_widget.show_track(checked)

    def _toggle_pose_overlay(self, checked: bool):
        """show/hide pose overlay for subject."""
        self._central_widget.overlay_pose(checked)

    def _toggle_landmark_overlay(self, checked: bool):
        """show/hide landmark features."""
        self._central_widget.overlay_landmarks(checked)

    def _toggle_segmentation_overlay(self, checked: bool):
        """show/hide segmentation overlay for subject."""
        self._central_widget.overlay_segmentation(checked)

    def _toggle_cm_units(self, checked: bool):
        """toggle project to use pixel units."""
        # TODO: Warn the user that features may need to be re-calculated
        self._project.save_behavior(
            self._central_widget.behavior, {"cm_units": checked}
        )

    def _toggle_social_features(self, checked: bool):
        """toggle project to use social features."""
        self._project.save_behavior(self._central_widget.behavior, {"social": checked})

    def _toggle_window_features(self, checked: bool):
        """toggle project to use window features."""
        self._project.save_behavior(self._central_widget.behavior, {"window": checked})

    def _toggle_fft_features(self, checked: bool):
        """toggle project to use fft features."""
        self._project.save_behavior(self._central_widget.behavior, {"fft": checked})

    def _toggle_segmentation_features(self, checked: bool):
        """toggle project to use segmentation features."""
        self._project.save_behavior(
            self._central_widget.behavior, {"segmentation": checked}
        )

    def _toggle_static_object_feature(self, checked: bool):
        """toggle project to use a specific static object feature set."""
        # get the key from the caller
        key = self.sender().text().split(" ")[1].lower()
        all_object_settings = self._project.get_behavior(
            self._central_widget.behavior
        ).get("static_objects", {})
        all_object_settings[key] = checked
        self._project.save_behavior(
            self._central_widget.behavior, {"static_objects": all_object_settings}
        )

    def _video_list_selection(self, filename: str):
        """handle a click on a new video in the list loaded into the main window dock"""
        try:
            self._central_widget.load_video(
                self._project.video_manager.video_path(filename)
            )
        except OSError as e:
            self.display_status_message(f"Unable to load video: {e}")
            self._project_load_error_callback(e)

    def _open_archive_behavior_dialog(self):
        dialog = ArchiveBehaviorDialog(self._central_widget.behaviors)
        dialog.behavior_archived.connect(self._archive_behavior_callback)
        dialog.exec_()

    def _archive_behavior_callback(self, behavior: str):
        self._project.archive_behavior(behavior)
        self._central_widget.remove_behavior(behavior)

    def _project_loaded_callback(self):
        """Callback function to be called when the project is loaded."""
        self._project = self._project_loader_thread.project
        self._project_loader_thread = None

        # The central_widget updates main_control_widget
        self.centralWidget().set_project(self._project)
        self.video_list.set_project(self._project)

        # Update which controls should be available
        self._archive_behavior.setEnabled(True)
        self.enable_cm_units.setEnabled(self._project.feature_manager.is_cm_unit)
        self.enable_social_features.setEnabled(
            self._project.feature_manager.can_use_social_features
        )
        self.enable_segmentation_features.setEnabled(
            self._project.feature_manager.can_use_segmentation_features
        )
        available_objects = self._project.feature_manager.static_objects
        for static_object, menu_item in self.enable_landmark_features.items():
            if static_object in available_objects:
                menu_item.setEnabled(True)
            else:
                menu_item.setEnabled(False)

        # update the recent project menu
        self._add_recent_project(self._project.project_paths.project_dir)
        self._progress_dialog.close()

    def _project_load_error_callback(self, error: Exception):
        """Callback function to be called when the project fails to load."""
        self._project_loader_thread = None
        self._progress_dialog.close()
        QtWidgets.QMessageBox.critical(self, "Error loading project", str(error))

    def show_license_dialog(self):
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

        return result

    def _update_recent_projects(self):
        """update the contents of the Recent Projects menu"""
        self._open_recent_menu.clear()
        recent_projects = self._settings.value(RECENT_PROJECTS_KEY, [], type=list)

        # add menu action for each of the recent projects
        for project_path in recent_projects:
            action = self._open_recent_menu.addAction(project_path)
            action.setData(project_path)
            action.triggered.connect(self._open_recent_project)

    def _add_recent_project(self, project_path: Path):
        """add a project to the recent projects list"""
        # project path in the _project_loaded_callback is a Path object, Qt needs a string to add to the menu
        path_str = str(project_path)

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

    def _open_recent_project(self):
        """open a recent project"""
        action = self.sender()
        if isinstance(action, QAction):
            project_path = action.data()
            if project_path:
                self.open_project(project_path)
