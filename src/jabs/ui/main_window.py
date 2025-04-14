import sys

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent

from jabs.project import export_training_data
from jabs.feature_extraction.landmark_features import LandmarkFeatureGroup
from jabs.version import version_str
from jabs.utils import FINAL_TRAIN_SEED, get_bool_env_var, hide_stderr
from .about_dialog import AboutDialog
from .central_widget import CentralWidget
from .project_loader_thread import ProjectLoaderThread
from .video_list_widget import VideoListDockWidget
from .archive_behavior_dialog import ArchiveBehaviorDialog
from .license_dialog import LicenseAgreementDialog
from .user_guide_viewer_widget import UserGuideDialog


USE_NATIVE_FILE_DIALOG = get_bool_env_var("JABS_NATIVE_FILE_DIALOG", True)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app_name: str, app_name_long: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle(f"{app_name_long} {version_str()}")
        self._central_widget = CentralWidget()
        self.setCentralWidget(self._central_widget)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setUnifiedTitleAndToolBarOnMac(True)

        self._app_name = app_name
        self._app_name_long = app_name_long
        self._project = None
        self._project_loader_thread = None
        self._progress_dialog = None

        self._status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self._status_bar)

        self._user_guide_window = None

        menu = self.menuBar()

        app_menu = menu.addMenu(self._app_name)
        file_menu = menu.addMenu('File')
        view_menu = menu.addMenu('View')
        feature_menu = menu.addMenu('Features')

        # open action
        open_action = QtGui.QAction('&Open Project', self)
        open_action.setShortcut(QtGui.QKeySequence(Qt.CTRL | Qt.Key_O))
        open_action.setStatusTip('Open Project')
        open_action.triggered.connect(self._show_project_open_dialog)
        file_menu.addAction(open_action)

        # about app
        about_action = QtGui.QAction(f' &About {self._app_name}', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self._show_about_dialog)
        app_menu.addAction(about_action)

        # user guide
        user_guide_action = QtGui.QAction(' &User Guide', self)
        user_guide_action.setStatusTip('Open User Guide')
        user_guide_action.setShortcut(QtGui.QKeySequence(Qt.CTRL | Qt.Key_U))
        user_guide_action.triggered.connect(self._open_user_guide)
        app_menu.addAction(user_guide_action)

        # exit action
        exit_action = QtGui.QAction(f' &Quit {self._app_name}', self)
        exit_action.setShortcut(QtGui.QKeySequence(Qt.CTRL | Qt.Key_Q))
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(QtCore.QCoreApplication.quit)
        app_menu.addAction(exit_action)

        # export training data action
        self._export_training = QtGui.QAction('Export Training Data', self)
        self._export_training.setShortcut(QtGui.QKeySequence(Qt.CTRL | Qt.Key_T))
        self._export_training.setStatusTip('Export training data for this classifier')
        self._export_training.setEnabled(False)
        self._export_training.triggered.connect(self._export_training_data)
        file_menu.addAction(self._export_training)

        # archive behavior action
        self._archive_behavior = QtGui.QAction('Archive Behavior', self)
        self._archive_behavior.setStatusTip('Open Archive Behavior Dialog')
        self._archive_behavior.setEnabled(False)
        self._archive_behavior.triggered.connect(self._open_archive_behavior_dialog)
        file_menu.addAction(self._archive_behavior)

        # video playlist menu item
        self.view_playlist = QtGui.QAction('View Playlist', self)
        self.view_playlist.setCheckable(True)
        self.view_playlist.triggered.connect(self._toggle_video_list)
        view_menu.addAction(self.view_playlist)

        self.show_track = QtGui.QAction('Show Track', self)
        self.show_track.setCheckable(True)
        self.show_track.triggered.connect(self._toggle_track)
        view_menu.addAction(self.show_track)

        self.overlay_pose = QtGui.QAction('Overlay Pose', self)
        self.overlay_pose.setCheckable(True)
        self.overlay_pose.triggered.connect(self._toggle_pose_overlay)
        view_menu.addAction(self.overlay_pose)

        self.overlay_landmark = QtGui.QAction('Overlay Landmarks', self)
        self.overlay_landmark.setCheckable(True)
        self.overlay_landmark.triggered.connect(self._toggle_landmark_overlay)
        view_menu.addAction(self.overlay_landmark)

        # Test add check mark for overlay segmentation
        self.overlay_segmentation = QtGui.QAction('Overlay Segmentation', self)
        self.overlay_segmentation.setCheckable(True)
        self.overlay_segmentation.triggered.connect(self._toggle_segmentation_overlay)
        view_menu.addAction(self.overlay_segmentation)

        # Feature subset actions
        # All these settings should be updated whenever the behavior_changed event occurs
        self._central_widget.controls.behavior_changed.connect(self.behavior_changed_event)
        self._central_widget.controls.new_behavior_label.connect(self.behavior_label_add_event)

        self.enable_cm_units = QtGui.QAction('CM Units', self)
        self.enable_cm_units.setCheckable(True)
        self.enable_cm_units.triggered.connect(self._toggle_cm_units)
        feature_menu.addAction(self.enable_cm_units)

        self.enable_window_features = QtGui.QAction('Enable Window Features', self)
        self.enable_window_features.setCheckable(True)
        self.enable_window_features.triggered.connect(self._toggle_window_features)
        feature_menu.addAction(self.enable_window_features)

        self.enable_fft_features = QtGui.QAction('Enable Signal Features', self)
        self.enable_fft_features.setCheckable(True)
        self.enable_fft_features.triggered.connect(self._toggle_fft_features)
        feature_menu.addAction(self.enable_fft_features)

        self.enable_social_features = QtGui.QAction('Enable Social Features', self)
        self.enable_social_features.setCheckable(True)
        self.enable_social_features.triggered.connect(self._toggle_social_features)
        feature_menu.addAction(self.enable_social_features)

        # Static objects
        enable_landmark_features = {}
        for landmark_name in LandmarkFeatureGroup._feature_map.keys():
            landmark_action = QtGui.QAction(f'Enable {landmark_name.capitalize()} Features', self)
            landmark_action.setCheckable(True)
            landmark_action.triggered.connect(self._toggle_static_object_feature)
            feature_menu.addAction(landmark_action)
            enable_landmark_features[landmark_name] = landmark_action
        self.enable_landmark_features = enable_landmark_features

        self.enable_segmentation_features = QtGui.QAction('Enable Segmentation Features', self)
        self.enable_segmentation_features.setCheckable(True)
        self.enable_segmentation_features.triggered.connect(self._toggle_segmentation_features)
        feature_menu.addAction(self.enable_segmentation_features)

        # playlist widget added to dock on left side of main window
        self.video_list = VideoListDockWidget()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.video_list)
        self.video_list.setFloating(False)
        self.video_list.setFeatures(
            QtWidgets.QDockWidget.NoDockWidgetFeatures |
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetMovable)

        # if the playlist visibility changes, make sure the view_playlists
        # check mark is set correctly
        self.video_list.visibilityChanged.connect(self.view_playlist.setChecked)

        # handle event where user selects a different video in the playlist
        self.video_list.selectionChanged.connect(self._video_list_selection)

        # handle event to set status of File-Export Training Data action
        self._central_widget.export_training_status_change.connect(
            self._export_training.setEnabled)

    def keyPressEvent(self, event: QKeyEvent):
        """
        override keyPressEvent so we can pass some key press events on
        to the centralWidget
        """
        key = event.key()

        # pass along some of the key press events to the central widget
        if key in [
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Space,
            QtCore.Qt.Key_Z,
            QtCore.Qt.Key_X,
            QtCore.Qt.Key_C,
            QtCore.Qt.Key_Escape,
            QtCore.Qt.Key_Question
        ]:
            self.centralWidget().keyPressEvent(event)
        elif key == QtCore.Qt.Key_T:
            self.show_track.trigger()
        elif key == QtCore.Qt.Key_P:
            self.overlay_pose.trigger()
        elif key == QtCore.Qt.Key_L:
            self.overlay_landmark.trigger()
        else:
            # anything else pass on to the super class keyPressEvent
            super(MainWindow, self).keyPressEvent(event)

    def open_project(self, project_path: str):
        """ open a new project directory """
        self._progress_dialog = QtWidgets.QProgressDialog(
            "Loading project...", None, 0, 0, self)
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self._progress_dialog.show()
        self._project_loader_thread = ProjectLoaderThread(project_path)
        self._project_loader_thread.project_loaded.connect(self._project_loaded_callback)
        self._project_loader_thread.load_error.connect(self._project_load_error_callback)
        self._project_loader_thread.start()

    def behavior_changed_event(self, new_behavior: str):
        """ menu items to change when a new behavior is selected. """
        # skip if no behavior assigned (only should occur during new project)
        if new_behavior is None or new_behavior == '':
            return

        # Populate settings based project data
        behavior_metadata = self._project.settings_manager.get_behavior(new_behavior)
        self.enable_cm_units.setChecked(behavior_metadata.get('cm_units', False))
        self.enable_window_features.setChecked(behavior_metadata.get('window', False))
        self.enable_fft_features.setChecked(behavior_metadata.get('fft', False))
        self.enable_social_features.setChecked(behavior_metadata.get('social', False))
        self.enable_segmentation_features.setChecked(behavior_metadata.get('segmentation', False))
        static_settings = behavior_metadata.get('static_objects', {})
        for static_object, menu_item in self.enable_landmark_features.items():
            menu_item.setChecked(static_settings.get(static_object, False))

    def behavior_label_add_event(self, behaviors: list[str]):
        """ handle project updates required when user adds new behavior labels """

        # check for new behaviors
        for behavior in behaviors:
            if behavior not in self._project.settings_manager.project_settings["behavior"].keys():
                # save new behavior with default settings
                self._project.settings_manager.save_behavior(behavior, {})

    def display_status_message(self, message: str, duration: int = 3000):
        """
        display a message in the main window status bar
        :param message: message to display
        :param duration: duration of the message in milliseconds. Use 0 to
        display the message until clear_status_bar() is called
        :return: None
        """
        if duration < 0:
            raise ValueError("duration must be >= 0")
        self._status_bar.showMessage(message, duration)

    def clear_status_bar(self):
        """
        clear the status bar message
        :return: None
        """
        self._status_bar.clearMessage()

    def _show_project_open_dialog(self):
        """ prompt the user to select a project directory and open it """
        options = QtWidgets.QFileDialog.Options()
        if not USE_NATIVE_FILE_DIALOG:
            options |= QtWidgets.QFileDialog.DontUseNativeDialog

        # on macOS QFileDialog can cause some error messages to be written to stderr but the dialog still works
        # so hide anything written to stderr while we're showing the dialog. we can use a Qt based file dialog instead
        # of the native one by setting the env variable USE_NATIVE_FILE_DIALOG to a non-true value (default is True)
        with hide_stderr():
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Project Directory", options=options)

        if directory:
            self.open_project(directory)

    def _show_about_dialog(self):
        dialog = AboutDialog(f"{self._app_name_long} ({self._app_name})")
        dialog.exec_()

    def _open_user_guide(self):
        """ show the user guide document in a separate window """
        if self._user_guide_window is None:
            self._user_guide_window = UserGuideDialog(
                f"{self._app_name_long} ({self._app_name})")
        self._user_guide_window.show()

    def _export_training_data(self):

        if not self._central_widget.classify_button_enabled:
            # classify button disabled, don't allow exporting training data
            QtWidgets.QMessageBox.warning(
                self, "Unable to export training data",
                "Classifier has not been trained, or classifier parameters "
                "have changed.\n\n"
                "You must train the classifier before export.")
            return

        try:
            out_path = export_training_data(self._project,
                                            self._central_widget.behavior,
                                            self._project.feature_manager.min_pose_version,
                                            self._central_widget.classifier_type,
                                            FINAL_TRAIN_SEED)
            self.display_status_message(f"Training data exported: {out_path}",
                                        5000)
        except OSError as e:
            print(f"Unable to export training data: {e}", file=sys.stderr)

    def _toggle_video_list(self, checked: bool):
        """ show/hide video list """
        if not checked:
            # user unchecked
            self.video_list.hide()
        else:
            # user checked
            self.video_list.show()

    def _toggle_track(self, checked: bool):
        """ show/hide track overlay for subject. """
        self._central_widget.show_track(checked)

    def _toggle_pose_overlay(self, checked: bool):
        """ show/hide pose overlay for subject. """
        self._central_widget.overlay_pose(checked)

    def _toggle_landmark_overlay(self, checked: bool):
        """ show/hide landmark features. """
        self._central_widget.overlay_landmarks(checked)

    def _toggle_segmentation_overlay(self, checked: bool):
        """ show/hide segmentation overlay for subject. """
        self._central_widget.overlay_segmentation(checked)

    def _toggle_cm_units(self, checked: bool):
        """ toggle project to use pixel units. """
        # TODO: Warn the user that features may need to be re-calculated
        self._project.save_behavior(self._central_widget.behavior, {'cm_units': checked})

    def _toggle_social_features(self, checked: bool):
        """ toggle project to use social features. """
        self._project.save_behavior(self._central_widget.behavior, {'social': checked})

    def _toggle_window_features(self, checked: bool):
        """ toggle project to use window features. """
        self._project.save_behavior(self._central_widget.behavior, {'window': checked})

    def _toggle_fft_features(self, checked: bool):
        """ toggle project to use fft features. """
        self._project.save_behavior(self._central_widget.behavior, {'fft': checked})

    def _toggle_segmentation_features(self, checked: bool):
        """ toggle project to use segmentation features. """
        self._project.save_behavior(self._central_widget.behavior, {'segmentation': checked})

    def _toggle_static_object_feature(self, checked: bool):
        """ toggle project to use a specific static object feature set. """
        # get the key from the caller
        key = self.sender().text().split(' ')[1].lower()
        all_object_settings = self._project.get_behavior(self._central_widget.behavior).get('static_objects', {})
        all_object_settings[key] = checked
        self._project.save_behavior(self._central_widget.behavior, {'static_objects': all_object_settings})

    def _video_list_selection(self, filename: str):
        """
        handle a click on a new video in the list loaded into the main
        window dock
        """
        try:
            self._central_widget.load_video(self._project.video_path(filename))
        except OSError as e:
            self.display_status_message(f"Unable to load video: {e}")

    def _open_archive_behavior_dialog(self):
        dialog = ArchiveBehaviorDialog(self._central_widget.behaviors)
        dialog.behavior_archived.connect(self._archive_behavior_callback)
        dialog.exec_()

    def _archive_behavior_callback(self, behavior: str):
        self._project.archive_behavior(behavior)
        self._central_widget.remove_behavior(behavior)

    def _project_loaded_callback(self):
        """
        Callback function to be called when the project is loaded.
        """
        self._project = self._project_loader_thread.project
        self._project_loader_thread = None

        # The central_widget updates main_control_widget
        self.centralWidget().set_project(self._project)
        self.video_list.set_project(self._project)

        # Update which controls should be available
        self._archive_behavior.setEnabled(True)
        self.enable_cm_units.setEnabled(self._project.feature_manager.is_cm_unit)
        self.enable_social_features.setEnabled(self._project.feature_manager.can_use_social_features)
        self.enable_segmentation_features.setEnabled(self._project.feature_manager.can_use_segmentation_features)
        available_objects = self._project.feature_manager.static_objects
        for static_object, menu_item in self.enable_landmark_features.items():
            if static_object in available_objects:
                menu_item.setEnabled(True)
            else:
                menu_item.setEnabled(False)
        self._progress_dialog.close()

    def _project_load_error_callback(self, error: Exception):
        """
        Callback function to be called when the project fails to load.
        """
        self._project_loader_thread = None
        self._progress_dialog.close()
        QtWidgets.QMessageBox.critical(
            self, "Error loading project", str(error))

    def show_license_dialog(self):
        dialog = LicenseAgreementDialog(self)
        result = dialog.exec_()
        return result
