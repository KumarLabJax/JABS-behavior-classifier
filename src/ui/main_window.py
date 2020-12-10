import sys

from PyQt5 import QtWidgets, QtCore

from src.project import Project, export_training_data
from src.version import version_str
from .about_dialog import AboutDialog
from .central_widget import CentralWidget
from .video_list_widget import VideoListDockWidget


class MainWindow(QtWidgets.QMainWindow):

    loadVideoAsyncSignal = QtCore.pyqtSignal(str)

    def __init__(self, app_name="Behavior Classifier", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle(f"{app_name} ({version_str()})")
        self._central_widget = CentralWidget()
        self.setCentralWidget(self._central_widget)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._app_name = app_name

        self.setUnifiedTitleAndToolBarOnMac(True)

        self._project = None

        self.loadVideoAsyncSignal.connect(self._load_video_async,
                                          QtCore.Qt.QueuedConnection)

        menu = self.menuBar()

        app_menu = menu.addMenu(self._app_name)
        file_menu = menu.addMenu('File')
        view_menu = menu.addMenu('View')

        # save action
        self.save_action = QtWidgets.QAction('&Save Labels', self)
        self.save_action.setShortcut('Ctrl+S')
        self.save_action.setStatusTip('Save Labels')
        self.save_action.triggered.connect(self._save_project)
        self.save_action.setEnabled(False)
        file_menu.addAction(self.save_action)

        # open action
        open_action = QtWidgets.QAction('&Open Project', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open Project')
        open_action.triggered.connect(self._show_project_open_dialog)
        file_menu.addAction(open_action)

        # about app
        about_action = QtWidgets.QAction(f' &About {self._app_name}', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self._show_about_dialog)
        app_menu.addAction(about_action)

        # exit action
        exit_action = QtWidgets.QAction(' &Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(QtWidgets.qApp.quit)
        app_menu.addAction(exit_action)

        # export training data action
        self._export_training = QtWidgets.QAction('Export Training Data', self)
        self._export_training.setShortcut('Ctrl+T')
        self._export_training.setStatusTip('Export training data for this classifier')
        self._export_training.setEnabled(False)
        self._export_training.triggered.connect(self._export_training_data)
        file_menu.addAction(self._export_training)

        # video playlist menu item
        self.view_playlist = QtWidgets.QAction('View Playlist', self,
                                               checkable=True)
        self.view_playlist.triggered.connect(self._toggle_video_list)

        view_menu.addAction(self.view_playlist)

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
        self._central_widget.export_training_status_change.connect(self._export_training.setEnabled)

    def keyPressEvent(self, event):
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
            QtCore.Qt.Key_L,
            QtCore.Qt.Key_T
        ]:
            self.centralWidget().keyPressEvent(event)

        else:
            # anything else pass on to the super class keyPressEvent
            super(MainWindow, self).keyPressEvent(event)

    def open_project(self, project_path):
        """ open a new project directory """
        self._project = Project(project_path)
        self.centralWidget().set_project(self._project)
        self.video_list.set_project(self._project)
        self.save_action.setEnabled(True)

    def _show_project_open_dialog(self):
        """ prompt the user to select a project directory and open it """
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Project Directory", options=options)

        if directory:
            self.open_project(directory)

    def _show_about_dialog(self):
        dialog = AboutDialog(self._app_name)
        dialog.exec_()

    def _save_project(self):
        """
        save current project state. Handles the File->Save menu action triggered
        signal.
        """

        # save the labels for the active video
        current_video_labels = self.centralWidget().get_labels()
        self._project.save_annotations(current_video_labels)

        # save labels for any other videos that have been worked on this session
        self._project.save_cached_annotations()

        # save other project metadata
        settings = self._project.metadata

        settings['selected_behavior'] = self._central_widget.behavior()
        settings['behaviors'] = self._central_widget.behavior_labels()
        settings['classifier'] = self._central_widget.classifier_type.name

        self._project.save_metadata(settings)

    def _export_training_data(self):
        # TODO make window_size configurable
        # (needs to be set based on user preferences for specific behavior)
        window_size = 5

        try:
            export_training_data(self._project, self._central_widget.behavior(),
                                 window_size, self._central_widget.c_type)
        except OSError as e:
            print(f"Unable to export training data: {e}", file=sys.stderr)

    def _toggle_video_list(self, checked):
        """ show/hide video list """
        if not checked:
            # user unchecked
            self.video_list.hide()
        else:
            # user checked
            self.video_list.show()

    def _video_list_selection(self, filename):
        """
        handle a click on a new video in the by sending a signal to an
        asynchronous event handler
        """
        self.loadVideoAsyncSignal.emit(str(filename))

    @QtCore.pyqtSlot(str)
    def _load_video_async(self, filename):
        """ process signal requesting to load a new video file """
        try:
            self.centralWidget().load_video(self._project.video_path(filename))
        except OSError as e:
            # TODO: display a dialog box or status bar message
            print(e)
