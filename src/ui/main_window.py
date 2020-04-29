from PyQt5 import QtWidgets, QtCore

from src.labeler.project import Project
from .playlist_widget import PlaylistWidget
from .central_widget import CentralWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Behavior Classifier")
        self.setCentralWidget(CentralWidget())

        self.setUnifiedTitleAndToolBarOnMac(True)

        self._project = None

        menu = self.menuBar()

        app_menu = menu.addMenu('Behavior Classifier')
        file_menu = menu.addMenu('File')
        view_menu = menu.addMenu('View')

        # exit action
        exit_action = QtWidgets.QAction(' &Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(QtWidgets.qApp.quit)
        app_menu.addAction(exit_action)

        # video playlist menu item
        self.view_playlist = QtWidgets.QAction('View Playlist', self,
                                               checkable=True)
        self.view_playlist.triggered.connect(self._togglePlaylist)
        view_menu.addAction(self.view_playlist)

        # playlist widget added to dock on left side of main window
        self.playlist = PlaylistWidget()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.playlist)
        self.playlist.setFloating(False)

        # if the playlist visibility changes, make sure the view_playlists
        # checkmark is set correctly
        self.playlist.visibilityChanged.connect(self.view_playlist.setChecked)

        # handle event where user selects a different video in the playlist
        self.playlist.selectionChanged.connect(self._video_playlist_selection)

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
            QtCore.Qt.Key_C
        ]:
            self.centralWidget().keyPressEvent(event)

        else:
            # anything else pass on to the super keyPressEvent
            super(MainWindow, self).keyPressEvent(event)

    def open_project(self, project_path):
        self._project = Project(project_path)
        self.playlist.set_project(self._project)

    def _togglePlaylist(self, checked):
        if not checked:
            # user unchecked
            self.playlist.hide()
        else:
            # user checked
            self.playlist.show()

    def _video_playlist_selection(self, filename):
        self.centralWidget().load_video(self._project.make_path(filename))
