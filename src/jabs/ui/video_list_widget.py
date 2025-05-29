from PySide6 import QtCore, QtWidgets


class _VideoListWidget(QtWidgets.QListWidget):
    """QListView that has been modified to not allow deselecting current selection without selecting a new row"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setSortingEnabled(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        # don't take focus otherwise up/down arrows will change video
        # when the user is intending to skip forward/back frames
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def selectionCommand(self, index, event=None):
        """Override to prevent deselection of the current row."""
        if self.selectedIndexes() and self.selectedIndexes()[0].row() == index.row():
            return QtCore.QItemSelectionModel.SelectionFlag.NoUpdate
        return super().selectionCommand(index, event)


class VideoListDockWidget(QtWidgets.QDockWidget):
    """dock for listing video files associated with the project."""

    selectionChanged = QtCore.Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Project Videos")
        self.file_list = _VideoListWidget(self)
        self.setWidget(self.file_list)
        self._project = None
        self.file_list.currentItemChanged.connect(self._selection_changed)

    def _selection_changed(self, current, _):
        """Emit signal when the selected video changes."""
        if current:
            self.selectionChanged.emit(current.text())

    def set_project(self, project):
        """Update the video list with the active project's videos and select first video in list."""
        self._project = project
        self.file_list.clear()
        self.file_list.addItems(self._project.video_manager.videos)
        if self._project.video_manager.videos:
            self.file_list.setCurrentRow(0)
