from PyQt5 import QtWidgets, QtCore


class _VideoListWidget(QtWidgets.QListWidget):
    """
    QListView that has been modified to not allow deselecting current selection
    without selecting a new row
    """
    def __init__(self):
        super(_VideoListWidget, self).__init__()
        # only allow one selection
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSortingEnabled(True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        # don't allow items to be edited
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)

    def selectionCommand(self, index, event=None):
        """ override QListView """
        selected = self.selectedIndexes()
        if len(selected) == 1 and selected[0].row() == index.row():
            # don't allow "no selection", so ignore clicks on the already
            # selected row to prevent Ctrl/Cmd + click deselection
            return QtCore.QItemSelectionModel.NoUpdate
        else:
            return super(_VideoListWidget, self).selectionCommand(index, event)


class PlaylistWidget(QtWidgets.QDockWidget):
    """
    dock for listing video files associated with the project.
    dock is floating and can
    """

    selectionChanged = QtCore.pyqtSignal(str)

    def __init__(self):
        super(PlaylistWidget, self).__init__()
        self.setWindowTitle("Project Videos")
        self.file_list = _VideoListWidget()
        self.setWidget(self.file_list)

        self._project = None

        # connect to the model selectionChanged signal
        self.file_list.currentItemChanged.connect(self._selection_changed)

    def _selection_changed(self, current, previous):
        """ signal main window that use changed selected video """
        self.selectionChanged.emit(current.text())

    def set_project(self, project):
        self._project = project
        self.file_list.clear()
        for video in self._project.videos:
            self.file_list.addItem(video)

        # select the first video in the list
        if len(self._project.videos):
            self.file_list.setCurrentRow(0)
