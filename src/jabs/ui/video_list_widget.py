from PySide6 import QtCore, QtWidgets

from jabs.behavior_search import SearchHit


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
        self._suppress_selection_event = False
        self.file_list.currentItemChanged.connect(self._selection_changed)

    def _selection_changed(self, current, _):
        """Emit signal when the selected video changes."""
        if self._suppress_selection_event:
            return
        if current:
            video = current.data(QtCore.Qt.ItemDataRole.UserRole)
            self.selectionChanged.emit(video)

    def set_project(self, project):
        """Update the video list with the active project's videos and select first video in list."""
        self._project = project
        self.file_list.clear()

        for video in self._project.video_manager.videos:
            item = QtWidgets.QListWidgetItem(video)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, video)
            self.file_list.addItem(item)

        if self._project.video_manager.videos:
            self.file_list.setCurrentRow(0)

    def select_video(self, key, suppress_event: bool = False):
        """
        Select the video in the list whose UserRole data matches `key`.

        If silence_event is True, suppress the selectionChanged signal.

        Args:
            key: The key to match against the UserRole data of the list items.
            suppress_event: If True, suppress the selectionChanged signal during selection.
        """
        self._suppress_selection_event = suppress_event
        try:
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.data(QtCore.Qt.ItemDataRole.UserRole) == key:
                    self.file_list.setCurrentItem(item)
                    break
        finally:
            self._suppress_selection_event = False

    def show_search_results(self, search_results: list[SearchHit]):
        """Update the video list with search results.

        This will cause a hit count indicator to appear to the
        right of each video in the list having at least one hit.

        Args:
            search_results: A list of SearchHit objects
        """
        count_map = {}
        for hit in search_results:
            count_map[hit.file] = count_map.get(hit.file, 0) + 1

        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            video = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if video in count_map:
                item.setText(f"{video} ({count_map[video]})")
            else:
                item.setText(video)
