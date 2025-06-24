from PySide6 import QtCore, QtGui, QtWidgets

from jabs.behavior_search import SearchHit


class _VideoListWidget(QtWidgets.QListWidget):
    """QListView that has been modified to not allow deselecting current selection without selecting a new row"""

    class HighlightTextDelegate(QtWidgets.QStyledItemDelegate):
        """Custom item delegate that highlights the background and text color for selected items.

        Changes the default highlighting of the selected item in the list, ensuring selected video name is
        clearly visible by drawing its text with the highlighted text color.
        """

        def paint(
            self,
            painter: QtGui.QPainter,
            option: QtWidgets.QStyleOptionViewItem,
            index: QtCore.QModelIndex,
        ):
            """Paints the text.

            Paints text, highlighting the background and text color
            for selected items to ensure visibility. For selected rows, fills the
            background with the highlight color and draws the text using the
            highlighted text color. For unselected rows, uses the default rendering.

            Args:
                painter: The QPainter object used for drawing.
                option: The style options for the item.
                index: The model index of the item being painted.
            """
            if option.state & QtWidgets.QStyle.StateFlag.State_Selected:  # type: ignore[attr-defined]
                painter.save()

                # Fill background with current accent color
                painter.fillRect(option.rect, option.palette.accent())  # type: ignore[attr-defined]

                # Set pen to highlighted text color
                painter.setPen(option.palette.color(QtGui.QPalette.ColorRole.HighlightedText))  # type: ignore[attr-defined]

                # Draw the text manually
                text = index.data(QtCore.Qt.ItemDataRole.DisplayRole)
                rect = option.rect.adjusted(4, 0, -4, 0)  # type: ignore[attr-defined]
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignVCenter, text)
                painter.restore()
            else:
                super().paint(painter, option, index)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setSortingEnabled(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setItemDelegate(self.HighlightTextDelegate(self))

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
        self._project = None
        self._suppress_selection_event = False

        self._video_filter_box = QtWidgets.QLineEdit(self)
        self._video_filter_box.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self._video_filter_box.setClearButtonEnabled(True)
        self._video_filter_box.setPlaceholderText("Filter videos...")
        self._file_list = _VideoListWidget(self)

        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video_filter_box)
        layout.addWidget(self._file_list)
        self.setWidget(container)

        self._file_list.currentItemChanged.connect(self._selection_changed)
        self._video_filter_box.textChanged.connect(self._filter_list)

    def _selection_changed(self, current, _):
        """Emit signal when the selected video changes."""
        if self._suppress_selection_event:
            return
        if current:
            video = current.data(QtCore.Qt.ItemDataRole.UserRole)
            self.selectionChanged.emit(video)
        else:
            self.selectionChanged.emit("")

    def _filter_list(self, text):
        """Filter the video list based on the text entered in the filter box."""
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def set_project(self, project):
        """Update the video list with the active project's videos and select first video in list."""
        self._suppress_selection_event = True
        self._project = project
        self._file_list.clear()
        self._suppress_selection_event = False
        self._video_filter_box.clear()

        for video in self._project.video_manager.videos:
            item = QtWidgets.QListWidgetItem(video)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, video)
            self._file_list.addItem(item)

        if self._project.video_manager.videos:
            self._file_list.setCurrentRow(0)

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
            for i in range(self._file_list.count()):
                item = self._file_list.item(i)
                if item.data(QtCore.Qt.ItemDataRole.UserRole) == key:
                    self._file_list.setCurrentItem(item)
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

        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            video = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if video in count_map:
                item.setText(f"{video} ({count_map[video]})")
            else:
                item.setText(video)
