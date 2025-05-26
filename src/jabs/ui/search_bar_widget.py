from PySide6 import QtCore, QtWidgets

from jabs.project.project import Project
from jabs.ui.behavior_search_query import BehaviorSearchQuery


class SearchBarWidget(QtWidgets.QWidget):
    """A custom widget for displaying a behavior search bar."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._project: Project = None
        self._search_query = None

        self.label = QtWidgets.QLabel("Searching for:")
        self.text_label = QtWidgets.QLabel("")
        font = self.text_label.font()
        font.setBold(True)
        self.text_label.setFont(font)

        self.prev_button = QtWidgets.QToolButton()
        self.prev_button.setText("<")
        self.prev_button.setToolTip("Find Previous")
        self.prev_button.clicked.connect(self._on_prev_clicked)

        self.next_button = QtWidgets.QToolButton()
        self.next_button.setText(">")
        self.next_button.setToolTip("Find Next")
        self.next_button.clicked.connect(self._on_next_clicked)

        self.done_button = QtWidgets.QToolButton()
        self.done_button.setText("Done")
        self.done_button.clicked.connect(self._on_done_clicked)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.addWidget(self.label)
        layout.addWidget(self.text_label)
        layout.addStretch()

        # Group prev/next tightly together
        btn_group = QtWidgets.QHBoxLayout()
        btn_group.setSpacing(0)
        btn_group.setContentsMargins(0, 0, 0, 0)
        btn_group.addWidget(self.prev_button)
        btn_group.addWidget(self.next_button)
        layout.addLayout(btn_group)

        layout.addWidget(self.done_button)

        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setObjectName("SearchBarWidget")
        self.setStyleSheet("""
            #SearchBarWidget {
                border-bottom: 1px solid #ccc;
            }
        """)

        self.update_behavior_search_query(None)

    @property
    def project(self) -> Project | None:
        """Get the current project."""
        return self._project

    def update_project(self, project: Project | None):
        """Update the current project."""
        self._project = project
        if self._search_query is not None:
            self.set_behavior_search_query(self._search_query)

    @property
    def behavior_search_query(self) -> BehaviorSearchQuery | None:
        """Get the current behavior search query."""
        return self._search_query

    def update_behavior_search_query(self, search_query: BehaviorSearchQuery | None):
        """Set the behavior search query and update the text label."""
        self._search_query = search_query
        if search_query is None:
            self.setVisible(False)
            self.text_label.setText("")
        else:
            self.setVisible(True)
            self.text_label.setText(search_query.describe())

    def _on_prev_clicked(self):
        print("Previous button clicked")

    def _on_next_clicked(self):
        print("Next button clicked")

    def _on_done_clicked(self):
        self.update_behavior_search_query(None)

    def _perform_search(self):
        """Perform the search based on the current query and project."""
        if self._project is None or self._search_query is None:
            return

        # Implement the search logic here
