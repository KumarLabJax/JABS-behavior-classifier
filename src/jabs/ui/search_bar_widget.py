from PySide6 import QtCore, QtWidgets

from jabs.behavior_search import (
    BehaviorSearchQuery,
    LabelBehaviorSearchQuery,
    PredictionBehaviorSearchQuery,
    PredictionSearchKind,
    SearchHit,
    search_behaviors,
)
from jabs.project import Project


class SearchBarWidget(QtWidgets.QWidget):
    """A custom widget for displaying a behavior search bar."""

    # Signal emitted when the current search hit changes.
    # This will be either a SearchHit object or None if no hits are found.
    current_search_hit_changed = QtCore.Signal(object)
    search_results_changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._project: Project = None
        self._search_query = None
        self._search_results: list[SearchHit] = []
        self._current_result_index = 0

        self.label = QtWidgets.QLabel("Searching for:")
        self.text_label = QtWidgets.QLabel("")
        font = self.text_label.font()
        font.setBold(True)
        self.text_label.setFont(font)

        self.result_count_label = QtWidgets.QLabel("(Not found)")

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
        layout.addWidget(self.result_count_label)

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

        self.update_search(None)

    @property
    def project(self) -> Project | None:
        """Get the current project."""
        return self._project

    def update_project(self, project: Project | None):
        """Update the current project."""
        self._project = project
        if self._search_query is not None:
            self.update_search(self._search_query)

    @property
    def behavior_search_query(self) -> BehaviorSearchQuery | None:
        """Get the current behavior search query."""
        return self._search_query

    def update_search(self, search_query: BehaviorSearchQuery | None):
        """Set the behavior search query and update the text label."""
        self._search_query = search_query
        self._search_results = []
        self._current_result_index = 0

        if search_query is None:
            self.setVisible(False)
        else:
            self.setVisible(True)
            self.text_label.setText(_describe_query(search_query))
            self._search_results = search_behaviors(self._project, search_query)

        self._update_result_count_label()
        self.search_results_changed.emit(self._search_results)
        self.current_search_hit_changed.emit(self.current_search_hit)

    def _on_prev_clicked(self):
        if self._search_results and self._current_result_index > 0:
            self._current_result_index -= 1
            self._update_result_count_label()
            self.current_search_hit_changed.emit(self.current_search_hit)

    def _on_next_clicked(self):
        if self._search_results and self._current_result_index < len(self._search_results) - 1:
            self._current_result_index += 1
            self._update_result_count_label()
            self.current_search_hit_changed.emit(self.current_search_hit)

    def _on_done_clicked(self):
        self.update_search(None)

    def _update_result_count_label(self):
        if self._search_results:
            self.result_count_label.setText(
                f"({self._current_result_index + 1} of {len(self._search_results)})"
            )
        else:
            self.result_count_label.setText("(Not found)")

    @property
    def current_search_hit(self) -> SearchHit | None:
        """Get the current search hit based on the current index."""
        if self._current_result_index < len(self._search_results):
            return self._search_results[self._current_result_index]

        return None


def _describe_query(query: BehaviorSearchQuery) -> str:
    """Return a descriptive string for the given search query."""
    match query:
        case LabelBehaviorSearchQuery(behavior_label=None, positive=True, negative=True):
            return "All behaviors positive & negative labels"
        case LabelBehaviorSearchQuery(behavior_label=None, positive=True):
            return "All behaviors positive labels"
        case LabelBehaviorSearchQuery(behavior_label=None, negative=True):
            return "All behaviors negative labels"
        case LabelBehaviorSearchQuery(behavior_label=behavior_label, positive=True, negative=True):
            return f"{behavior_label} & Not {behavior_label} labels"
        case LabelBehaviorSearchQuery(behavior_label=behavior_label, positive=True):
            return f"{behavior_label} labels"
        case LabelBehaviorSearchQuery(behavior_label=behavior_label, negative=True):
            return f"Not {behavior_label} labels"
        case PredictionBehaviorSearchQuery(
            search_kind=search_kind,
            behavior_label=behavior_label,
            prob_greater_value=gt,
            prob_less_value=lt,
            min_contiguous_frames=frames,
        ):
            parts = []
            match search_kind:
                case PredictionSearchKind.POSITIVE_PREDICTION:
                    if behavior_label is None:
                        parts.append("Positive behavior prediction")
                    else:
                        parts.append(f"{behavior_label} prediction")
                case PredictionSearchKind.NEGATIVE_PREDICTION:
                    if behavior_label is None:
                        parts.append("Negative behavior prediction")
                    else:
                        parts.append(f"Not {behavior_label} prediction")
                case PredictionSearchKind.PROBABILITY_RANGE:
                    if behavior_label is not None:
                        parts.append(f"{behavior_label} prediction")

                    if gt is not None and lt is not None:
                        parts.append(f"{gt} < behavior prob. < {lt}")
                    elif gt is not None:
                        parts.append(f"behavior prob. > {gt}")
                    elif lt is not None:
                        parts.append(f"behavior prob. < {lt}")

            if frames is not None and frames > 1:
                parts.append(f"with at least {frames} contiguous frames")

            return " ".join(parts)
        case _:
            return "No Search"
