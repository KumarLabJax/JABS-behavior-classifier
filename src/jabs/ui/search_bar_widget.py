import json
from collections.abc import Iterable
from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from jabs.project.project import Project
from jabs.ui.behavior_search_query import (
    BehaviorSearchQuery,
    LabelBehaviorSearchQuery,
    PredictionLabelSearchQuery,
)


@dataclass(frozen=True)
class _SearchHit:
    file: str
    identity: str
    start_frame: int
    end_frame: int

    @staticmethod
    def sorted_search_hits(hits: Iterable["_SearchHit"]) -> list["_SearchHit"]:
        """
        Return a list of search hits sorted by file, identity (numeric if possible), and start_frame.

        Args:
            hits: An iterable of _SearchHit objects.

        Returns:
            A list of _SearchHit objects sorted by file, identity, and start_frame.
            Identity is sorted numerically if possible, otherwise alphanumerically.
        """
        try:
            return sorted(
                hits,
                key=lambda hit: (hit.file, int(hit.identity), hit.start_frame),
            )
        except ValueError:
            # there is at least one identity that cannot be converted to an int
            return sorted(
                hits,
                key=lambda hit: (hit.file, hit.identity, hit.start_frame),
            )


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
            self.update_behavior_search_query(self._search_query)

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
            self.text_label.setText(_describe_query(search_query))
            self._perform_search()

    def _on_prev_clicked(self):
        print("Previous button clicked")

    def _on_next_clicked(self):
        print("Next button clicked")

    def _on_done_clicked(self):
        self.update_behavior_search_query(None)

    def _perform_search(self):
        """Perform the search based on the current query and project."""
        search_results = _SearchHit.sorted_search_hits(self._perform_search_gen())
        if search_results:
            print(f"Search found {len(search_results)} results.")
            # Here you could update the UI to display the search results
        else:
            print("No results found.")

    def _perform_search_gen(self):
        """Perform the search based on the current query and project.

        This is a generator that yields search hits.
        """
        if self._project is None or self._search_query is None:
            return

        match self._search_query:
            case LabelBehaviorSearchQuery() as label_query:
                print("Searching for labels...")

                if label_query.positive or label_query.negative:
                    video_manager = self._project.video_manager
                    sorted_videos = sorted(video_manager.videos)

                    for video in sorted_videos:
                        anno_path = video_manager.annotations_path(video)
                        if anno_path.exists():
                            print(f"Found annotations for {video} at {anno_path}")
                            with anno_path.open() as f:
                                anno_dict = json.load(f)

                            labels = anno_dict.get("labels", {})
                            for identity, identified_labels in labels.items():
                                for behavior, blocks in identified_labels.items():
                                    for block in blocks:
                                        block_matches_query = (
                                            behavior == label_query.behavior_label
                                            or label_query.behavior_label is None
                                        ) and (
                                            (label_query.positive and block["present"])
                                            or (
                                                label_query.negative
                                                and not block["present"]
                                            )
                                        )
                                        if block_matches_query:
                                            yield _SearchHit(
                                                file=video,
                                                identity=identity,
                                                start_frame=block["start"],
                                                end_frame=block["end"],
                                            )

            case PredictionLabelSearchQuery() as pred_query:  # noqa: F841
                print("Searching for predictions...")
                # Your prediction search logic goes here

            case _:
                print("Unknown query type or unsupported search.")


def _describe_query(query: BehaviorSearchQuery) -> str:
    """Return a descriptive string for the given search query."""
    match query:
        case LabelBehaviorSearchQuery(
            behavior_label=None, positive=True, negative=True
        ):
            return "All behaviors positive & negative labels"
        case LabelBehaviorSearchQuery(behavior_label=None, positive=True):
            return "All behaviors positive labels"
        case LabelBehaviorSearchQuery(behavior_label=None, negative=True):
            return "All behaviors negative labels"
        case LabelBehaviorSearchQuery(
            behavior_label=behavior_label, positive=True, negative=True
        ):
            return f"{behavior_label} & Not {behavior_label} labels"
        case LabelBehaviorSearchQuery(behavior_label=behavior_label, positive=True):
            return f"{behavior_label} labels"
        case LabelBehaviorSearchQuery(behavior_label=behavior_label, negative=True):
            return f"Not {behavior_label} labels"
        case PredictionLabelSearchQuery(
            prob_greater_value=gt, prob_less_value=lt, min_contiguous_frames=frames
        ):
            parts = []
            if gt is not None and lt is not None:
                parts.append(f"{gt} < behavior prob. < {lt}")
            elif gt is not None:
                parts.append(f"behavior prob. > {gt}")
            elif lt is not None:
                parts.append(f"behavior prob. < {lt}")
            if not parts:
                return "No Search"
            if frames is not None:
                parts.append(f"with at least {frames} contiguous frames")
            return " ".join(parts)
        case _:
            return "No Search"
