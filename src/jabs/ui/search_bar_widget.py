from PySide6 import QtCore, QtWidgets

from jabs.behavior_search import (
    BehaviorSearchQuery,
    LabelBehaviorSearchQuery,
    PredictionBehaviorSearchQuery,
    PredictionSearchKind,
    SearchHit,
    TimelineAnnotationSearchQuery,
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

        self._project: Project | None = None
        self._search_query: BehaviorSearchQuery | None = None
        self._search_results: list[SearchHit] = []
        self._current_result_index: int | None = None

        # we pay attention to the video name and frame position
        # because we want "next" and "previous" to be relative
        # to the current position in the selected video
        self._current_video_name: str | None = None
        self._current_frame_position: int = -1

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
        self.video_frame_position_changed(None, -1)
        if self._search_query is not None:
            self.update_search(self._search_query)

    def video_frame_position_changed(self, video_name: str | None, frame_position: int):
        """Update the current video name and frame position.

        This is used to determine the current position in the video which
        influences the "next" and "previous" search results.

        Args:
            video_name (str | None): The name of the current video, or None if not applicable.
            frame_position (int): The current frame position in the video.
        """
        self._current_video_name = video_name
        self._current_frame_position = frame_position

    @property
    def behavior_search_query(self) -> BehaviorSearchQuery | None:
        """Get the current behavior search query."""
        return self._search_query

    @property
    def search_results(self) -> list[SearchHit]:
        """Get the current search results."""
        return self._search_results

    def update_search(self, search_query: BehaviorSearchQuery | None):
        """Set the behavior search query and update the text label."""
        self._search_query = search_query
        self._search_results = []
        self._current_result_index = None

        if search_query is None:
            self.setVisible(False)
        else:
            self.setVisible(True)
            self.text_label.setText(_describe_query(search_query))
            self._search_results = search_behaviors(self._project, search_query)
            print(f"Search results: {len(self._search_results)} hits found.")

        self._update_result_count_label()
        self.search_results_changed.emit(self._search_results)
        self.current_search_hit_changed.emit(self.current_search_hit)

    def _find_search_result_at_current_video_frame(self) -> tuple[SearchHit | None, int] | None:
        """Find the search result that matches the current video and frame position.

        Returns:
            tuple[SearchHit | None, int] | None:
                A tuple of (SearchHit object or None, index) unless the current
                video/frame position is invalid, in which case it returns None.
        """
        if self._current_video_name is None or self._current_frame_position < 0:
            # If we don't have a current video name or frame position return None
            return None
        else:
            file_frame_key_dict = {
                "video_name": self._current_video_name,
                "frame_index": self._current_frame_position,
            }
            return _binary_search_with_comparator(
                self._search_results,
                file_frame_key_dict,
                _compare_file_frame_vs_search_hit,
            )

    def _search_hit_intersects_current_frame(self, hit: SearchHit | None) -> bool:
        """Check if the given search hit intersects with the current video and frame position."""
        if hit is None:
            return False

        return (
            hit.file == self._current_video_name
            and hit.start_frame <= self._current_frame_position <= hit.end_frame
        )

    def _seek_to_search_result(self, hit_index: int | None):
        """Seek to the search result at the given index.

        This safely updates the current result index and emits the
        current_search_hit_changed signal if the hit_index is not None.

        This method also resets the current video name and frame
        position to None and -1, respectively, when seeking to a
        new search result.

        Args:
            hit_index (int | None): The index of the search result to
                seek to, or None to indicate no search results.
        """
        if not self._search_results:
            # force the hit index to None if there are no search results
            hit_index = None

        if hit_index is not None:
            # clamp the hit index to a valid range if it is not None
            hit_index = max(0, min(hit_index, len(self._search_results) - 1))

            # We can discard the current video name and frame position because
            # we are seeking to a new search result.
            self._current_video_name = None
            self._current_frame_position = -1

            self._current_result_index = hit_index
            self._update_result_count_label()
            self.current_search_hit_changed.emit(self.current_search_hit)

    def _on_prev_clicked(self):
        if self._search_hit_intersects_current_frame(self.current_search_hit):
            # If the current search hit intersects with the current frame, we can
            # just decrement the index without checking the current video/frame position.
            self._seek_to_search_result(
                self._current_result_index - 1 if self._current_result_index is not None else 0
            )
        else:
            # we need to find the index of the search results that corresponds
            # to the current video and frame position and then decrement
            match self._find_search_result_at_current_video_frame():
                case None:
                    # there is not current video/frame position, so we just
                    # decrement the index
                    self._seek_to_search_result(
                        self._current_result_index - 1
                        if self._current_result_index is not None
                        else 0
                    )
                case (None, index):
                    # our search gave us an insertion index, so we decrement from that
                    # to find the previous result
                    self._seek_to_search_result(index - 1)
                case (_, index):
                    # we found a search hit that overlaps the current video/frame position.
                    # rewind from that until we get a hit that does not overlap
                    while index > 0:
                        index -= 1
                        curr_hit = self._search_results[index]
                        if not self._search_hit_intersects_current_frame(curr_hit):
                            break

                    self._seek_to_search_result(index)

    def _on_next_clicked(self):
        if self._search_hit_intersects_current_frame(self.current_search_hit):
            # If the current search hit intersects with the current frame, we can
            # just increment the index without checking the current video/frame position.
            self._seek_to_search_result(
                self._current_result_index + 1 if self._current_result_index is not None else 0
            )
        else:
            # we need to find the index of the search results that corresponds
            # to the current video and frame position and then increment
            match self._find_search_result_at_current_video_frame():
                case None:
                    # there is not current video/frame position, so we just
                    # increment the index
                    self._seek_to_search_result(
                        self._current_result_index + 1
                        if self._current_result_index is not None
                        else 0
                    )
                case (None, index):
                    # our search gave us an insertion index, so that should be
                    # the index of the next result
                    self._seek_to_search_result(index)
                case (_, index):
                    # we found a search hit that overlaps the current video/frame position.
                    # fast forward from that until we get a hit that does not overlap
                    while index < len(self._search_results) - 1:
                        index += 1
                        curr_hit = self._search_results[index]
                        if not self._search_hit_intersects_current_frame(curr_hit):
                            break

                    self._seek_to_search_result(index)

    def _on_done_clicked(self):
        self.update_search(None)

    def _update_result_count_label(self):
        if self._search_results:
            self.result_count_label.setText(
                f"({self._current_result_index + 1} of {len(self._search_results)})"
                if self._current_result_index is not None
                else f"({len(self._search_results)} results)"
            )
        else:
            self.result_count_label.setText("(Not found)")

    @property
    def current_search_hit(self) -> SearchHit | None:
        """Get the current search hit based on the current index."""
        curr_index_valid = self._current_result_index is not None
        if curr_index_valid and 0 <= self._current_result_index < len(self._search_results):
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
            min_contiguous_frames=min_frames,
            max_contiguous_frames=max_frames,
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

            if max_frames is not None and min_frames is not None:
                parts.append(f"having {min_frames} to {max_frames} contiguous frames")
            elif min_frames is not None and min_frames > 1:
                parts.append(f"having at least {min_frames} contiguous frames")
            elif max_frames is not None and max_frames > 1:
                parts.append(f"having at most {max_frames} contiguous frames")

            return " ".join(parts)

        case TimelineAnnotationSearchQuery(
            tag=tag,
            min_contiguous_frames=min_frames,
            max_contiguous_frames=max_frames,
        ):
            parts = []
            if tag is None:
                parts.append("All timeline annotations")
            else:
                parts.append(f"Timeline annotations with tag '{tag}'")

            if max_frames is not None and min_frames is not None:
                parts.append(f"having {min_frames} to {max_frames} contiguous frames")
            elif min_frames is not None and min_frames > 1:
                parts.append(f"having at least {min_frames} contiguous frames")
            elif max_frames is not None and max_frames > 1:
                parts.append(f"having at most {max_frames} contiguous frames")

            return " ".join(parts)
        case _:
            return "No Search"


def _binary_search_with_comparator(arr, key, cmp):
    """
    Binary search on a sorted array using a custom comparator.

    Args:
        arr: Sorted list to search.
        key: Target value to find.
        cmp: Comparator function taking (key, element) and returning:
            - Negative if key < element,
            - Zero if key == element,
            - Positive if key > element.

    Returns:
        A tuple (found_item, index) where:
        - found_item is the item found in the array, or None if not found.
        - index is the index of the found item, or the insertion point if not found.
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        curr_item = arr[mid]
        comparison = cmp(key, curr_item)
        if comparison < 0:
            hi = mid - 1
        elif comparison > 0:
            lo = mid + 1
        else:
            return (curr_item, mid)

    return (None, lo)


def _compare_file_frame_vs_search_hit(file_frame_key_dict: dict, search_hit: SearchHit) -> int:
    """
    Compare a file/frame dictionary to a SearchHit object for binary search.

    Compares the video name and frame index of file_frame_key_dict (which must have
    keys "video_name" and "frame_index") against a SearchHit object.

    Returns:
        -1 if the key is before the search hit,
         0 if the key overlaps the search hit,
         1 if the key is after the search hit.
    """
    video_name_key = file_frame_key_dict["video_name"]
    frame_index_key = file_frame_key_dict["frame_index"]

    if video_name_key != search_hit.file:
        return -1 if video_name_key < search_hit.file else 1

    if frame_index_key < search_hit.start_frame:
        return -1
    elif frame_index_key > search_hit.end_frame:
        return 1
    else:
        return 0
