import json
from collections.abc import Iterable
from dataclasses import dataclass

from jabs.project import Project


@dataclass(frozen=True)
class BehaviorSearchQuery:
    """Base class for behavior search queries."""

    pass


@dataclass(frozen=True)
class LabelBehaviorSearchQuery(BehaviorSearchQuery):
    """Query for label behavior search."""

    behavior_label: str | None = None
    positive: bool = False
    negative: bool = False


@dataclass(frozen=True)
class PredictionLabelSearchQuery(BehaviorSearchQuery):
    """Query for prediction label search."""

    prob_greater_value: float | None = None
    prob_less_value: float | None = None
    min_contiguous_frames: int | None = None


@dataclass(frozen=True)
class SearchHit:
    """Represents a search hit with file, identity, and frame range information."""

    file: str
    identity: str
    start_frame: int
    end_frame: int


def search_behaviors(
    project: Project | None, search_query: BehaviorSearchQuery
) -> list[SearchHit]:
    """Perform the search based on the current query and project."""
    return _sorted_search_results(_search_behaviors_gen(project, search_query))


def _search_behaviors_gen(
    project: Project | None, search_query: BehaviorSearchQuery
) -> Iterable[SearchHit]:
    """Perform the search based on the current query and project.

    This is a generator that yields search hits.
    """
    if project is None or search_query is None:
        return

    match search_query:
        case LabelBehaviorSearchQuery() as label_query:
            print("Searching for labels...")

            if label_query.positive or label_query.negative:
                video_manager = project.video_manager
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
                                        yield SearchHit(
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


def _sorted_search_results(hits: Iterable["SearchHit"]) -> list["SearchHit"]:
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
