from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from jabs.project import Project, TrackLabels


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


class PredictionSearchKind(Enum):
    """Enumeration for different kinds of prediction searches."""

    POSITIVE_PREDICTION = auto()
    NEGATIVE_PREDICTION = auto()
    PROBABILITY_RANGE = auto()


@dataclass(frozen=True)
class PredictionBehaviorSearchQuery(BehaviorSearchQuery):
    """Query for prediction label search."""

    search_kind: PredictionSearchKind
    behavior_label: str | None = None
    prob_greater_value: float | None = None
    prob_less_value: float | None = None
    min_contiguous_frames: int | None = None
    max_contiguous_frames: int | None = None


@dataclass(frozen=True)
class TimelineAnnotationSearchQuery(BehaviorSearchQuery):
    """Query for timeline annotation search."""

    tag: str | None = None
    min_contiguous_frames: int | None = None
    max_contiguous_frames: int | None = None


@dataclass(frozen=True)
class SearchHit:
    """Represents a search hit with file, identity, and frame range information."""

    file: str
    identity: str
    behavior: str | None
    start_frame: int
    end_frame: int


def search_behaviors(
    project: Project | None, search_query: BehaviorSearchQuery
) -> list[SearchHit]:
    """Perform the search based on the current query and project."""
    return _sorted_search_results(list(_search_behaviors_gen(project, search_query)))


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
            if label_query.positive or label_query.negative:
                video_manager = project.video_manager
                sorted_videos = sorted(video_manager.videos)

                for video in sorted_videos:
                    anno_dict = video_manager.load_annotations(video)
                    if anno_dict is not None:
                        labels = (
                            anno_dict["unfragmented_labels"]
                            if "unfragmented_labels" in anno_dict
                            else anno_dict.get("labels", {})
                        )
                        for identity, identified_labels in labels.items():
                            for behavior, blocks in identified_labels.items():
                                for block in blocks:
                                    block_matches_query = (
                                        behavior == label_query.behavior_label
                                        or label_query.behavior_label is None
                                    ) and (
                                        (label_query.positive and block["present"])
                                        or (label_query.negative and not block["present"])
                                    )
                                    if block_matches_query:
                                        yield SearchHit(
                                            file=video,
                                            identity=str(identity),
                                            behavior=behavior,
                                            start_frame=block["start"],
                                            end_frame=block["end"],
                                        )

        case PredictionBehaviorSearchQuery() as pred_query:
            proj_settings = project.settings_manager.project_settings
            if pred_query.behavior_label is None:
                behavior_dict = proj_settings.get("behavior", {})
                behaviors = list(behavior_dict.keys())
            else:
                behaviors = [pred_query.behavior_label]

            video_manager = project.video_manager
            sorted_videos = sorted(video_manager.videos)
            for video in sorted_videos:
                for behavior in behaviors:
                    # Load predictions for the video and behavior
                    preds, probs, _ = project.prediction_manager.load_predictions(video, behavior)

                    for aid, animal_probs in probs.items():
                        # do a couple of checks before generating search hits
                        animal_probs = probs.get(aid)
                        if animal_probs is None or animal_probs.size == 0:
                            continue

                        animal_preds = preds.get(aid)
                        if animal_preds is None or animal_preds.size != animal_probs.size:
                            print(
                                f"WARNING: Skipping {video} for {aid} as predictions and probabilities "
                                f"do not match in size."
                            )
                            continue

                        # get contiguous intervals of frames that meet the probability criteria
                        for start, end in _gen_contig_true_intervals(
                            pred_query, animal_preds, animal_probs
                        ):
                            # check if the search hit meets the min/max contiguous frames criteria
                            interval_length = end - start + 1
                            if (
                                pred_query.min_contiguous_frames is not None
                                and interval_length < pred_query.min_contiguous_frames
                            ):
                                continue

                            if (
                                pred_query.max_contiguous_frames is not None
                                and interval_length > pred_query.max_contiguous_frames
                            ):
                                continue

                            yield SearchHit(
                                file=video,
                                identity=str(aid),
                                behavior=behavior,
                                start_frame=start,
                                end_frame=end,
                            )

        case TimelineAnnotationSearchQuery() as timeline_query:
            video_manager = project.video_manager
            for video in video_manager.videos:
                anno_dict = video_manager.load_annotations(video)

                # file does not have a corresponding annotation file, skip it
                if anno_dict is None:
                    continue

                # file has jabs/annotations/<video>.json, grab timeline annotations
                for annotation in anno_dict.get("annotations", []):
                    if timeline_query.tag is None or annotation.get("tag") == timeline_query.tag:
                        start = annotation["start"]
                        end = annotation["end"]

                        # check if the search hit meets the min/max contiguous frames criteria
                        interval_length = end - start + 1
                        if (
                            timeline_query.min_contiguous_frames is not None
                            and interval_length < timeline_query.min_contiguous_frames
                        ):
                            continue

                        if (
                            timeline_query.max_contiguous_frames is not None
                            and interval_length > timeline_query.max_contiguous_frames
                        ):
                            continue

                        identity = annotation.get("identity")
                        if identity is not None:
                            identity = str(identity)

                        yield SearchHit(
                            file=video,
                            identity=identity,
                            behavior=None,
                            start_frame=start,
                            end_frame=end,
                        )

        case _:
            raise ValueError("Unknown query type or unsupported search.")


def _sorted_search_results(hits: Iterable[SearchHit]) -> list[SearchHit]:
    """
    Return a list of sorted search hits.

    Return a list of search hits sorted by file, identity (numeric if possible), start_frame and behavior.

    Args:
        hits: An iterable of _SearchHit objects.

    Returns:
        A list of _SearchHit objects sorted by file, identity, and start_frame.
        Identity is sorted numerically if possible, otherwise alphanumerically.
    """
    try:
        return sorted(
            hits,
            key=lambda hit: (hit.file, hit.start_frame, int(hit.identity), hit.behavior),
        )
    except (ValueError, TypeError):
        # there is at least one identity that cannot be converted to an int
        return sorted(
            hits,
            key=lambda hit: (
                hit.file,
                hit.start_frame,
                "" if hit.identity is None else hit.identity,
                hit.behavior,
            ),
        )


def _gen_contig_true_intervals(
    pred_query: PredictionBehaviorSearchQuery,
    animal_predictions: np.ndarray,
    animal_probabilities: np.ndarray,
) -> Iterable[tuple[int, int]]:
    if animal_predictions.size == 0 or animal_probabilities.size == 0:
        return

    if animal_predictions.size != animal_probabilities.size:
        print("WARNING: Predictions and probabilities do not match in size. Skipping this animal.")
        return

    match pred_query.search_kind:
        case PredictionSearchKind.POSITIVE_PREDICTION:
            crit_mask = animal_predictions == TrackLabels.Label.BEHAVIOR.value
        case PredictionSearchKind.NEGATIVE_PREDICTION:
            crit_mask = animal_predictions == TrackLabels.Label.NOT_BEHAVIOR.value
        case PredictionSearchKind.PROBABILITY_RANGE:
            # convert probabilities to a 0.0 to 1.0 scale where the most confident
            # NOT_BEHAVIOR is 0.0 and the most confident BEHAVIOR is 1.0.
            #
            # Also note that we use eps to nudge probabilities away from 0.5 because
            # you can end up with counterintuitive results if you don't.
            not_behavior_mask = animal_predictions == TrackLabels.Label.NOT_BEHAVIOR.value
            animal_probabilities = animal_probabilities.copy() + np.finfo(float).eps
            animal_probabilities[not_behavior_mask] = 1.0 - animal_probabilities[not_behavior_mask]
            animal_probabilities = np.clip(animal_probabilities, 0.0, 1.0)

            crit_mask = np.isin(
                animal_predictions,
                [TrackLabels.Label.BEHAVIOR.value, TrackLabels.Label.NOT_BEHAVIOR.value],
            )
            if pred_query.prob_greater_value is not None:
                crit_mask &= animal_probabilities >= pred_query.prob_greater_value
            if pred_query.prob_less_value is not None:
                crit_mask &= animal_probabilities <= pred_query.prob_less_value
        case _:
            raise ValueError(f"Unsupported search kind: {pred_query.search_kind}")

    # exit early if no criteria are met
    if not np.any(crit_mask):
        return

    # find start and end of all contiguous true for crit_mask
    crit_mask_diff = np.diff(crit_mask.astype(int))
    starts = np.where(crit_mask_diff == 1)[0] + 1
    ends = np.where(crit_mask_diff == -1)[0]

    if crit_mask[0]:
        starts = np.insert(starts, 0, 0)
    if crit_mask[-1]:
        ends = np.append(ends, crit_mask.size - 1)

    # yield start and end indices of contiguous blocks
    yield from zip(starts, ends, strict=True)
