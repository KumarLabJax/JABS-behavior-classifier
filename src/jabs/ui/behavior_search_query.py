from dataclasses import dataclass


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
