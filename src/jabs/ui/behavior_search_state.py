from dataclasses import dataclass


@dataclass
class LabelBehaviorSearchState:
    """State for label behavior search."""

    positive: bool = False
    negative: bool = False


@dataclass
class PredictionLabelSearchState:
    """State for prediction label search."""

    prob_greater_value: float | None = None
    prob_less_value: float | None = None
    min_contiguous_frames: int | None = None


@dataclass
class BehaviorSearchState:
    """State for behavior search dialog."""

    label_search_state: LabelBehaviorSearchState | None = None
    prediction_search_state: PredictionLabelSearchState | None = None
