from dataclasses import dataclass


@dataclass
class LabelBehaviorSearchQuery:
    """Query for label behavior search."""

    positive: bool = False
    negative: bool = False


@dataclass
class PredictionLabelSearchQuery:
    """Query for prediction label search."""

    prob_greater_value: float | None = None
    prob_less_value: float | None = None
    min_contiguous_frames: int | None = None


@dataclass
class BehaviorSearchQuery:
    """Query for behavior search dialog."""

    label_search_query: LabelBehaviorSearchQuery | None = None
    prediction_search_query: PredictionLabelSearchQuery | None = None

    def describe(self) -> str:
        """Return a descriptive string for the current search query."""
        no_search = "No Search"
        if self.label_search_query:
            parts = []
            if self.label_search_query.positive:
                parts.append("positive labels")
            if self.label_search_query.negative:
                parts.append("negative labels")
            if not parts:
                return no_search
            return " & ".join(parts)
        elif self.prediction_search_query:
            parts = []

            gt = self.prediction_search_query.prob_greater_value
            lt = self.prediction_search_query.prob_less_value

            if gt is not None and lt is not None:
                parts.append(f"{gt} < behavior prob. < {lt}")
            else:
                if gt is not None:
                    parts.append(f"behavior prob. > {gt}")
                elif lt is not None:
                    parts.append(f"behavior prob. < {lt}")

            if not parts:
                return no_search

            if self.prediction_search_query.min_contiguous_frames is not None:
                parts.append(
                    f"with at least {self.prediction_search_query.min_contiguous_frames} contiguous frames"
                )

            return " ".join(parts)
        else:
            return no_search
