import numpy as np

from ...events import BehaviorEvents, ClassLabels
from .postprocessing_stage import KwargHelp, PostprocessingStage, StageHelp


class BoutDurationFilterStage(PostprocessingStage):
    """Filter that removes predictions shorter than a specified duration.

    Args:
        min_duration (int): Minimum duration (in frames) for a prediction to be kept.
    """

    name = "duration_filter_stage"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "min_duration" not in kwargs:
            raise ValueError(f"min_duration must be specified for {self.name}.")

        if not isinstance(kwargs["min_duration"], int) or kwargs["min_duration"] <= 0:
            raise ValueError("min_duration must be a positive integer.")

        self._config["min_duration"] = kwargs["min_duration"]

    def apply(self, classes: np.ndarray) -> np.ndarray:
        """Apply the duration filter to the predictions.

        Args:
            classes (np.ndarray): The predicted classes.

        Returns:
            np.ndarray: classes after applying the filter.
        """
        rle_data = BehaviorEvents.from_vector(classes)

        bouts_to_remove = np.logical_and(
            rle_data.durations < self._config["min_duration"],
            rle_data.states == ClassLabels.BEHAVIOR,
        )

        if np.any(bouts_to_remove):
            rle_data.delete_bouts(np.where(bouts_to_remove)[0])

        return rle_data.to_vector()

    def help(self) -> StageHelp:
        """Get help information about the stage.

        Returns:
            FilterHelp: Dataclass with a general description and kwarg descriptions.
        """
        return StageHelp(
            description="Removes predictions shorter than a specified duration.",
            kwargs={
                "min_duration": KwargHelp(
                    description="Minimum duration (in frames) for a prediction to be kept.",
                    type="int",
                ),
            },
        )
