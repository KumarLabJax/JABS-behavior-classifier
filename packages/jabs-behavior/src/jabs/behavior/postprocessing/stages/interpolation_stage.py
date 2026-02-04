import numpy as np

from ...events import BehaviorEvents, ClassLabels
from .postprocessing_stage import PostprocessingStage, StageHelp


class GapInterpolationStage(PostprocessingStage):
    """Stage that interpolates gaps in predictions."""

    name = "interpolation_stage"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "max_interpolation_gap" not in kwargs:
            raise ValueError(f"max_interpolation_gap must be specified for {self.name}.")

        if (
            not isinstance(kwargs["max_interpolation_gap"], int)
            or kwargs["max_interpolation_gap"] <= 0
        ):
            raise ValueError("max_interpolation_gap must be a positive integer.")

        self._config = {"max_interpolation_gap": kwargs["max_interpolation_gap"]}

    def apply(self, classes: np.ndarray) -> np.ndarray:
        """Apply gap interpolation to the predictions.

        Args:
            classes (np.ndarray): The predicted classes.

        Returns:
            np.ndarray: Classes after interpolation.
        """
        rle_data = BehaviorEvents.from_vector(classes)
        no_prediction_gaps_to_fill = np.logical_and(
            rle_data.durations <= self._config["max_interpolation_gap"],
            rle_data.states == ClassLabels.NONE,
        )

        if np.any(no_prediction_gaps_to_fill):
            rle_data.delete_bouts(np.where(no_prediction_gaps_to_fill)[0])

        return rle_data.to_vector()

    def help(self) -> StageHelp:
        """Get help information about the stage.

        Returns:
            FilterHelp: Dataclass with a general description and kwarg descriptions.
        """
        return StageHelp(description="Interpolates gaps in predictions.", kwargs={})
