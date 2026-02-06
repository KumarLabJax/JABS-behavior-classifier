import textwrap

import numpy as np

from ...events import BehaviorEvents, ClassLabels
from .postprocessing_stage import KwargHelp, PostprocessingStage, StageHelp


class GapInterpolationStage(PostprocessingStage):
    """Stage that interpolates gaps in predictions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "max_interpolation_gap" not in kwargs:
            raise ValueError("max_interpolation_gap must be specified for GapInterpolationStage.")

        if (
            not isinstance(kwargs["max_interpolation_gap"], int)
            or kwargs["max_interpolation_gap"] <= 0
        ):
            raise ValueError("max_interpolation_gap must be a positive integer.")

        self._config = {"max_interpolation_gap": kwargs["max_interpolation_gap"]}

    def apply(self, classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Apply gap interpolation to the predictions.

        Args:
            classes (np.ndarray): The predicted classes.
            probabilities(np.ndarray): The predicted probabilities. (Not used in this stage.)

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

    @classmethod
    def help(cls) -> StageHelp:
        """Get help information about the stage.

        Returns:
            FilterHelp: Dataclass with a general description and kwarg descriptions.
        """
        return StageHelp(
            description="Fill short gaps in predicted behavior bouts by interpolating missing frames.",
            description_long=textwrap.dedent("""
              The Interpolation Stage fills short gaps in predictions (such as when there is missing pose) by
              interpolating the class for the missing frames. The missing frames are interpolated using the
              surrounding classes -- if the class on both sides of the gap is the same, the gap is filled
              with that class. If the classes differ, the is gap is split between the two classes so that the
              first half matches the previous class and the second half matches the following class.
            """),
            kwargs={
                "max_interpolation_gap": KwargHelp(
                    description="Maximum gap (in frames) that will be filled using interpolation.",
                    type="int",
                ),
            },
        )
