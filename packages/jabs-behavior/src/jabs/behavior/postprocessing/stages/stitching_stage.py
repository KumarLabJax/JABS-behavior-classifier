import numpy as np

from ...events import BehaviorEvents, ClassLabels
from .postprocessing_stage import KwargHelp, PostprocessingStage, StageHelp


class BoutStitchingStage(PostprocessingStage):
    """Postprocessing stage that combines predictions that are separated by short gaps.

    Args:
        max_stitch_gap (int): Maximum gap duration (in frames) allowed between bouts to be stitched together.
    """

    name = "bout_stitching_stage"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "max_stitch_gap" not in kwargs:
            raise ValueError(f"max_stitch_gap must be specified for {self.name}.")

        if not isinstance(kwargs["max_stitch_gap"], int) or kwargs["max_stitch_gap"] <= 0:
            raise ValueError("max_stitch_gap must be a positive integer.")

        self._config["max_stitch_gap"] = kwargs["max_stitch_gap"]

    def apply(self, classes: np.ndarray) -> np.ndarray:
        """Apply stitching to the predictions.

        Args:
            classes (np.ndarray): The predicted classes.

        Returns:
            np.ndarray: Classes after applying the stitching.
        """
        rle_data = BehaviorEvents.from_vector(classes)

        # find short bouts of NOT_BEHAVIOR -- we can stitch across these gaps
        bouts_to_remove = np.logical_and(
            rle_data.durations <= self._config["max_stitch_gap"],
            rle_data.states == ClassLabels.NOT_BEHAVIOR,
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
            description="Combines predictions that are separated by short gaps.",
            kwargs={
                "max_stitch_gap": KwargHelp(
                    description="Maximum gap duration (in frames) allowed between bouts to be stitched together.",
                    type="int",
                ),
            },
        )
