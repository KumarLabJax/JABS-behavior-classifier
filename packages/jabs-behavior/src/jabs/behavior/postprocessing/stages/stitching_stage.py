import textwrap

import numpy as np

from ...events import BehaviorEvents, ClassLabels
from .postprocessing_stage import KwargHelp, PostprocessingStage, StageHelp


class BoutStitchingStage(PostprocessingStage):
    """Postprocessing stage that combines predictions that are separated by short gaps.

    Args:
        max_stitch_gap (int): Maximum gap duration (in frames) allowed between bouts to be stitched together.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "max_stitch_gap" not in kwargs:
            raise ValueError("max_stitch_gap must be specified for BoutStitchingStage.")

        if not isinstance(kwargs["max_stitch_gap"], int) or kwargs["max_stitch_gap"] <= 0:
            raise ValueError("max_stitch_gap must be a positive integer.")

        self._config["max_stitch_gap"] = kwargs["max_stitch_gap"]

    def apply(self, classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Apply stitching to the predictions.

        Only a NOT_BEHAVIOR run that actually separates two BEHAVIOR bouts is
        stitched across. A short NOT_BEHAVIOR run at the very start or end of
        the vector, or one bordering frames with no prediction, has a BEHAVIOR
        bout on at most one side, so removing it would relabel not-behavior
        frames instead of joining two bouts.

        Args:
            classes (np.ndarray): The predicted classes.
            probabilities (np.ndarray): The predicted probabilities. (Not used in this stage.)

        Returns:
            np.ndarray: Classes after applying the stitching.
        """
        rle_data = BehaviorEvents.from_vector(classes)
        states = rle_data.states

        # a bout is a stitchable gap only if the bouts on both sides are BEHAVIOR;
        # the first and last bouts have no neighbor on one side, so they never are
        flanked_by_behavior = np.zeros(len(states), dtype=bool)
        if len(states) > 2:
            flanked_by_behavior[1:-1] = np.logical_and(
                states[:-2] == ClassLabels.BEHAVIOR, states[2:] == ClassLabels.BEHAVIOR
            )

        # find short bouts of NOT_BEHAVIOR -- we can stitch across these gaps
        short_not_behavior = np.logical_and(
            rle_data.durations <= self._config["max_stitch_gap"],
            states == ClassLabels.NOT_BEHAVIOR,
        )
        bouts_to_remove = np.logical_and(short_not_behavior, flanked_by_behavior)

        if np.any(bouts_to_remove):
            rle_data.delete_bouts(np.where(bouts_to_remove)[0])

        return rle_data.to_vector()

    @classmethod
    def help(cls) -> StageHelp:
        """Get help information about the stage.

        Returns:
            FilterHelp: Dataclass with a general description and kwarg descriptions.
        """
        return StageHelp(
            description="Combines predictions that are separated by short gaps.",
            description_long=textwrap.dedent("""
            The Stitching Stage connects behavior bouts that are separated by short gaps of not-behavior prediction.
            A short run of not-behavior prediction that does not separate two behavior bouts -- one at the start or
            end of the video, or one bordering frames with no prediction -- is left unchanged.
            """),
            kwargs={
                "max_stitch_gap": KwargHelp(
                    description="Maximum gap duration (in frames) allowed between bouts to be stitched together.",
                    type="int",
                    default=3,
                ),
            },
        )
