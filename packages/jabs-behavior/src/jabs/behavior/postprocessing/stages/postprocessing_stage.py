import abc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KwargHelp:
    """Dataclass for kwarg help information."""

    description: str
    type: str


@dataclass(frozen=True)
class StageHelp:
    """Dataclass for stage help information."""

    description: str
    kwargs: dict[str, KwargHelp]


class PostprocessingStage(abc.ABC):
    """Base class for post-processing stages."""

    name = "postprocessing_stage"

    legacy_names: tuple[str, ...] = ()

    def __init__(self, **kwargs) -> None:
        self._config = {}

    @property
    def config(self) -> dict:
        """Get the filter configuration."""
        return self._config

    @abc.abstractmethod
    def apply(self, classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Apply the stage to the predictions.

        Args:
            classes (np.ndarray): The predicted classes.
            probabilities (np.ndarray): The predicted probabilities.

        Returns:
            np.ndarray: Classes after applying the stage transformation.
        """
        raise NotImplementedError("Subclasses must implement the apply method.")

    @abc.abstractmethod
    def help(self) -> StageHelp:
        """Get help information about the stage.

        Returns:
            StageHelp: Dataclass with a general description and kwarg descriptions.

        Example:
            return StageHelp(
                description="Removes short bouts below a minimum duration.",
                kwargs={
                    "min_duration": KwargHelp(
                        description="Minimum duration (in frames) for a bout to be kept.",
                        type="int"
                    ),
                }
            )
        """
        raise NotImplementedError("Subclasses must implement the help method.")
