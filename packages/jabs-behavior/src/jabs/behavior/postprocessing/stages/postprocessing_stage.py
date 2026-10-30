import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KwargHelp:
    """Dataclass for kwarg help information.

    Attributes:
        description: Human-readable description of the parameter.
        type: Type name string (e.g. ``"int"``).
        default: Suggested default value, or ``None`` if no default is defined.
            Used by config generators to populate template files.
    """

    description: str
    type: str
    default: Any = field(default=None, compare=False)


@dataclass(frozen=True)
class StageHelp:
    """Dataclass for stage help information."""

    description: str
    description_long: str | None = None
    kwargs: dict[str, KwargHelp] | None = None


class PostprocessingStage(abc.ABC):
    """Base class for post-processing stages."""

    legacy_names: tuple[str, ...] = ()

    def __init__(self, **kwargs) -> None:
        self._config = {}

    @property
    def config(self) -> dict:
        """Get the filter configuration."""
        return self._config

    def _require_positive_int(self, kwargs: dict[str, Any], name: str) -> int:
        """Return a required positive-integer keyword argument.

        Every stage's constructor takes its tuning parameter as a keyword
        argument that must be present and must be a positive integer, so the
        check lives here rather than being repeated in each stage. Keeping one
        definition is what stops the stages from drifting into reporting the
        same mistake differently.

        Args:
            kwargs: The keyword arguments the stage was constructed with.
            name: Name of the keyword argument to read.

        Returns:
            The validated value.

        Raises:
            ValueError: If ``name`` is absent from ``kwargs``, or its value is
                not an integer greater than zero.
        """
        if name not in kwargs:
            raise ValueError(f"{name} must be specified for {type(self).__name__}.")

        value = kwargs[name]
        # bool is an int subclass, but True is not a frame count
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

        return value

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

    @classmethod
    @abc.abstractmethod
    def help(cls) -> StageHelp:
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
