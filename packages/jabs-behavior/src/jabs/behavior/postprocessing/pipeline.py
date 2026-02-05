import numpy as np

from .stages import PostprocessingStage, stage_registry


class PostprocessingPipeline:
    """Pipeline for applying a series of post-processing filters.

    Args:
        config (dict[str, dict]): Configuration dictionary specifying the filters to apply
                       and their parameters.
    """

    def __init__(self, config: dict[str, dict | None]) -> None:
        self._filters: list[PostprocessingStage] = []

        registry = stage_registry()
        for key, value in config.items():
            if key not in registry:
                raise ValueError(f"Filter '{key}' is not recognized.")

            filter_class = registry[key]
            filter_kwargs = value or {}
            filter_instance = filter_class(**filter_kwargs)
            self._filters.append(filter_instance)

    def run(self, classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Run the post-processing pipeline on the predicted classes.

        Args:
            classes (np.ndarray): The predicted classes.
            probabilities (np.ndarray): The predicted probabilities.

        Returns:
            np.ndarray: Classes after applying all filters in the pipeline.

        """
        for filter_instance in self._filters:
            classes = filter_instance.apply(classes, probabilities)
        return classes
