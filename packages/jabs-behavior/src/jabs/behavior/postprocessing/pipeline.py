import logging
from typing import Any

import jsonschema
import numpy as np

from .stages import PostprocessingStage, stage_registry

logger = logging.getLogger("jabs.behavior.postprocessing.pipeline")

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "parameters": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        "stage_name": {"type": "string"},
    },
    "required": ["stage_name", "parameters"],
    "additionalProperties": False,
}


class PostprocessingPipeline:
    """Pipeline for applying a series of post-processing stages.

    Args:
        config list[dict[str, str | dict | None]]: Configuration listing the
          pipeline stages to execute and their parameters.
    """

    def __init__(self, config: list[dict[str, Any]]) -> None:
        self._stages: list[PostprocessingStage] = []

        # validate the config against the schema
        for i, stage in enumerate(config):
            try:
                jsonschema.validate(stage, CONFIG_SCHEMA)
            except jsonschema.ValidationError as e:
                raise ValueError(f"Invalid config for stage {i}: {e.message}") from e

        registry = stage_registry()
        for stage in config:
            # config can include disabled stages, so check if stage is enabled before
            # adding to pipeline. If "enabled" key is missing, default to True (i.e. stage is enabled).
            if not stage.get("enabled", True):
                logger.debug("Skipping stage %s", stage["stage_name"])
                continue
            else:
                logger.debug("Adding stage %s", stage["stage_name"])

            stage_name = stage["stage_name"]
            if stage_name not in registry:
                raise ValueError(f"Stage '{stage_name}' is not recognized.")

            stage_class = registry[stage_name]
            params = stage.get("parameters", {})

            # instantiate the stage, handle case where params is None
            stage_instance = stage_class(**params) if params else stage_class()
            self._stages.append(stage_instance)

    def run(self, classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Run the post-processing pipeline on the predicted classes.

        Args:
            classes (np.ndarray): The predicted classes.
            probabilities (np.ndarray): The predicted probabilities.

        Returns:
            np.ndarray: Classes after applying all filters in the pipeline.

        """
        for filter_instance in self._stages:
            classes = filter_instance.apply(classes, probabilities)
        return classes
