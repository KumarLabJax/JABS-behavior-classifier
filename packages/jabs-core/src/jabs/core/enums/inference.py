"""Enumerations for JABS inference."""

from enum import Enum


class SamplingStrategy(Enum):
    """The strategy used for sampling frames for inference."""

    UNIFORM = "uniform"


class ConfidenceMetric(Enum):
    """How inference confidence was computed."""

    MEAN_KEYPOINT_MAX_SIGMOID = "mean_keypoint_max_sigmoid"


class Method(Enum):
    """The Aggregation method for how per-frame outputs are aggregated."""

    MEAN = "mean"
