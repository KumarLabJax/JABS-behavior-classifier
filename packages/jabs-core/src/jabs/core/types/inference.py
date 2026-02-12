from dataclasses import dataclass

from jabs.core.enums import ConfidenceMetric, Method, SamplingStrategy


@dataclass(frozen=True)
class InferenceSampling:
    """Sampling details for selecting frames.

    Attributes:
        num_frames: Number of frames requested for inference.
        frame_indices: Frame indices actually sampled.
        strategy: Sampling strategy name (e.g., "uniform").
    """

    num_frames: int
    frame_indices: list[int]
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM


@dataclass(frozen=True)
class AggregationSpec:
    """Definition of how per-frame outputs are aggregated.

    Attributes:
        confidence_threshold: Minimum confidence required to keep an item.
        confidence_metric: Description of how confidence was computed.
        method: Aggregation method (e.g., "mean").
    """

    confidence_threshold: float
    confidence_metric: ConfidenceMetric = ConfidenceMetric.MEAN_KEYPOINT_MAX_SIGMOID
    method: Method = Method.MEAN
