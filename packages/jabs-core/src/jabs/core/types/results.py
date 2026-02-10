from dataclasses import dataclass, field
from typing import Any

from jabs.core.types.inference import AggregationSpec, InferenceSampling
from jabs.core.types.keypoints import FrameKeypoints, KeypointAnnotation
from jabs.core.types.model import ModelInfo
from jabs.core.types.video import VideoInfo


@dataclass(frozen=True)
class InferenceRunMetadata:
    """Metadata describing how an inference run was produced.

    Attributes:
        video: Video metadata.
        model: Model metadata.
        sampling: Frame sampling metadata.
        aggregation: Aggregation metadata.
        created_at: Optional timestamp (ISO 8601).
        extra: Optional free-form metadata (e.g., device, library versions).
    """

    video: VideoInfo
    model: ModelInfo
    sampling: InferenceSampling
    aggregation: AggregationSpec
    created_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KeypointInferenceResult:
    """Complete output for keypoint inference with aggregation.

    Attributes:
        metadata: Metadata needed to reproduce aggregation.
        frames: Per-frame keypoint inference results.
        annotation: Aggregated annotation for the video.
    """

    metadata: InferenceRunMetadata
    frames: list[FrameKeypoints]
    annotation: KeypointAnnotation
