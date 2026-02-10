"""Type definitions for JABS data structures."""

from .inference import AggregationSpec, InferenceSampling
from .keypoints import FrameKeypoints, FrameKeypointsData, KeypointAnnotation
from .model import ModelInfo
from .results import InferenceRunMetadata, KeypointInferenceResult
from .video import VideoInfo

__all__ = [
    "AggregationSpec",
    "FrameKeypoints",
    "FrameKeypointsData",
    "InferenceRunMetadata",
    "InferenceSampling",
    "KeypointAnnotation",
    "KeypointInferenceResult",
    "ModelInfo",
    "VideoInfo",
]
