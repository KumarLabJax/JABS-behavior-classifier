"""Type definitions for JABS data structures."""

from .feature_cache import FeatureCacheMetadata, PerFrameCacheData
from .inference import AggregationSpec, InferenceSampling
from .keypoints import FrameKeypoints, FrameKeypointsData, KeypointAnnotation
from .model import ModelInfo
from .pose import DynamicObjectData, PoseData
from .prediction import BehaviorPrediction, ClassifierMetadata
from .results import InferenceRunMetadata, KeypointInferenceResult
from .video import VideoInfo

__all__ = [
    "AggregationSpec",
    "BehaviorPrediction",
    "ClassifierMetadata",
    "DynamicObjectData",
    "FeatureCacheMetadata",
    "FrameKeypoints",
    "FrameKeypointsData",
    "InferenceRunMetadata",
    "InferenceSampling",
    "KeypointAnnotation",
    "KeypointInferenceResult",
    "ModelInfo",
    "PerFrameCacheData",
    "PoseData",
    "VideoInfo",
]
