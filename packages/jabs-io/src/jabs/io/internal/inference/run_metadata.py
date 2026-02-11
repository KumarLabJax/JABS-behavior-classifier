"""Concrete Parquet adapters for domain types.

Design principle: Use native Parquet types wherever possible.
- Scalars -> native columns
- numpy arrays -> fixed-size lists of floats
- Lists of ints -> native list<int64>
- Nested dataclasses -> nested structs
- Dicts -> JSON string (Parquet maps are less portable)
"""

from __future__ import annotations

import json

try:
    import pyarrow as pa
except ImportError:
    pa = None

from jabs.core.enums import StorageFormat
from jabs.core.types import InferenceRunMetadata, ModelInfo
from jabs.core.types.inference import AggregationSpec, InferenceSampling
from jabs.core.types.video import VideoInfo
from jabs.io.base import ParquetAdapter
from jabs.io.registry import register_adapter


@register_adapter(StorageFormat.PARQUET, InferenceRunMetadata, priority=10)
class InferenceRunMetadataAdapter(ParquetAdapter):
    """Parquet adapter for InferenceRunMetadata.

    Storage: Single row with nested structs for composed types.
    - video: struct<path, width, height, fps, frame_count>
    - model: struct<checkpoint_path, backbone, num_keypoints, ...>
    - sampling: struct<num_frames, frame_indices, strategy>
    - aggregation: struct<confidence_threshold, confidence_metric, method>
    - created_at: string (nullable)
    - extra: string (JSON-encoded dict)
    """

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return data_type is InferenceRunMetadata

    @staticmethod
    def _video_info_type():
        return pa.struct(
            [
                pa.field("path", pa.string()),
                pa.field("width", pa.int64()),
                pa.field("height", pa.int64()),
                pa.field("fps", pa.float64()),
                pa.field("frame_count", pa.int64()),
            ]
        )

    @staticmethod
    def _model_info_type():
        return pa.struct(
            [
                pa.field("checkpoint_path", pa.string()),
                pa.field("backbone", pa.string()),
                pa.field("num_keypoints", pa.int64()),
                pa.field("input_size", pa.list_(pa.int64(), 2)),
                pa.field("output_stride", pa.int64()),
                pa.field("decode_use_dark", pa.bool_()),
                pa.field("decode_sigma", pa.float64()),
            ]
        )

    @staticmethod
    def _sampling_type():
        return pa.struct(
            [
                pa.field("num_frames", pa.int64()),
                pa.field("frame_indices", pa.list_(pa.int64())),
                pa.field("strategy", pa.string()),
            ]
        )

    @staticmethod
    def _aggregation_type():
        return pa.struct(
            [
                pa.field("confidence_threshold", pa.float64()),
                pa.field("confidence_metric", pa.string()),
                pa.field("method", pa.string()),
            ]
        )

    def schema(self) -> pa.Schema:  # noqa: D102
        return pa.schema(
            [
                pa.field("video", self._video_info_type()),
                pa.field("model", self._model_info_type()),
                pa.field("sampling", self._sampling_type()),
                pa.field("aggregation", self._aggregation_type()),
                pa.field("created_at", pa.string()),
                pa.field("extra", pa.string()),
            ]
        )

    def _to_record(self, data: InferenceRunMetadata) -> dict:
        return {
            "video": {
                "path": data.video.path,
                "width": data.video.width,
                "height": data.video.height,
                "fps": data.video.fps,
                "frame_count": data.video.frame_count,
            },
            "model": {
                "checkpoint_path": data.model.checkpoint_path,
                "backbone": data.model.backbone,
                "num_keypoints": data.model.num_keypoints,
                "input_size": list(data.model.input_size),
                "output_stride": data.model.output_stride,
                "decode_use_dark": data.model.decode_use_dark,
                "decode_sigma": data.model.decode_sigma,
            },
            "sampling": {
                "num_frames": data.sampling.num_frames,
                "frame_indices": data.sampling.frame_indices,
                "strategy": data.sampling.strategy,
            },
            "aggregation": {
                "confidence_threshold": data.aggregation.confidence_threshold,
                "confidence_metric": data.aggregation.confidence_metric,
                "method": data.aggregation.method,
            },
            "created_at": data.created_at,
            "extra": json.dumps(data.extra),
        }

    def _from_record(self, record: dict, data_type: type | None = None):
        extra_json = record["extra"]

        return InferenceRunMetadata(
            video=VideoInfo(**record["video"]),
            model=ModelInfo(
                **{
                    **record["model"],
                    "input_size": tuple(record["model"]["input_size"]),
                }
            ),
            sampling=InferenceSampling(**record["sampling"]),
            aggregation=AggregationSpec(**record["aggregation"]),
            created_at=record["created_at"],
            extra=json.loads(extra_json) if extra_json else {},
        )
