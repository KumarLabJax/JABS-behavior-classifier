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
from typing import ClassVar

import numpy as np

try:
    import pyarrow as pa
except ImportError:
    pa = None


from jabs.io.backends.parquet.backend import PARQUET_ADAPTERS
from jabs.io.types.inference import AggregationSpec, InferenceSampling
from jabs.io.types.keypoints import FrameKeypoints, FrameKeypointsData, KeypointAnnotation
from jabs.io.types.model import ModelInfo
from jabs.io.types.results import InferenceRunMetadata
from jabs.io.types.video import VideoInfo


@PARQUET_ADAPTERS.register()
class FrameKeypointsDataAdapter:
    """Parquet adapter for FrameKeypointsData.

    Storage: One row per frame.
    - frame_index: int64
    - keypoints: list<list<float64>> representing (K, 2) array
    - confidence: float64 (nullable)
    """

    data_type: ClassVar[type] = FrameKeypointsData

    def to_table(self, data: FrameKeypointsData) -> pa.Table:
        """Convert FrameKeypointsData to a PyArrow Table."""
        frame_indices = []
        keypoints_list = []
        confidences = []

        for frame in data.frames:
            frame_indices.append(frame.frame_index)
            # Convert (K, 2) array to list of [x, y] pairs
            keypoints_list.append(frame.keypoints.tolist())
            confidences.append(frame.confidence)

        return pa.table(
            {
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "keypoints": pa.array(
                    keypoints_list,
                    type=pa.list_(pa.list_(pa.float64(), 2)),
                ),
                "confidence": pa.array(confidences, type=pa.float64()),
            },
            metadata={b"data_type": b"FrameKeypointsData"},
        )

    def from_table(self, table: pa.Table) -> FrameKeypointsData:
        """Convert a PyArrow Table to FrameKeypointsData."""
        frames = []

        for i in range(table.num_rows):
            frame_index = table["frame_index"][i].as_py()
            keypoints = np.array(table["keypoints"][i].as_py(), dtype=np.float64)
            confidence = table["confidence"][i].as_py()

            frames.append(
                FrameKeypoints(
                    frame_index=frame_index,
                    keypoints=keypoints,
                    confidence=confidence,
                )
            )

        return FrameKeypointsData(frames=frames)

    def schema(self) -> pa.Schema:
        """Get the PyArrow schema for this adapter."""
        return pa.schema(
            [
                pa.field("frame_index", pa.int64()),
                pa.field("keypoints", pa.list_(pa.list_(pa.float64(), 2))),
                pa.field("confidence", pa.float64()),
            ]
        )


@PARQUET_ADAPTERS.register()
class KeypointAnnotationAdapter:
    """Parquet adapter for KeypointAnnotation.

    Storage: Single row.
    - keypoints: list<list<float64>> representing (K, 2) array
    - kept_frame_indices: list<int64>
    - mean_confidence: float64 (nullable)
    """

    data_type: ClassVar[type] = KeypointAnnotation

    def to_table(self, data: KeypointAnnotation) -> pa.Table:
        """Convert KeypointAnnotation to a PyArrow Table."""
        return pa.table(
            {
                "keypoints": pa.array(
                    [data.keypoints.tolist()],
                    type=pa.list_(pa.list_(pa.float64(), 2)),
                ),
                "kept_frame_indices": pa.array(
                    [data.kept_frame_indices],
                    type=pa.list_(pa.int64()),
                ),
                "mean_confidence": pa.array(
                    [data.mean_confidence],
                    type=pa.float64(),
                ),
            },
            metadata={b"data_type": b"KeypointAnnotation"},
        )

    def from_table(self, table: pa.Table) -> KeypointAnnotation:
        """Convert a PyArrow Table to KeypointAnnotation."""
        keypoints = np.array(table["keypoints"][0].as_py(), dtype=np.float64)
        kept_frame_indices = table["kept_frame_indices"][0].as_py()
        mean_confidence = table["mean_confidence"][0].as_py()

        return KeypointAnnotation(
            keypoints=keypoints,
            kept_frame_indices=kept_frame_indices,
            mean_confidence=mean_confidence,
        )

    def schema(self) -> pa.Schema:
        """Get the PyArrow schema for this adapter."""
        return pa.schema(
            [
                pa.field("keypoints", pa.list_(pa.list_(pa.float64(), 2))),
                pa.field("kept_frame_indices", pa.list_(pa.int64())),
                pa.field("mean_confidence", pa.float64()),
            ]
        )


@PARQUET_ADAPTERS.register()
class InferenceRunMetadataAdapter:
    """Parquet adapter for InferenceRunMetadata.

    Storage: Single row with nested structs for composed types.
    - video: struct<path, width, height, fps, frame_count>
    - model: struct<checkpoint_path, backbone, num_keypoints, ...>
    - sampling: struct<num_frames, frame_indices, strategy>
    - aggregation: struct<confidence_threshold, confidence_metric, method>
    - created_at: string (nullable)
    - extra: string (JSON-encoded dict)
    """

    data_type: ClassVar[type] = InferenceRunMetadata

    # Define struct schemas for nested types
    VIDEO_INFO_TYPE = pa.struct(
        [
            pa.field("path", pa.string()),
            pa.field("width", pa.int64()),
            pa.field("height", pa.int64()),
            pa.field("fps", pa.float64()),
            pa.field("frame_count", pa.int64()),
        ]
    )

    MODEL_INFO_TYPE = pa.struct(
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

    SAMPLING_TYPE = pa.struct(
        [
            pa.field("num_frames", pa.int64()),
            pa.field("frame_indices", pa.list_(pa.int64())),
            pa.field("strategy", pa.string()),
        ]
    )

    AGGREGATION_TYPE = pa.struct(
        [
            pa.field("confidence_threshold", pa.float64()),
            pa.field("confidence_metric", pa.string()),
            pa.field("method", pa.string()),
        ]
    )

    def to_table(self, data: InferenceRunMetadata) -> pa.Table:
        """Convert InferenceRunMetadata to a PyArrow Table."""
        video_dict = {
            "path": data.video.path,
            "width": data.video.width,
            "height": data.video.height,
            "fps": data.video.fps,
            "frame_count": data.video.frame_count,
        }

        model_dict = {
            "checkpoint_path": data.model.checkpoint_path,
            "backbone": data.model.backbone,
            "num_keypoints": data.model.num_keypoints,
            "input_size": list(data.model.input_size),
            "output_stride": data.model.output_stride,
            "decode_use_dark": data.model.decode_use_dark,
            "decode_sigma": data.model.decode_sigma,
        }

        sampling_dict = {
            "num_frames": data.sampling.num_frames,
            "frame_indices": data.sampling.frame_indices,
            "strategy": data.sampling.strategy,
        }

        aggregation_dict = {
            "confidence_threshold": data.aggregation.confidence_threshold,
            "confidence_metric": data.aggregation.confidence_metric,
            "method": data.aggregation.method,
        }

        return pa.table(
            {
                "video": pa.array([video_dict], type=self.VIDEO_INFO_TYPE),
                "model": pa.array([model_dict], type=self.MODEL_INFO_TYPE),
                "sampling": pa.array([sampling_dict], type=self.SAMPLING_TYPE),
                "aggregation": pa.array([aggregation_dict], type=self.AGGREGATION_TYPE),
                "created_at": pa.array([data.created_at], type=pa.string()),
                "extra": pa.array([json.dumps(data.extra)], type=pa.string()),
            },
            metadata={b"data_type": b"InferenceRunMetadata"},
        )

    def from_table(self, table: pa.Table) -> InferenceRunMetadata:
        """Convert a PyArrow Table to InferenceRunMetadata."""
        video_dict = table["video"][0].as_py()
        model_dict = table["model"][0].as_py()
        sampling_dict = table["sampling"][0].as_py()
        aggregation_dict = table["aggregation"][0].as_py()

        video = VideoInfo(
            path=video_dict["path"],
            width=video_dict["width"],
            height=video_dict["height"],
            fps=video_dict["fps"],
            frame_count=video_dict["frame_count"],
        )

        model = ModelInfo(
            checkpoint_path=model_dict["checkpoint_path"],
            backbone=model_dict["backbone"],
            num_keypoints=model_dict["num_keypoints"],
            input_size=tuple(model_dict["input_size"]),
            output_stride=model_dict["output_stride"],
            decode_use_dark=model_dict["decode_use_dark"],
            decode_sigma=model_dict["decode_sigma"],
        )

        sampling = InferenceSampling(
            num_frames=sampling_dict["num_frames"],
            frame_indices=sampling_dict["frame_indices"],
            strategy=sampling_dict["strategy"],
        )

        aggregation = AggregationSpec(
            confidence_threshold=aggregation_dict["confidence_threshold"],
            confidence_metric=aggregation_dict["confidence_metric"],
            method=aggregation_dict["method"],
        )

        extra_json = table["extra"][0].as_py()
        extra = json.loads(extra_json) if extra_json else {}

        return InferenceRunMetadata(
            video=video,
            model=model,
            sampling=sampling,
            aggregation=aggregation,
            created_at=table["created_at"][0].as_py(),
            extra=extra,
        )

    def schema(self) -> pa.Schema:
        """Get the PyArrow schema for this adapter."""
        return pa.schema(
            [
                pa.field("video", self.VIDEO_INFO_TYPE),
                pa.field("model", self.MODEL_INFO_TYPE),
                pa.field("sampling", self.SAMPLING_TYPE),
                pa.field("aggregation", self.AGGREGATION_TYPE),
                pa.field("created_at", pa.string()),
                pa.field("extra", pa.string()),
            ]
        )
