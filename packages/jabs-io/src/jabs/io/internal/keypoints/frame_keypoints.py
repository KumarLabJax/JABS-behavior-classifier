"""Concrete Parquet adapters for domain types.

Design principle: Use native Parquet types wherever possible.
- Scalars -> native columns
- numpy arrays -> fixed-size lists of floats
- Lists of ints -> native list<int64>
- Nested dataclasses -> nested structs
- Dicts -> JSON string (Parquet maps are less portable)
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from jabs.core.enums import StorageFormat
from jabs.core.types.keypoints import (
    FrameKeypoints,
    FrameKeypointsData,
)
from jabs.io.base import ParquetAdapter
from jabs.io.registry import register_adapter


@register_adapter(StorageFormat.PARQUET, FrameKeypointsData, priority=10)
class FrameKeypointsDataAdapter(ParquetAdapter):
    """Parquet adapter for FrameKeypointsData.

    Storage: One row per frame.
    - frame_index: int64
    - keypoints: list<list<float64, 2>> representing (K, 2) array
    - confidence: float64 (nullable)
    """

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return data_type is FrameKeypointsData

    def schema(self) -> pa.Schema:  # noqa: D102
        return pa.schema(
            [
                pa.field("frame_index", pa.int64()),
                pa.field("keypoints", pa.list_(pa.list_(pa.float64(), 2))),
                pa.field("confidence", pa.float64()),
            ]
        )

    def _to_record(self, data: FrameKeypointsData) -> dict:
        """Flatten FrameKeypointsData into one record per frame.

        Since this type contains a list of frames that should each become
        a row, we override encode() to handle the expansion directly.
        """
        raise NotImplementedError("Use encode() directly for FrameKeypointsData")

    def _from_record(self, record: dict, data_type: type | None = None):
        raise NotImplementedError("Use decode() directly for FrameKeypointsData")

    def encode(self, data: FrameKeypointsData | list[FrameKeypointsData]) -> pa.Table:
        """Convert FrameKeypointsData to a PyArrow Table.

        Each FrameKeypoints in data.frames becomes a row in the table.
        Encoding a list of FrameKeypointsData concatenates all frames.
        """
        frames = [f for d in data for f in d.frames] if isinstance(data, list) else data.frames

        records = [
            {
                "frame_index": frame.frame_index,
                "keypoints": frame.keypoints.tolist(),
                "confidence": frame.confidence,
            }
            for frame in frames
        ]

        return pa.Table.from_pylist(records, schema=self.schema())

    def decode(self, data: pa.Table, data_type: type | None = None) -> FrameKeypointsData:
        """Convert a PyArrow Table back to FrameKeypointsData."""
        frames = [
            FrameKeypoints(
                frame_index=row["frame_index"],
                keypoints=np.array(row["keypoints"], dtype=np.float64),
                confidence=row["confidence"],
            )
            for row in data.to_pylist()
        ]
        return FrameKeypointsData(frames=frames)
