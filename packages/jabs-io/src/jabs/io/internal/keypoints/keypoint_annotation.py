"""Concrete Parquet adapters for domain types.

Design principle: Use native Parquet types wherever possible.
- Scalars -> native columns
- numpy arrays -> fixed-size lists of floats
- Lists of ints -> native list<int64>
- Nested dataclasses -> nested structs
- Dicts -> JSON string (Parquet maps are less portable)
"""

import numpy as np

try:
    import pyarrow as pa
except ImportError:
    pa = None

from jabs.core.enums import StorageFormat
from jabs.core.types.keypoints import (
    KeypointAnnotation,
)
from jabs.io.base import ParquetAdapter
from jabs.io.registry import register_adapter


@register_adapter(StorageFormat.PARQUET, KeypointAnnotation, priority=10)
class KeypointAnnotationAdapter(ParquetAdapter):
    """Parquet adapter for KeypointAnnotation.

    Storage: Single row.
    - keypoints: list<list<float64, 2>> representing (K, 2) array
    - kept_frame_indices: list<int64>
    - mean_confidence: float64 (nullable)
    """

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return data_type is KeypointAnnotation

    def schema(self) -> pa.Schema:  # noqa: D102
        return pa.schema(
            [
                pa.field("keypoints", pa.list_(pa.list_(pa.float64(), 2))),
                pa.field("kept_frame_indices", pa.list_(pa.int64())),
                pa.field("mean_confidence", pa.float64()),
            ]
        )

    def _to_record(self, data: KeypointAnnotation) -> dict:
        return {
            "keypoints": data.keypoints.tolist(),
            "kept_frame_indices": data.kept_frame_indices,
            "mean_confidence": data.mean_confidence,
        }

    def _from_record(self, record: dict, data_type: type | None = None):
        return KeypointAnnotation(
            keypoints=np.array(record["keypoints"], dtype=np.float64),
            kept_frame_indices=record["kept_frame_indices"],
            mean_confidence=record["mean_confidence"],
        )
