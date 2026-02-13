import types
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from typing import Union, get_args, get_origin

import numpy as np

from jabs.core.enums import StorageFormat
from jabs.io.base import ParquetAdapter
from jabs.io.registry import register_adapter


@register_adapter(StorageFormat.PARQUET, priority=5)
class DataclassParquetAdapter(ParquetAdapter):
    """Parquet adapter for arbitrary dataclasses.

    Handles serialization via `asdict()` and deserialization via
    `data_type(**dict)`. Works for dataclasses with Arrow-compatible
    fields (primitives, lists, nested dataclasses with same constraints).
    Numpy arrays and scalars are recursively converted to native Python
    types, and datetimes are normalized to timezone-aware UTC.
    """

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return is_dataclass(data_type)

    @staticmethod
    def _normalize_value(obj):
        """Normalize a single value for Arrow compatibility."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, datetime):
            return obj.astimezone(timezone.utc)
        return obj

    @classmethod
    def _normalize_recursive(cls, obj):
        """Recursively normalize all values for Arrow compatibility."""
        if isinstance(obj, dict):
            return {k: cls._normalize_recursive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._normalize_recursive(item) for item in obj]
        return cls._normalize_value(obj)

    @staticmethod
    def _is_datetime_type(field_type) -> bool:
        """Check if a type annotation represents datetime or Optional[datetime]."""
        if field_type is datetime:
            return True
        origin = get_origin(field_type)
        if origin is Union or origin is types.UnionType:
            return datetime in get_args(field_type)
        return False

    def _to_record(self, data) -> dict:
        raw = asdict(data)
        return self._normalize_recursive(raw)

    def _from_record(self, record: dict, data_type: type | None = None):
        if data_type is None:
            return record
        for f in fields(data_type):
            val = record.get(f.name)
            if self._is_datetime_type(f.type) and isinstance(val, datetime) and val.tzinfo is None:
                record[f.name] = val.replace(tzinfo=timezone.utc)
        return data_type(**record)
