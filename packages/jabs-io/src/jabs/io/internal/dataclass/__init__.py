"""Dataclass adapters for JSON and Parquet formats."""

from .json import DataclassJSONAdapter
from .parquet import DataclassParquetAdapter

__all__ = [
    "DataclassJSONAdapter",
    "DataclassParquetAdapter",
]
