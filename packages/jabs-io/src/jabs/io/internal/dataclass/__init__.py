"""Dataclass adapters for JSON, Parquet, and HDF5 formats."""

from .hdf5 import DataclassHDF5Adapter
from .json import DataclassJSONAdapter
from .parquet import DataclassParquetAdapter

__all__ = [
    "DataclassHDF5Adapter",
    "DataclassJSONAdapter",
    "DataclassParquetAdapter",
]
