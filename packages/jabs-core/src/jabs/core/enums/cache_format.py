"""Enum for feature cache storage format."""

from enum import Enum


class CacheFormat(str, Enum):
    """Storage format for a project's feature cache.

    Inheriting from str allows for easy serialization to/from JSON (the enum
    will automatically be serialized using the enum value).
    """

    HDF5 = "hdf5"
    PARQUET = "parquet"
