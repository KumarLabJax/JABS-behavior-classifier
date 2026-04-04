"""Enum for feature cache storage format."""

from enum import Enum


class CacheFormat(Enum):
    """Storage format for a project's feature cache."""

    HDF5 = "hdf5"
    PARQUET = "parquet"
