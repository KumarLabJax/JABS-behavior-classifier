from enum import Enum


class StorageFormat(Enum):
    """Supported storage formats."""

    JSON = "json"
    HDF5 = "hdf5"
    PARQUET = "parquet"
    CSV = "csv"
