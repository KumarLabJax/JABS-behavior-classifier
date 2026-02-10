import json
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    import h5py
except ImportError:
    h5py = None

StorageType = TypeVar("StorageType")
DomainType = TypeVar("DomainType")


class Adapter(ABC):
    """Abstract base class for all storage adapters.

    Defines the contract for encoding/decoding domain objects to and from
    a storage format, and for reading/writing that format to files.
    Both single instances and lists are supported through a union type
    signature, allowing each backend to handle batching in whatever way
    is natural for its storage format.
    """

    @classmethod
    @abstractmethod
    def can_handle(cls, data_type: type) -> bool:
        """Check if this adapter can handle the given data type."""

    @abstractmethod
    def write(self, data: DomainType | list[DomainType], path: str | Path, **kwargs) -> None:
        """Encode and write domain object(s) to a file."""

    @abstractmethod
    def read(
        self, path: str | Path, data_type: type | None = None
    ) -> DomainType | list[DomainType]:
        """Read a file and decode into domain object(s)."""


class JSONAdapter(Adapter):
    """Base class for JSON-based adapters.

    Accepts a single domain object or a list. A single object is serialized
    as a JSON object; a list is serialized as a JSON array. Decoding
    mirrors this: a JSON array returns a list, a JSON object returns a
    single instance.

    Concrete adapters must implement `_encode_one` and `_decode_one`
    for single-instance conversion. The base class handles list dispatch,
    file I/O, and JSON serialization.
    """

    @abstractmethod
    def _encode_one(self, data) -> dict:
        """Convert a single domain object to a JSON-serializable dict."""

    @abstractmethod
    def _decode_one(self, data: dict, data_type: type | None = None):
        """Reconstruct a single domain object from a parsed dict."""

    @staticmethod
    def _json_default(obj):
        """Handle non-JSON-serializable types during encoding.

        Provides sensible defaults for common scientific Python types.
        Subclasses can override this to handle additional types or
        change the conversion behavior.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, datetime):
            return obj.astimezone(timezone.utc).isoformat()
        if is_dataclass(obj):
            return asdict(obj)

        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def encode(self, data) -> str:  # noqa: D102
        if isinstance(data, list):
            payload = [self._encode_one(item) for item in data]
        else:
            payload = self._encode_one(data)
        return json.dumps(payload, default=self._json_default)

    def decode(self, data: str, data_type: type | None = None):  # noqa: D102
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return [self._decode_one(item, data_type) for item in parsed]
        return self._decode_one(parsed, data_type)

    def write(self, data, path: str, **kwargs) -> None:  # noqa: D102
        encoded = self.encode(data)
        Path(path).write_text(encoded, encoding="utf-8")

    def read(self, path: str, data_type: type | None = None):  # noqa: D102
        text = Path(path).read_text(encoding="utf-8")
        return self.decode(text, data_type)


class ParquetAdapter(Adapter):
    """Base class for Parquet/Arrow-based adapters.

    Accepts a single domain object or a list and always produces a
    `pa.Table` on encode. On decode, a table with a single row returns
    a single instance; multiple rows return a list.

    Concrete adapters must implement `_to_record` and `_from_record`
    for converting between domain objects and plain dicts. The base
    class handles Arrow Table construction, row dispatch, and
    Parquet file I/O.
    """

    def __init__(self):
        self._require_pyarrow()

    @staticmethod
    def _require_pyarrow() -> None:
        if pa is None or pq is None:
            raise ImportError(
                "pyarrow is required to use Parquet. Install jabs-io with the 'parquet' extra."
            )

    @abstractmethod
    def _to_record(self, data) -> dict:
        """Convert a single domain object to a flat dict for Arrow."""

    @abstractmethod
    def _from_record(self, record: dict, data_type: type | None = None):
        """Reconstruct a single domain object from a dict row."""

    def schema(self) -> pa.Schema | None:
        """Return an explicit PyArrow schema, or None for automatic inference."""
        return None

    def encode(self, data) -> pa.Table:  # noqa: D102
        if not isinstance(data, list):
            data = [data]
        records = [self._to_record(item) for item in data]
        arrow_schema = self.schema()
        if arrow_schema is not None:
            return pa.Table.from_pylist(records, schema=arrow_schema)
        return pa.Table.from_pylist(records)

    def decode(self, data: pa.Table, data_type: type | None = None):  # noqa: D102
        rows = data.to_pylist()
        results = [self._from_record(row, data_type) for row in rows]
        if len(results) == 1:
            return results[0]
        return results

    def write(self, data, path: str, **kwargs) -> None:  # noqa: D102
        table = data if isinstance(data, pa.Table) else self.encode(data)
        pq.write_table(table, path, **kwargs)

    def read(self, path: str, data_type: type | None = None):  # noqa: D102
        table = pq.read_table(path)
        return self.decode(table, data_type)


class HDF5Adapter(Adapter):
    """Base class for HDF5-based adapters.

    HDF5 is inherently file-based, so ``encode``/``decode`` are not
    meaningful and raise ``NotImplementedError``.  Real work is done
    in ``write`` and ``read``, which delegate to the abstract helpers
    ``_write_one`` and ``_read_one``.

    Lists are stored as numbered subgroups (``_item_0``, ``_item_1``, ...)
    with a root attribute ``_list_length`` recording the count.

    Concrete adapters must implement ``_write_one`` and ``_read_one``.
    """

    def __init__(self):
        if h5py is None:
            raise ImportError(
                "h5py is required to use HDF5. Install jabs-io with the 'hdf5' extra."
            )

    @abstractmethod
    def _write_one(self, data, group) -> None:
        """Write a single domain object into an h5py Group."""

    @abstractmethod
    def _read_one(self, group, data_type: type | None = None):
        """Read a single domain object from an h5py Group."""

    def write(self, data, path: str | Path, **kwargs) -> None:  # noqa: D102
        with h5py.File(path, "w") as h5:
            if isinstance(data, list):
                h5.attrs["_list_length"] = len(data)
                for i, item in enumerate(data):
                    grp = h5.create_group(f"_item_{i}")
                    self._write_one(item, grp)
            else:
                self._write_one(data, h5)

    def read(self, path: str | Path, data_type: type | None = None):  # noqa: D102
        with h5py.File(path, "r") as h5:
            if "_list_length" in h5.attrs:
                length = int(h5.attrs["_list_length"])
                return [self._read_one(h5[f"_item_{i}"], data_type) for i in range(length)]
            return self._read_one(h5, data_type)
