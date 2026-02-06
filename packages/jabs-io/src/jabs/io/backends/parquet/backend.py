"""Parquet storage backend with type-based adapter dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised in runtime environments without pyarrow
    pa = None  # type: ignore
    pq = None  # type: ignore

from jabs.io.backends.base import BackendAdapterRegistry
from jabs.io.backends.parquet.base import ParquetAdapter

T = TypeVar("T")

# Supported file extensions
PARQUET_EXTENSIONS = frozenset({".parquet", ".pq"})


def _require_pyarrow() -> None:
    if pa is None or pq is None:
        raise ImportError(
            "pyarrow is required for ParquetBackend. Install jabs-io with the 'parquet' extra."
        )


PARQUET_ADAPTERS: BackendAdapterRegistry[ParquetAdapter] = BackendAdapterRegistry()


class ParquetBackend:
    """Storage backend for Parquet files.

    Uses an adapter registry for type-based dispatch, making it open for
    extension—adding support for a new type requires only registering a
    new adapter, not modifying this class.

    Example:
        >>> backend = ParquetBackend()
        >>> backend.save(my_keypoints, Path("data.parquet"))
        >>> loaded = backend.load(Path("data.parquet"), FrameKeypointsData)

    Extending with a custom type:
        >>> @PARQUET_ADAPTERS.register()
        ... class MyDataAdapter:
        ...     data_type = MyData
        ...     def to_table(self, data): ...
        ...     def from_table(self, table): ...
        >>> backend.registry.register(MyDataAdapter())
    """

    def __init__(self) -> None:
        _require_pyarrow()
        self.registry = PARQUET_ADAPTERS

    def can_handle(self, path: Path) -> bool:
        """Check if this backend can handle the given file path.

        Args:
            path: File path to check.

        Returns:
            True if the file extension is .parquet or .pq
        """
        return path.suffix.lower() in PARQUET_EXTENSIONS

    def load(self, path: Path, data_type: type[T]) -> T:
        """Load data from a Parquet file.

        Args:
            path: Path to the Parquet file.
            data_type: The domain type to deserialize into.

        Returns:
            The deserialized domain object.

        Raises:
            KeyError: If no adapter is registered for data_type.
            FileNotFoundError: If the file doesn't exist.
            pyarrow.ArrowInvalid: If the file is not valid Parquet.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        adapter = self.registry.get(data_type)
        table = pq.read_table(path)
        return adapter.from_table(table)

    def save(self, data: Any, path: Path) -> None:
        """Save data to a Parquet file.

        Args:
            data: The domain object to serialize.
            path: Destination file path.

        Raises:
            KeyError: If no adapter is registered for the data's type.
        """
        adapter = self.registry.get_for_instance(data)
        table = adapter.to_table(data)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(table, path)

    def load_with_metadata(self, path: Path, data_type: type[T]) -> tuple[T, dict]:
        """Load data along with Parquet file metadata.

        Useful for inspecting storage details or debugging.

        Args:
            path: Path to the Parquet file.
            data_type: The domain type to deserialize into.

        Returns:
            Tuple of (domain object, metadata dict).
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        adapter = self.registry.get(data_type)
        table = pq.read_table(path)

        metadata = {}
        if table.schema.metadata:
            metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

        return adapter.from_table(table), metadata
