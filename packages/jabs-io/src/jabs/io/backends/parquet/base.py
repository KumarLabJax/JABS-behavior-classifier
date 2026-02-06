"""Parquet adapter protocol for type-specific serialization."""

from __future__ import annotations

from typing import ClassVar, Protocol, TypeVar

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised in runtime environments without pyarrow
    pa = None  # type: ignore
    pq = None  # type: ignore

T = TypeVar("T")


def _require_pyarrow() -> None:
    if pa is None or pq is None:
        raise ImportError(
            "pyarrow is required for ParquetBackend. Install jabs-io with the 'parquet' extra."
        )


class ParquetAdapter(Protocol[T]):
    """Protocol for converting domain types to/from PyArrow Tables.

    Each adapter handles one domain type, providing bidirectional conversion
    between the domain object and a PyArrow Table suitable for Parquet storage.

    The adapter is responsible for:
    - Defining an appropriate schema for the data
    - Handling nested structures (arrays, dicts) via Parquet's native types
    - Preserving type information for round-trip serialization

    Example:
        >>> class MyDataAdapter:
        ...     data_type: ClassVar[type] = MyData
        ...
        ...     def to_table(self, data: MyData) -> pa.Table:
        ...         return pa.table({"field": [data.field]})
        ...
        ...     def from_table(self, table: pa.Table) -> MyData:
        ...         return MyData(field=table["field"][0].as_py())
    """

    data_type: ClassVar[type]
    """The domain type this adapter serializes."""

    def to_table(self, data: T) -> pa.Table:
        """Convert a domain object to a PyArrow Table.

        Args:
            data: The domain object to serialize.

        Returns:
            A PyArrow Table representing the data.
        """
        ...

    def from_table(self, table: pa.Table) -> T:
        """Convert a PyArrow Table back to a domain object.

        Args:
            table: The table read from a Parquet file.

        Returns:
            The reconstructed domain object.
        """
        ...

    def schema(self) -> pa.Schema:
        """Return the PyArrow schema for this data type.

        This is optional but useful for documentation and validation.

        Returns:
            The schema that to_table() produces.
        """
        ...
