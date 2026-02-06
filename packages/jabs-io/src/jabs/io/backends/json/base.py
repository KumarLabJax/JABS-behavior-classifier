"""Parquet adapter protocol for type-specific serialization."""

from __future__ import annotations

from typing import ClassVar, Protocol, TypeVar

T = TypeVar("T")


class JsonAdapter(Protocol[T]):
    """Protocol for converting domain types to/from JSON.

    Each adapter converts a domain type to JSON.
    """

    data_type: ClassVar[type]

    def to_json(self, data: T) -> str:
        """Serialize a dataclass instance to a JSON string."""

    def from_json(self, json_str: str) -> T:
        """Deserialize a JSON string to a dataclass i nstance."""
