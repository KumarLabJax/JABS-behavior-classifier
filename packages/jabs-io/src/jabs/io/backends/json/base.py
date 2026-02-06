"""JSON adapter protocol for type-specific serialization."""

from __future__ import annotations

from typing import Protocol, TypeVar

T = TypeVar("T")


class JsonAdapter(Protocol[T]):
    """Protocol for converting domain types to/from JSON.

    Each adapter converts a domain type to JSON.
    """

    @classmethod
    def can_handle(cls, data_type):
        """Check if this adapter can handle the given data type."""

    def to_json(self, data: T) -> str:
        """Serialize a dataclass instance to a JSON string."""
        ...

    def from_json(self, json_str: str, data_type: type[T]) -> T:
        """Deserialize a JSON string to a dataclass instance."""
        ...
