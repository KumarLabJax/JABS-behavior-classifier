"""JSON storage backend with generic dataclass support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from jabs.io.backends.base import BackendAdapterRegistry
from jabs.io.backends.json.base import JsonAdapter

T = TypeVar("T")

JSON_EXTENSIONS = frozenset({".json"})


JSON_ADAPTERS: BackendAdapterRegistry[JsonAdapter] = BackendAdapterRegistry()


class JsonBackend:
    """Storage backend for JSON files.

    Uses an adapter registry for type-based dispatch. Includes a built-in
    `DataclassAdapter` that handles simple dataclasses via asdict/kwargs.

    Example:
        >>> backend = JsonBackend()
        >>> backend.save(my_config, Path("config.json"))
        >>> loaded = backend.load(Path("config.json"), MyConfig)
    """

    def __init__(self) -> None:
        self.registry = JSON_ADAPTERS

    def can_handle(self, path: Path) -> bool:
        """Check if this backend can handle the given file path."""
        return path.suffix.lower() in JSON_EXTENSIONS

    def load(self, path: Path, data_type: type[T]) -> T:
        """Load data from a JSON file.

        Args:
            path: Path to the JSON file.
            data_type: The domain type to deserialize into.

        Returns:
            The deserialized domain object.

        Raises:
            KeyError: If no adapter is registered for data_type.
            FileNotFoundError: If the file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        adapter = self.registry.get(data_type)
        return adapter.from_json(path.read_text(), data_type)

    def save(self, data: Any, path: Path) -> None:
        """Save data to a JSON file.

        Args:
            data: The domain object to serialize.
            path: Destination file path.

        Raises:
            KeyError: If no adapter is registered for the data's type.
        """
        adapter = self.registry.get_for_instance(data)
        json_str = adapter.to_json(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str)
