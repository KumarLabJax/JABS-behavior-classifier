"""Base protocols and generic registry for storage backends."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Adapter(Protocol[T]):
    """Base protocol for all adapters."""

    ...


@runtime_checkable
class BackendAdapter(Adapter[T], Protocol[T]):
    """Protocol for type-specific serialization adapters within a backend.

    Each adapter handles conversion between a specific domain type and
    the backend's native format (e.g., PyArrow Table for Parquet).
    """

    data_type: ClassVar[type]


@runtime_checkable
class PredicateAdapter(Adapter[T], Protocol[T]):
    """Protocol for predicate-based adapters that handle categories of types.

    Extends the adapter contract with a `can_handle` method for dynamic
    type matching. Checked in registration order when no exact match exists.
    """

    def can_handle(self, data_type: type) -> bool:
        """Check if this adapter can handle the given data type."""
        ...


AdapterT = TypeVar("AdapterT", bound="Adapter[Any]")


class BackendAdapterRegistry(Generic[AdapterT]):
    """Generic registry for type-based adapter dispatch.

    Each storage backend maintains its own registry instance with adapters
    specific to that backend's serialization format.

    Example:
        >>> registry: BackendAdapterRegistry[ParquetAdapter] = BackendAdapterRegistry()
        >>> registry.register(FrameKeypointsDataAdapter())
        >>> adapter = registry.get(FrameKeypointsData)
    """

    def __init__(self) -> None:
        self._adapters: dict[type, AdapterT] = {}
        self._predicate_adapters: list[AdapterT] = []

    def register(self) -> Callable[[T], T]:
        """Decorator to register an adapter.

        If the adapter implements `can_handle` (PredicateAdapter protocol),
        it is registered as a predicate adapter. Otherwise, it is registered
        as an exact-match adapter keyed by its `data_type`.

        Raises:
            ValueError: If an exact-match adapter for this type is already registered.
        """

        def _register(cls: T) -> T:
            if isinstance(cls, PredicateAdapter):
                self._predicate_adapters.append(cls())  # type: ignore[arg-type]
            else:
                data_type = cls.data_type
                if data_type in self._adapters:
                    raise ValueError(
                        f"Adapter already registered for {data_type.__name__}. "
                        f"Existing: {self._adapters[data_type].__class__.__name__}"
                    )
                self._adapters[data_type] = cls()  # type: ignore[assignment]
            return cls

        return _register

    def get(self, data_type: type) -> AdapterT:
        """Get the adapter for a specific type.

        Args:
            data_type: The domain type to look up.

        Returns:
            The registered adapter for this type.

        Raises:
            KeyError: If no adapter is registered for this type.
        """
        if data_type in self._adapters:
            return self._adapters[data_type]

        for adapter in self._predicate_adapters:
            if adapter.can_handle(data_type):
                return adapter

        registered = ", ".join(t.__name__ for t in self._adapters)
        predicate_names = ", ".join(type(a).__name__ for a in self._predicate_adapters)
        parts = [f"No adapter registered for {data_type.__name__}."]
        if registered:
            parts.append(f"Exact-match types: [{registered}].")
        if predicate_names:
            parts.append(f"Predicate adapters checked (all declined): [{predicate_names}].")
        raise KeyError(" ".join(parts))

    def get_for_instance(self, data: Any) -> AdapterT:
        """Get the adapter for a data instance.

        Args:
            data: A domain object to find an adapter for.

        Returns:
            The registered adapter for this object's type.

        Raises:
            KeyError: If no adapter is registered for this object's type.
        """
        return self.get(type(data))

    def list_types(self) -> list[type]:
        """Return all registered types in registration order."""
        return list(self._adapters.keys())


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for file format backends.

    Each backend handles reading/writing domain objects to a specific
    file format. Backends use an internal adapter registry for type dispatch.
    """

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Check if this backend can handle the given file path.

        Args:
            path: File path to check (typically by extension).

        Returns:
            True if this backend supports the file format.
        """
        ...

    @abstractmethod
    def load(self, path: Path, data_type: type[T]) -> T:
        """Load data from a file.

        Args:
            path: Path to the file to read.
            data_type: The expected domain type to deserialize into.

        Returns:
            The deserialized domain object.

        Raises:
            KeyError: If no adapter is registered for the data_type.
            FileNotFoundError: If the file doesn't exist.
        """
        ...

    @abstractmethod
    def save(self, data: Any, path: Path) -> None:
        """Save data to a file.

        Args:
            data: The domain object to serialize.
            path: Destination file path.

        Raises:
            KeyError: If no adapter is registered for the data's type.
        """
        ...
