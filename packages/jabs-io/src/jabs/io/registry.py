from collections import defaultdict
from pathlib import Path
from typing import Any

from jabs.core.enums import StorageFormat
from jabs.io.base import Adapter

PATH_TO_STORAGE_FORMAT = {
    ".json": StorageFormat.JSON,
    ".csv": StorageFormat.CSV,
    ".parquet": StorageFormat.PARQUET,
    ".pq": StorageFormat.PARQUET,
    ".h5": StorageFormat.HDF5,
    ".hdf5": StorageFormat.HDF5,
}


class AdapterRegistry:
    """Central registry for managing adapters across storage formats and data types.

    Adapters are registered either for a specific (format, domain_type) pair
    or as polymorphic adapters that are resolved at lookup time via
    `can_handle()`. All adapters must be subclasses of the `Adapter` ABC
    or one of its backend-specific subclasses (JSONAdapter, ParquetAdapter,
    HDF5Adapter).
    """

    def __init__(self):
        self._adapters: dict[tuple[StorageFormat, type], list[tuple[int, Adapter]]] = defaultdict(
            list
        )
        self._format_adapters: dict[StorageFormat, set[type]] = defaultdict(set)
        self._type_formats: dict[type, set[StorageFormat]] = defaultdict(set)
        self._polymorphic_adapters: dict[StorageFormat, list[tuple[int, Adapter]]] = defaultdict(
            list
        )

    def register(
        self,
        storage_format: StorageFormat,
        adapter: Adapter,
        domain_type: type | None = None,
        priority: int = 0,
    ) -> None:
        """Register an adapter for a specific storage format and domain type combination.

        Args:
            storage_format: The storage format this adapter handles.
            adapter: The adapter instance to register. Must be a subclass of Adapter.
            domain_type: The domain type this adapter converts to and from.
                If None, the adapter is registered as polymorphic and will be
                checked via can_handle() for any type lookup.
            priority: Higher priority adapters are tried first. Defaults to 0.

        Raises:
            TypeError: If the adapter is not a subclass of Adapter.
            ValueError: If domain_type is provided but the adapter cannot handle it.
        """
        if not isinstance(adapter, Adapter):
            raise TypeError(f"Adapter must be a subclass of Adapter, got {type(adapter).__name__}")

        if domain_type is None:
            self._insert_by_priority(self._polymorphic_adapters[storage_format], priority, adapter)
        else:
            if not adapter.can_handle(domain_type):
                raise ValueError(f"Adapter {adapter} claims it cannot handle type {domain_type}")

            key = (storage_format, domain_type)
            self._insert_by_priority(self._adapters[key], priority, adapter)
            self._format_adapters[storage_format].add(domain_type)
            self._type_formats[domain_type].add(storage_format)

    @staticmethod
    def _insert_by_priority(
        adapter_list: list[tuple[int, Adapter]], priority: int, adapter: Adapter
    ) -> None:
        """Insert an adapter into a priority-sorted list (highest first)."""
        insert_pos = 0
        for i, (existing_priority, _) in enumerate(adapter_list):
            if priority > existing_priority:
                insert_pos = i
                break
            insert_pos = i + 1
        adapter_list.insert(insert_pos, (priority, adapter))

    def get_adapter(
        self,
        storage_format: StorageFormat,
        domain_type: type,
    ) -> Adapter | None:
        """Retrieve the highest priority adapter for a format-type combination.

        Attempts an exact match lookup first for optimal performance. If no
        exact match is found, falls back to checking polymorphic adapters
        via their can_handle() methods, caching successful matches.

        Args:
            storage_format: The storage format to encode to or decode from.
            domain_type: The domain type to convert to or from.

        Returns:
            The highest priority adapter if one exists, otherwise None.
        """
        key = (storage_format, domain_type)
        adapters = self._adapters.get(key, [])
        if adapters:
            return adapters[0][1]

        for priority, adapter in self._polymorphic_adapters.get(storage_format, []):
            if adapter.can_handle(domain_type):
                self._adapters[key].append((priority, adapter))
                self._format_adapters[storage_format].add(domain_type)
                self._type_formats[domain_type].add(storage_format)
                return adapter

        return None

    def get_all_adapters(
        self,
        storage_format: StorageFormat,
        domain_type: type,
    ) -> list[Adapter]:
        """Retrieve all adapters for a format-type combination, ordered by priority.

        Combines both exact matches and compatible polymorphic adapters.

        Args:
            storage_format: The storage format to encode to or decode from.
            domain_type: The domain type to convert to or from.

        Returns:
            List of adapters ordered by priority (highest first).
        """
        key = (storage_format, domain_type)
        exact_matches = self._adapters.get(key, [])

        polymorphic_matches = [
            (priority, adapter)
            for priority, adapter in self._polymorphic_adapters.get(storage_format, [])
            if adapter.can_handle(domain_type) and not any(a is adapter for _, a in exact_matches)
        ]

        all_matches = exact_matches + polymorphic_matches
        all_matches.sort(key=lambda x: x[0], reverse=True)

        return [adapter for _, adapter in all_matches]

    def supports_type(self, storage_format: StorageFormat, domain_type: type) -> bool:
        """Check if any adapter exists for the given format-type combination."""
        if domain_type in self._format_adapters[storage_format]:
            return True
        return any(
            adapter.can_handle(domain_type)
            for _, adapter in self._polymorphic_adapters.get(storage_format, [])
        )

    def get_supported_types(self, storage_format: StorageFormat) -> set[type]:
        """Get all domain types with explicitly registered adapters for a given format.

        Note: Returns only types with exact matches, not all types that
        polymorphic adapters could potentially handle.
        """
        return self._format_adapters[storage_format].copy()

    def get_supported_formats(self, domain_type: type) -> set[StorageFormat]:
        """Get all storage formats with explicitly registered adapters for a given type.

        Note: Returns only formats with exact matches, not all formats where
        polymorphic adapters could potentially handle the type.
        """
        return self._type_formats[domain_type].copy()


# ---------------------------------------------------------------------------
# Global registry and convenience functions
# ---------------------------------------------------------------------------

_global_registry = AdapterRegistry()


def register_adapter(
    storage_format: StorageFormat,
    domain_type: type | None = None,
    priority: int = 0,
):
    """Decorator for registering adapter classes with the global registry.

    Args:
        storage_format: The storage format this adapter handles.
        domain_type: The domain type this adapter converts to and from.
            If None, the adapter is registered as polymorphic.
        priority: Higher priority adapters are tried first. Defaults to 0.

    Example:
        @register_adapter(StorageFormat.JSON, dict, priority=10)
        class DictJSONAdapter(JSONAdapter):
            ...

        @register_adapter(StorageFormat.PARQUET, priority=5)
        class DataclassParquetAdapter(ParquetAdapter):
            ...
    """

    def decorator(adapter_class):
        adapter_instance = adapter_class()
        _global_registry.register(storage_format, adapter_instance, domain_type, priority)
        return adapter_class

    return decorator


def get_adapter(storage_format: StorageFormat, domain_type: type) -> Adapter | None:
    """Convenience function to get an adapter from the global registry."""
    return _global_registry.get_adapter(storage_format, domain_type)


def get_storage_format(path: Path) -> StorageFormat:
    """Resolve a file path to its corresponding StorageFormat."""
    suffix = path.suffix.lower()
    if suffix not in PATH_TO_STORAGE_FORMAT:
        raise ValueError(f"Unsupported file extension: {suffix}")
    return PATH_TO_STORAGE_FORMAT[suffix]


def get_domain_type(inst: Any) -> type | None:
    """Convenience to resolve a domain type from an instance."""
    if isinstance(inst, list):
        if not inst:
            raise ValueError(
                "Cannot infer domain type from an empty list; provide a non-empty list "
                "or specify the domain type explicitly."
            )
        return type(inst[0])
    return type(inst)
