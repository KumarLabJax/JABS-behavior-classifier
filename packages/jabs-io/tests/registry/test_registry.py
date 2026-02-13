"""Tests for AdapterRegistry and module-level convenience functions."""

from pathlib import Path

import pytest

from jabs.core.enums import StorageFormat
from jabs.io.base import Adapter
from jabs.io.registry import (
    PATH_TO_STORAGE_FORMAT,
    AdapterRegistry,
    get_domain_type,
    get_storage_format,
)
from tests.conftest import SampleRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter_class(can_handle_types=None):
    """Create a minimal Adapter subclass that can be registered."""
    handle_types = can_handle_types or set()

    class MockAdapter(Adapter):
        can_handle_call_count = 0

        @classmethod
        def can_handle(cls, data_type):
            cls.can_handle_call_count += 1
            return data_type in handle_types

        def write(self, data, path, **kwargs):
            pass

        def read(self, path, data_type=None):
            pass

    return MockAdapter


# ---------------------------------------------------------------------------
# AdapterRegistry.register
# ---------------------------------------------------------------------------


def test_register_with_domain_type():
    """Registering with an explicit domain type makes the adapter discoverable."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, adapter_cls, SampleRecord)
    assert isinstance(reg.get_adapter(StorageFormat.JSON, SampleRecord), adapter_cls)


def test_register_polymorphic():
    """Registering without a domain type creates a polymorphic adapter."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, adapter_cls, domain_type=None)
    # Should be discoverable via can_handle fallback
    assert isinstance(reg.get_adapter(StorageFormat.JSON, SampleRecord), adapter_cls)


def test_register_rejects_non_adapter():
    """Registering a non-Adapter instance raises TypeError."""
    reg = AdapterRegistry()
    with pytest.raises(TypeError, match="must be a subclass of Adapter"):
        reg.register(StorageFormat.JSON, "not an adapter", SampleRecord)


def test_register_rejects_adapter_that_cannot_handle_type():
    """Registering an adapter that cannot handle the claimed type raises ValueError."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class(set())  # can_handle returns False for everything
    with pytest.raises(ValueError, match="cannot handle type"):
        reg.register(StorageFormat.JSON, adapter_cls, SampleRecord)


# ---------------------------------------------------------------------------
# AdapterRegistry.get_adapter — priority
# ---------------------------------------------------------------------------


def test_higher_priority_wins():
    """The adapter with the highest priority is returned first."""
    reg = AdapterRegistry()
    low_cls = _make_adapter_class({SampleRecord})
    high_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, low_cls, SampleRecord, priority=0)
    reg.register(StorageFormat.JSON, high_cls, SampleRecord, priority=10)
    assert isinstance(reg.get_adapter(StorageFormat.JSON, SampleRecord), high_cls)


def test_equal_priority_first_registered_wins():
    """When priorities tie, the first registered adapter wins."""
    reg = AdapterRegistry()
    first_cls = _make_adapter_class({SampleRecord})
    second_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, first_cls, SampleRecord, priority=0)
    reg.register(StorageFormat.JSON, second_cls, SampleRecord, priority=0)
    assert isinstance(reg.get_adapter(StorageFormat.JSON, SampleRecord), first_cls)


def test_get_adapter_returns_none_for_unknown():
    """get_adapter returns None when no adapter matches."""
    reg = AdapterRegistry()
    assert reg.get_adapter(StorageFormat.JSON, SampleRecord) is None


# ---------------------------------------------------------------------------
# Polymorphic caching
# ---------------------------------------------------------------------------


def test_polymorphic_lookup_is_cached():
    """After a polymorphic adapter matches, subsequent lookups should be O(1)."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, adapter_cls, domain_type=None)

    # First lookup triggers can_handle
    result1 = reg.get_adapter(StorageFormat.JSON, SampleRecord)
    # Reset counter after first lookup
    initial_count = adapter_cls.can_handle_call_count

    # Second lookup should hit cache (no additional can_handle call)
    result2 = reg.get_adapter(StorageFormat.JSON, SampleRecord)

    assert isinstance(result1, adapter_cls)
    assert isinstance(result2, adapter_cls)
    # can_handle should not have been called again for the second lookup
    assert adapter_cls.can_handle_call_count == initial_count


# ---------------------------------------------------------------------------
# AdapterRegistry.get_all_adapters
# ---------------------------------------------------------------------------


def test_get_all_adapters_combines_exact_and_polymorphic():
    """get_all_adapters returns both exact and polymorphic matches, sorted by priority."""
    reg = AdapterRegistry()
    exact_cls = _make_adapter_class({SampleRecord})
    poly_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, exact_cls, SampleRecord, priority=5)
    reg.register(StorageFormat.JSON, poly_cls, domain_type=None, priority=3)

    all_adapters = reg.get_all_adapters(StorageFormat.JSON, SampleRecord)
    assert exact_cls in all_adapters
    assert poly_cls in all_adapters
    # Higher priority first
    assert all_adapters[0] is exact_cls


def test_get_all_adapters_no_duplicates():
    """A polymorphic adapter already cached as exact should not appear twice."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, adapter_cls, domain_type=None)

    # Trigger caching
    reg.get_adapter(StorageFormat.JSON, SampleRecord)

    all_adapters = reg.get_all_adapters(StorageFormat.JSON, SampleRecord)
    assert all_adapters.count(adapter_cls) == 1


def test_get_all_adapters_empty():
    """get_all_adapters returns an empty list when nothing matches."""
    reg = AdapterRegistry()
    assert reg.get_all_adapters(StorageFormat.JSON, SampleRecord) == []


# ---------------------------------------------------------------------------
# supports_type / get_supported_types / get_supported_formats
# ---------------------------------------------------------------------------


def test_supports_type_exact():
    """supports_type returns True for explicitly registered format-type pairs."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, adapter_cls, SampleRecord)
    assert reg.supports_type(StorageFormat.JSON, SampleRecord) is True
    assert reg.supports_type(StorageFormat.PARQUET, SampleRecord) is False


def test_supports_type_polymorphic():
    """supports_type returns True when a polymorphic adapter can handle the type."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, adapter_cls, domain_type=None)
    assert reg.supports_type(StorageFormat.JSON, SampleRecord) is True


def test_get_supported_types():
    """get_supported_types returns all explicitly registered types for a format."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, adapter_cls, SampleRecord)
    types = reg.get_supported_types(StorageFormat.JSON)
    assert SampleRecord in types


def test_get_supported_types_returns_copy():
    """get_supported_types returns a copy, not a reference to internal state."""
    reg = AdapterRegistry()
    types = reg.get_supported_types(StorageFormat.JSON)
    types.add(int)  # Mutate the copy
    assert int not in reg.get_supported_types(StorageFormat.JSON)


def test_get_supported_formats():
    """get_supported_formats returns all formats registered for a type."""
    reg = AdapterRegistry()
    adapter_cls = _make_adapter_class({SampleRecord})
    reg.register(StorageFormat.JSON, adapter_cls, SampleRecord)
    formats = reg.get_supported_formats(SampleRecord)
    assert StorageFormat.JSON in formats


def test_get_supported_formats_returns_copy():
    """get_supported_formats returns a copy, not a reference to internal state."""
    reg = AdapterRegistry()
    formats = reg.get_supported_formats(SampleRecord)
    formats.add(StorageFormat.CSV)
    assert StorageFormat.CSV not in reg.get_supported_formats(SampleRecord)


# ---------------------------------------------------------------------------
# Module-level: get_storage_format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix, expected",
    [
        (".json", StorageFormat.JSON),
        (".parquet", StorageFormat.PARQUET),
        (".pq", StorageFormat.PARQUET),
        (".h5", StorageFormat.HDF5),
        (".hdf5", StorageFormat.HDF5),
        (".csv", StorageFormat.CSV),
    ],
)
def test_get_storage_format(suffix, expected):
    """get_storage_format maps each file extension to the correct StorageFormat."""
    assert get_storage_format(Path(f"data{suffix}")) == expected


def test_get_storage_format_case_insensitive():
    """get_storage_format is case-insensitive for file extensions."""
    assert get_storage_format(Path("data.JSON")) == StorageFormat.JSON
    assert get_storage_format(Path("data.Parquet")) == StorageFormat.PARQUET


def test_get_storage_format_unsupported_extension():
    """get_storage_format raises ValueError for unknown extensions."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        get_storage_format(Path("data.xyz"))


def test_get_storage_format_no_extension():
    """get_storage_format raises ValueError when there is no file extension."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        get_storage_format(Path("noext"))


# ---------------------------------------------------------------------------
# Module-level: get_domain_type
# ---------------------------------------------------------------------------


def test_get_domain_type_single_instance():
    """get_domain_type returns the type of a single instance."""
    record = SampleRecord(name="x", value=1.0)
    assert get_domain_type(record) is SampleRecord


def test_get_domain_type_list():
    """get_domain_type returns the type of the first element in a list."""
    records = [SampleRecord(name="x", value=1.0)]
    assert get_domain_type(records) is SampleRecord


def test_get_domain_type_primitive():
    """get_domain_type works with primitive types."""
    assert get_domain_type(42) is int
    assert get_domain_type("hello") is str


# ---------------------------------------------------------------------------
# Module-level: register_adapter decorator
# ---------------------------------------------------------------------------


def test_register_adapter_decorator():
    """The @register_adapter decorator should make adapters discoverable in the global registry."""
    from jabs.io.registry import _global_registry

    # The DataclassJSONAdapter is registered at import time via @register_adapter
    # Verify it can handle dataclasses through the global registry
    adapter = _global_registry.get_adapter(StorageFormat.JSON, SampleRecord)
    assert adapter is not None


# ---------------------------------------------------------------------------
# PATH_TO_STORAGE_FORMAT completeness
# ---------------------------------------------------------------------------


def test_path_to_storage_format_covers_all_documented_extensions():
    """All extensions in PATH_TO_STORAGE_FORMAT should map to valid StorageFormat members."""
    for suffix, fmt in PATH_TO_STORAGE_FORMAT.items():
        assert suffix.startswith(".")
        assert isinstance(fmt, StorageFormat)
