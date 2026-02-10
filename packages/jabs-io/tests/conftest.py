"""Shared fixtures for jabs.io tests."""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from jabs.io.base import Adapter, JSONAdapter, ParquetAdapter

# ---------------------------------------------------------------------------
# Simple domain dataclass used across test modules
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    """A simple two-field dataclass for adapter tests."""

    name: str
    value: float


@dataclass
class TimestampedRecord:
    """A dataclass with a required datetime field."""

    label: str
    created_at: datetime


@dataclass
class OptionalTimestampRecord:
    """A dataclass with an optional datetime field."""

    label: str
    created_at: datetime | None = None


@dataclass
class ArrayRecord:
    """A dataclass with a list field."""

    data: list[float]


@dataclass
class NumpyRecord:
    """A dataclass with numpy array fields."""

    name: str
    values: np.ndarray
    labels: np.ndarray


@dataclass
class NestedRecord:
    """A dataclass containing another dataclass."""

    outer_name: str
    inner: SampleRecord


# ---------------------------------------------------------------------------
# Concrete adapter stubs for testing abstract base classes
# ---------------------------------------------------------------------------


class StubJSONAdapter(JSONAdapter):
    """Minimal concrete JSONAdapter for testing base-class logic."""

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return data_type is SampleRecord

    def _encode_one(self, data) -> dict:
        return {"name": data.name, "value": data.value}

    def _decode_one(self, data: dict, data_type=None):
        return SampleRecord(**data)


class StubParquetAdapter(ParquetAdapter):
    """Minimal concrete ParquetAdapter for testing base-class logic."""

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return data_type is SampleRecord

    def _to_record(self, data) -> dict:
        return {"name": data.name, "value": data.value}

    def _from_record(self, record: dict, data_type=None):
        return SampleRecord(**record)


class PolymorphicStubAdapter(Adapter):
    """Adapter that uses can_handle() to accept multiple types dynamically."""

    def __init__(self, handled_types=None):
        self._handled_types = handled_types or set()

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return False

    def encode(self, data):  # noqa: D102
        return data

    def decode(self, data, data_type=None):  # noqa: D102
        return data

    def write(self, data, path, **kwargs):  # noqa: D102
        pass

    def read(self, path, data_type=None):  # noqa: D102
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_record():
    """Return a single SampleRecord instance."""
    return SampleRecord(name="test", value=42.0)


@pytest.fixture
def sample_records():
    """Return a list of three SampleRecord instances."""
    return [
        SampleRecord(name="a", value=1.0),
        SampleRecord(name="b", value=2.0),
        SampleRecord(name="c", value=3.0),
    ]


@pytest.fixture
def stub_json_adapter():
    """Return a StubJSONAdapter instance."""
    return StubJSONAdapter()


@pytest.fixture
def stub_parquet_adapter():
    """Return a StubParquetAdapter instance."""
    return StubParquetAdapter()


@pytest.fixture
def mock_adapter():
    """A mock Adapter instance that passes isinstance checks."""
    adapter = MagicMock(spec=Adapter)
    adapter.can_handle = MagicMock(return_value=True)
    return adapter
