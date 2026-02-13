"""Tests for DataclassJSONAdapter."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from jabs.io.internal.dataclass.json import DataclassJSONAdapter
from tests.conftest import (
    OptionalTimestampRecord,
    SampleRecord,
    TimestampedRecord,
)


@pytest.fixture
def adapter():
    """Return a DataclassJSONAdapter instance."""
    return DataclassJSONAdapter()


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_type, expected",
    [
        (SampleRecord, True),
        (TimestampedRecord, True),
        (dict, False),
        (int, False),
        (str, False),
        (list, False),
    ],
    ids=["dataclass", "dataclass-datetime", "dict", "int", "str", "list"],
)
def test_can_handle(data_type, expected):
    """can_handle returns True only for dataclass types."""
    assert DataclassJSONAdapter.can_handle(data_type) is expected


# ---------------------------------------------------------------------------
# _is_datetime_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (datetime, True),
        (datetime | None, True),
        (str, False),
        (int | None, False),
        (str | datetime, True),
        (float, False),
    ],
    ids=["datetime", "optional-datetime", "str", "optional-int", "union-with-datetime", "float"],
)
def test_is_datetime_type(annotation, expected):
    """_is_datetime_type detects datetime and Optional[datetime] annotations."""
    assert DataclassJSONAdapter._is_datetime_type(annotation) is expected


# ---------------------------------------------------------------------------
# _encode_one
# ---------------------------------------------------------------------------


def test_encode_one_simple(adapter):
    """_encode_one converts a simple dataclass to a dict via asdict."""
    record = SampleRecord(name="a", value=1.5)
    result = adapter._encode_one(record)
    assert result == {"name": "a", "value": 1.5}


def test_encode_one_with_datetime(adapter):
    """_encode_one preserves datetime objects in the dict."""
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    record = TimestampedRecord(label="event", created_at=dt)
    result = adapter._encode_one(record)
    assert result["label"] == "event"
    assert result["created_at"] == dt


def test_encode_one_nested_dataclass(adapter):
    """_encode_one recursively flattens nested dataclasses via asdict."""

    @dataclass
    class Inner:
        x: int

    @dataclass
    class Outer:
        inner: Inner

    result = adapter._encode_one(Outer(inner=Inner(x=5)))
    assert result == {"inner": {"x": 5}}


# ---------------------------------------------------------------------------
# _decode_one
# ---------------------------------------------------------------------------


def test_decode_one_simple(adapter):
    """_decode_one reconstructs a dataclass from a plain dict."""
    result = adapter._decode_one({"name": "z", "value": 9.0}, SampleRecord)
    assert result == SampleRecord(name="z", value=9.0)


def test_decode_one_no_data_type_returns_dict(adapter):
    """_decode_one returns the raw dict when data_type is None."""
    raw = {"name": "z", "value": 9.0}
    result = adapter._decode_one(raw, data_type=None)
    assert result == raw
    assert isinstance(result, dict)


def test_decode_one_parses_datetime_string(adapter):
    """_decode_one converts ISO datetime strings to datetime objects."""
    iso = "2024-06-01T12:00:00+00:00"
    result = adapter._decode_one({"label": "e", "created_at": iso}, TimestampedRecord)
    assert isinstance(result.created_at, datetime)
    assert result.created_at.tzinfo is not None


def test_decode_one_parses_optional_datetime_string(adapter):
    """_decode_one parses ISO strings for Optional[datetime] fields too."""
    iso = "2024-01-01T00:00:00+00:00"
    result = adapter._decode_one({"label": "e", "created_at": iso}, OptionalTimestampRecord)
    assert isinstance(result.created_at, datetime)


def test_decode_one_optional_datetime_none(adapter):
    """_decode_one leaves None values alone for Optional[datetime] fields."""
    result = adapter._decode_one({"label": "e", "created_at": None}, OptionalTimestampRecord)
    assert result.created_at is None


def test_decode_one_leaves_non_str_datetime_alone(adapter):
    """If the datetime field already holds a datetime object, don't re-parse."""
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    result = adapter._decode_one({"label": "e", "created_at": dt}, TimestampedRecord)
    assert result.created_at is dt


# ---------------------------------------------------------------------------
# Full encode → JSON string → decode round-trips
# ---------------------------------------------------------------------------


def test_roundtrip_simple(adapter):
    """A simple dataclass survives encode-then-decode."""
    record = SampleRecord(name="rt", value=3.14)
    encoded = adapter.encode(record)
    decoded = adapter.decode(encoded, SampleRecord)
    assert decoded == record


def test_roundtrip_with_datetime(adapter):
    """A datetime-bearing dataclass survives encode-then-decode."""
    dt = datetime(2024, 6, 15, 8, 30, 0, tzinfo=timezone.utc)
    record = TimestampedRecord(label="ts", created_at=dt)
    encoded = adapter.encode(record)
    decoded = adapter.decode(encoded, TimestampedRecord)
    assert decoded == record


def test_roundtrip_optional_datetime_present(adapter):
    """An Optional[datetime] field with a value survives the round-trip."""
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    record = OptionalTimestampRecord(label="opt", created_at=dt)
    encoded = adapter.encode(record)
    decoded = adapter.decode(encoded, OptionalTimestampRecord)
    assert decoded == record


def test_roundtrip_optional_datetime_none(adapter):
    """An Optional[datetime] field set to None survives the round-trip."""
    record = OptionalTimestampRecord(label="opt", created_at=None)
    encoded = adapter.encode(record)
    decoded = adapter.decode(encoded, OptionalTimestampRecord)
    assert decoded == record


def test_roundtrip_list(adapter):
    """A list of dataclasses survives encode-then-decode."""
    records = [SampleRecord(name="a", value=1.0), SampleRecord(name="b", value=2.0)]
    encoded = adapter.encode(records)
    decoded = adapter.decode(encoded, SampleRecord)
    assert decoded == records


def test_roundtrip_empty_list(adapter):
    """An empty list survives encode-then-decode."""
    encoded = adapter.encode([])
    decoded = adapter.decode(encoded, SampleRecord)
    assert decoded == []


# ---------------------------------------------------------------------------
# Numpy values via _json_default
# ---------------------------------------------------------------------------


def test_encode_numpy_array_field(adapter):
    """numpy arrays in dataclass fields should serialize via _json_default."""

    @dataclass
    class NumpyDC:
        arr: np.ndarray

    record = NumpyDC(arr=np.array([1, 2, 3]))
    encoded = adapter.encode(record)
    parsed = json.loads(encoded)
    assert parsed["arr"] == [1, 2, 3]


def test_encode_numpy_scalar_field(adapter):
    """numpy scalars in dataclass fields should serialize via _json_default."""

    @dataclass
    class ScalarDC:
        val: float

    record = ScalarDC(val=np.float64(2.5))
    encoded = adapter.encode(record)
    parsed = json.loads(encoded)
    assert parsed["val"] == 2.5


# ---------------------------------------------------------------------------
# File I/O round-trips
# ---------------------------------------------------------------------------


def test_file_roundtrip_single(tmp_path, adapter):
    """A single dataclass survives a file write-then-read round-trip."""
    path = tmp_path / "single.json"
    record = SampleRecord(name="file", value=7.0)
    adapter.write(record, path)
    loaded = adapter.read(path, SampleRecord)
    assert loaded == record


def test_file_roundtrip_list(tmp_path, adapter):
    """A list of dataclasses survives a file write-then-read round-trip."""
    path = tmp_path / "list.json"
    records = [SampleRecord(name="a", value=1.0), SampleRecord(name="b", value=2.0)]
    adapter.write(records, path)
    loaded = adapter.read(path, SampleRecord)
    assert loaded == records


def test_file_roundtrip_datetime(tmp_path, adapter):
    """A datetime-bearing dataclass survives a file write-then-read round-trip."""
    path = tmp_path / "ts.json"
    dt = datetime(2024, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    record = TimestampedRecord(label="ts", created_at=dt)
    adapter.write(record, path)
    loaded = adapter.read(path, TimestampedRecord)
    assert loaded == record


def test_read_without_data_type_returns_dict(tmp_path, adapter):
    """Reading without a data_type returns a raw dict."""
    path = tmp_path / "raw.json"
    adapter.write(SampleRecord(name="x", value=0.0), path)
    loaded = adapter.read(path, data_type=None)
    assert isinstance(loaded, dict)
    assert loaded["name"] == "x"
