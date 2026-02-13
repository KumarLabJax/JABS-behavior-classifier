"""Tests for DataclassParquetAdapter."""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from jabs.io.internal.dataclass.parquet import DataclassParquetAdapter
from tests.conftest import (
    OptionalTimestampRecord,
    SampleRecord,
    TimestampedRecord,
)


@pytest.fixture
def adapter():
    """Return a DataclassParquetAdapter instance."""
    return DataclassParquetAdapter()


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
    assert DataclassParquetAdapter.can_handle(data_type) is expected


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
    assert DataclassParquetAdapter._is_datetime_type(annotation) is expected


# ---------------------------------------------------------------------------
# _normalize_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (np.array([1, 2, 3]), [1, 2, 3]),
        (np.float64(3.14), 3.14),
        (np.int32(7), 7),
        (np.bool_(True), True),
    ],
    ids=["ndarray", "float64", "int32", "bool_"],
)
def test_normalize_value_numpy(input_val, expected):
    """_normalize_value converts numpy types to native Python equivalents."""
    result = DataclassParquetAdapter._normalize_value(input_val)
    assert result == expected


def test_normalize_value_datetime_to_utc():
    """_normalize_value converts non-UTC datetimes to UTC."""
    from datetime import timedelta

    est = timezone(timedelta(hours=-5))
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=est)
    result = DataclassParquetAdapter._normalize_value(dt)
    assert result.tzinfo == timezone.utc
    assert result.hour == 17  # 12 EST = 17 UTC


def test_normalize_value_passthrough():
    """_normalize_value passes through types it does not need to convert."""
    assert DataclassParquetAdapter._normalize_value("hello") == "hello"
    assert DataclassParquetAdapter._normalize_value(42) == 42
    assert DataclassParquetAdapter._normalize_value(None) is None


# ---------------------------------------------------------------------------
# _normalize_recursive
# ---------------------------------------------------------------------------


def test_normalize_recursive_dict():
    """_normalize_recursive converts numpy values inside dicts."""
    result = DataclassParquetAdapter._normalize_recursive(
        {"a": np.float64(1.0), "b": np.array([2, 3])}
    )
    assert result == {"a": 1.0, "b": [2, 3]}


def test_normalize_recursive_list():
    """_normalize_recursive converts numpy values inside lists."""
    result = DataclassParquetAdapter._normalize_recursive([np.int32(1), np.int32(2)])
    assert result == [1, 2]


def test_normalize_recursive_nested():
    """_normalize_recursive handles arbitrarily nested dicts and lists."""
    result = DataclassParquetAdapter._normalize_recursive(
        {"outer": {"inner": np.array([10, 20])}, "items": [np.float64(3.14)]}
    )
    assert result == {"outer": {"inner": [10, 20]}, "items": [3.14]}


# ---------------------------------------------------------------------------
# _to_record
# ---------------------------------------------------------------------------


def test_to_record_simple(adapter):
    """_to_record converts a simple dataclass to a flat dict."""
    record = SampleRecord(name="a", value=1.5)
    result = adapter._to_record(record)
    assert result == {"name": "a", "value": 1.5}


def test_to_record_normalizes_numpy(adapter):
    """_to_record normalizes numpy values for Arrow compatibility."""

    @dataclass
    class NumpyDC:
        arr: list
        scalar: float

    record = NumpyDC(arr=np.array([1, 2]), scalar=np.float64(9.9))
    result = adapter._to_record(record)
    assert result["arr"] == [1, 2]
    assert isinstance(result["scalar"], float)


def test_to_record_normalizes_datetime(adapter):
    """_to_record normalizes datetimes to UTC."""
    dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    record = TimestampedRecord(label="x", created_at=dt)
    result = adapter._to_record(record)
    assert result["created_at"].tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# _from_record
# ---------------------------------------------------------------------------


def test_from_record_simple(adapter):
    """_from_record reconstructs a dataclass from a dict."""
    result = adapter._from_record({"name": "z", "value": 9.0}, SampleRecord)
    assert result == SampleRecord(name="z", value=9.0)


def test_from_record_no_data_type_returns_dict(adapter):
    """_from_record returns the raw dict when data_type is None."""
    raw = {"name": "z", "value": 9.0}
    result = adapter._from_record(raw, data_type=None)
    assert result is raw


def test_from_record_adds_utc_to_naive_datetime(adapter):
    """Parquet may strip timezone; _from_record should restore UTC for datetime fields."""
    naive_dt = datetime(2024, 6, 1, 12, 0, 0)
    result = adapter._from_record({"label": "e", "created_at": naive_dt}, TimestampedRecord)
    assert result.created_at.tzinfo == timezone.utc


def test_from_record_leaves_aware_datetime_alone(adapter):
    """_from_record does not modify already-aware datetime fields."""
    aware_dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = adapter._from_record({"label": "e", "created_at": aware_dt}, TimestampedRecord)
    assert result.created_at is aware_dt


def test_from_record_optional_datetime_naive(adapter):
    """_from_record restores UTC for naive datetimes in Optional[datetime] fields."""
    naive_dt = datetime(2024, 1, 1)
    result = adapter._from_record({"label": "e", "created_at": naive_dt}, OptionalTimestampRecord)
    assert result.created_at.tzinfo == timezone.utc


def test_from_record_optional_datetime_none(adapter):
    """_from_record leaves None alone for Optional[datetime] fields."""
    result = adapter._from_record({"label": "e", "created_at": None}, OptionalTimestampRecord)
    assert result.created_at is None


# ---------------------------------------------------------------------------
# Full encode → pa.Table → decode round-trips
# ---------------------------------------------------------------------------


def test_roundtrip_simple(adapter):
    """A simple dataclass survives encode-then-decode."""
    record = SampleRecord(name="rt", value=3.14)
    table = adapter.encode(record)
    decoded = adapter.decode(table, SampleRecord)
    assert decoded == record


def test_roundtrip_with_datetime(adapter):
    """A datetime-bearing dataclass survives encode-then-decode."""
    dt = datetime(2024, 6, 15, 8, 30, 0, tzinfo=timezone.utc)
    record = TimestampedRecord(label="ts", created_at=dt)
    table = adapter.encode(record)
    decoded = adapter.decode(table, TimestampedRecord)
    assert decoded == record


def test_roundtrip_list(adapter):
    """A list of dataclasses survives encode-then-decode."""
    records = [SampleRecord(name="a", value=1.0), SampleRecord(name="b", value=2.0)]
    table = adapter.encode(records)
    decoded = adapter.decode(table, SampleRecord)
    assert decoded == records


def test_roundtrip_empty_list(adapter):
    """Encoding an empty list produces a zero-row table."""
    table = adapter.encode([])
    assert table.num_rows == 0


# ---------------------------------------------------------------------------
# File I/O round-trips
# ---------------------------------------------------------------------------


def test_file_roundtrip_single(tmp_path, adapter):
    """A single dataclass survives a file write-then-read round-trip."""
    path = tmp_path / "single.parquet"
    record = SampleRecord(name="file", value=7.0)
    adapter.write(record, path)
    loaded = adapter.read(path, SampleRecord)
    assert loaded == record


def test_file_roundtrip_list(tmp_path, adapter):
    """A list of dataclasses survives a file write-then-read round-trip."""
    path = tmp_path / "list.parquet"
    records = [SampleRecord(name="a", value=1.0), SampleRecord(name="b", value=2.0)]
    adapter.write(records, path)
    loaded = adapter.read(path, SampleRecord)
    assert loaded == records


def test_file_roundtrip_datetime(tmp_path, adapter):
    """A datetime-bearing dataclass survives a file write-then-read round-trip."""
    path = tmp_path / "ts.parquet"
    dt = datetime(2024, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    record = TimestampedRecord(label="ts", created_at=dt)
    adapter.write(record, path)
    loaded = adapter.read(path, TimestampedRecord)
    assert loaded == record


def test_file_roundtrip_optional_datetime_present(tmp_path, adapter):
    """An Optional[datetime] with a value survives a file round-trip."""
    path = tmp_path / "opt.parquet"
    dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    record = OptionalTimestampRecord(label="opt", created_at=dt)
    adapter.write(record, path)
    loaded = adapter.read(path, OptionalTimestampRecord)
    assert loaded == record


def test_file_roundtrip_optional_datetime_none(tmp_path, adapter):
    """An Optional[datetime] set to None survives a file round-trip."""
    path = tmp_path / "opt_none.parquet"
    record = OptionalTimestampRecord(label="opt", created_at=None)
    adapter.write(record, path)
    loaded = adapter.read(path, OptionalTimestampRecord)
    assert loaded == record


def test_read_without_data_type_returns_dict(tmp_path, adapter):
    """Reading without a data_type returns a raw dict."""
    path = tmp_path / "raw.parquet"
    adapter.write(SampleRecord(name="x", value=0.0), path)
    loaded = adapter.read(path, data_type=None)
    assert isinstance(loaded, dict)
    assert loaded["name"] == "x"
