"""Tests for ParquetAdapter base class."""

from unittest.mock import patch

import pyarrow as pa
import pytest

from jabs.io.base import ParquetAdapter
from tests.conftest import SampleRecord, StubParquetAdapter

# ---------------------------------------------------------------------------
# pyarrow import guard
# ---------------------------------------------------------------------------


def test_require_pyarrow_raises_when_missing():
    """ParquetAdapter should raise ImportError if pyarrow is not available."""
    with (
        patch("jabs.io.base.pa", None),
        patch("jabs.io.base.pq", None),
        pytest.raises(ImportError, match="pyarrow is required"),
    ):
        StubParquetAdapter()


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


def test_encode_single(stub_parquet_adapter, sample_record):
    """Encoding a single object produces a one-row Arrow table."""
    table = stub_parquet_adapter.encode(sample_record)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 1
    assert table.column("name")[0].as_py() == "test"


def test_encode_list(stub_parquet_adapter, sample_records):
    """Encoding a list of objects produces a multi-row Arrow table."""
    table = stub_parquet_adapter.encode(sample_records)
    assert table.num_rows == 3


def test_encode_empty_list(stub_parquet_adapter):
    """Encoding an empty list produces a zero-row Arrow table."""
    table = stub_parquet_adapter.encode([])
    assert table.num_rows == 0


def test_decode_single_row_returns_instance(stub_parquet_adapter):
    """Decoding a single-row table returns a single domain object."""
    table = pa.table({"name": ["x"], "value": [5.0]})
    result = stub_parquet_adapter.decode(table)
    assert result == SampleRecord(name="x", value=5.0)


def test_decode_multiple_rows_returns_list(stub_parquet_adapter):
    """Decoding a multi-row table returns a list of domain objects."""
    table = pa.table({"name": ["a", "b"], "value": [1.0, 2.0]})
    result = stub_parquet_adapter.decode(table)
    assert isinstance(result, list)
    assert len(result) == 2


def test_encode_decode_roundtrip(stub_parquet_adapter, sample_record):
    """A single object survives an encode-then-decode round-trip."""
    table = stub_parquet_adapter.encode(sample_record)
    result = stub_parquet_adapter.decode(table)
    assert result == sample_record


def test_encode_decode_roundtrip_list(stub_parquet_adapter, sample_records):
    """A list of objects survives an encode-then-decode round-trip."""
    table = stub_parquet_adapter.encode(sample_records)
    result = stub_parquet_adapter.decode(table)
    assert result == sample_records


# ---------------------------------------------------------------------------
# schema() hook
# ---------------------------------------------------------------------------


def test_default_schema_is_none(stub_parquet_adapter):
    """The default schema() returns None for automatic inference."""
    assert stub_parquet_adapter.schema() is None


def test_explicit_schema_applied():
    """When schema() returns a schema, encode uses it."""

    class TypedParquetAdapter(ParquetAdapter):
        @classmethod
        def can_handle(cls, data_type):
            return True

        def _to_record(self, data):
            return data

        def _from_record(self, record, data_type=None):
            return record

        def schema(self):
            return pa.schema([("x", pa.int64()), ("y", pa.float64())])

    adapter = TypedParquetAdapter()
    table = adapter.encode([{"x": 1, "y": 2.0}])
    assert table.schema.field("x").type == pa.int64()
    assert table.schema.field("y").type == pa.float64()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def test_write_creates_file(tmp_path, stub_parquet_adapter, sample_record):
    """write() creates a Parquet file on disk."""
    path = tmp_path / "out.parquet"
    stub_parquet_adapter.write(sample_record, str(path))
    assert path.exists()


def test_read_returns_domain_object(tmp_path, stub_parquet_adapter, sample_record):
    """read() loads a Parquet file and returns a decoded domain object."""
    path = tmp_path / "roundtrip.parquet"
    stub_parquet_adapter.write(sample_record, str(path))
    loaded = stub_parquet_adapter.read(str(path))
    assert loaded == sample_record


def test_write_read_roundtrip_list(tmp_path, stub_parquet_adapter, sample_records):
    """A list of objects survives a write-then-read round-trip."""
    path = tmp_path / "list.parquet"
    stub_parquet_adapter.write(sample_records, str(path))
    loaded = stub_parquet_adapter.read(str(path))
    assert loaded == sample_records


def test_write_accepts_pre_encoded_table(tmp_path, stub_parquet_adapter, sample_record):
    """write() should accept a pre-built pa.Table directly."""
    table = stub_parquet_adapter.encode(sample_record)
    path = tmp_path / "pre_encoded.parquet"
    stub_parquet_adapter.write(table, str(path))
    loaded = stub_parquet_adapter.read(str(path))
    assert loaded == sample_record


def test_write_accepts_pathlib_path(tmp_path, stub_parquet_adapter, sample_record):
    """write() should accept both str and Path objects."""
    path = tmp_path / "pathlib.parquet"
    stub_parquet_adapter.write(sample_record, path)
    assert path.exists()


def test_read_accepts_pathlib_path(tmp_path, stub_parquet_adapter, sample_record):
    """read() should accept both str and Path objects."""
    path = tmp_path / "pathlib.parquet"
    stub_parquet_adapter.write(sample_record, path)
    result = stub_parquet_adapter.read(path)
    assert result == sample_record
