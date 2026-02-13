"""Tests for JSONAdapter base class."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from jabs.io.base import JSONAdapter
from tests.conftest import (
    SampleRecord,
)

# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


def test_encode_single(stub_json_adapter, sample_record):
    """Encoding a single object produces a JSON object string."""
    result = stub_json_adapter.encode(sample_record)
    parsed = json.loads(result)
    assert parsed == {"name": "test", "value": 42.0}


def test_encode_list(stub_json_adapter, sample_records):
    """Encoding a list of objects produces a JSON array string."""
    result = stub_json_adapter.encode(sample_records)
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    assert parsed[0]["name"] == "a"


def test_encode_empty_list(stub_json_adapter):
    """Encoding an empty list produces an empty JSON array."""
    result = stub_json_adapter.encode([])
    assert json.loads(result) == []


def test_decode_single(stub_json_adapter):
    """Decoding a JSON object string produces a single domain object."""
    raw = json.dumps({"name": "x", "value": 9.0})
    result = stub_json_adapter.decode(raw)
    assert result == SampleRecord(name="x", value=9.0)


def test_decode_list(stub_json_adapter):
    """Decoding a JSON array string produces a list of domain objects."""
    raw = json.dumps([{"name": "a", "value": 1.0}, {"name": "b", "value": 2.0}])
    result = stub_json_adapter.decode(raw)
    assert isinstance(result, list)
    assert len(result) == 2


def test_decode_empty_list(stub_json_adapter):
    """Decoding an empty JSON array produces an empty list."""
    raw = json.dumps([])
    result = stub_json_adapter.decode(raw)
    assert result == []


# ---------------------------------------------------------------------------
# _json_default serialization
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
def test_json_default_numpy(input_val, expected):
    """_json_default converts numpy types to native Python equivalents."""
    result = JSONAdapter._json_default(input_val)
    assert result == expected


def test_json_default_datetime():
    """_json_default converts aware datetimes to UTC ISO strings."""
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    result = JSONAdapter._json_default(dt)
    assert result == "2024-01-15T12:00:00+00:00"


def test_json_default_naive_datetime():
    """Naive datetimes should still be serializable (converted to local then UTC)."""
    dt = datetime(2024, 6, 1, 8, 30, 0)
    result = JSONAdapter._json_default(dt)
    assert isinstance(result, str)
    assert "2024-06-01" in result


def test_json_default_dataclass():
    """_json_default converts dataclass instances to dicts via asdict."""

    @dataclass
    class Inner:
        x: int

    result = JSONAdapter._json_default(Inner(x=5))
    assert result == {"x": 5}


def test_json_default_unsupported_type():
    """_json_default raises TypeError for types it cannot handle."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        JSONAdapter._json_default(object())


# ---------------------------------------------------------------------------
# Numpy values survive full encode round-trip
# ---------------------------------------------------------------------------


def test_numpy_values_in_encode():
    """Numpy values embedded in domain objects serialize via _json_default."""

    class NumpyJSONAdapter(JSONAdapter):
        @classmethod
        def can_handle(cls, data_type):
            return True

        def _encode_one(self, data) -> dict:
            return {"arr": data}

        def _decode_one(self, data, data_type=None):
            return data

    adapter = NumpyJSONAdapter()
    result = adapter.encode(np.array([10, 20]))
    parsed = json.loads(result)
    assert parsed == {"arr": [10, 20]}


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def test_write_creates_file(tmp_path, stub_json_adapter, sample_record):
    """write() creates a JSON file on disk with the encoded content."""
    path = tmp_path / "out.json"
    stub_json_adapter.write(sample_record, str(path))
    assert path.exists()
    content = json.loads(path.read_text())
    assert content["name"] == "test"


def test_read_returns_domain_object(tmp_path, stub_json_adapter):
    """read() loads a JSON file and returns a decoded domain object."""
    path = tmp_path / "in.json"
    path.write_text(json.dumps({"name": "rt", "value": 7.0}))
    result = stub_json_adapter.read(str(path))
    assert result == SampleRecord(name="rt", value=7.0)


def test_write_read_roundtrip_single(tmp_path, stub_json_adapter, sample_record):
    """A single object survives a write-then-read round-trip."""
    path = tmp_path / "roundtrip.json"
    stub_json_adapter.write(sample_record, str(path))
    loaded = stub_json_adapter.read(str(path))
    assert loaded == sample_record


def test_write_read_roundtrip_list(tmp_path, stub_json_adapter, sample_records):
    """A list of objects survives a write-then-read round-trip."""
    path = tmp_path / "roundtrip_list.json"
    stub_json_adapter.write(sample_records, str(path))
    loaded = stub_json_adapter.read(str(path))
    assert loaded == sample_records


def test_write_accepts_pathlib_path(tmp_path, stub_json_adapter, sample_record):
    """write() should accept both str and Path objects."""
    path = tmp_path / "pathlib.json"
    stub_json_adapter.write(sample_record, path)
    assert path.exists()


def test_read_accepts_pathlib_path(tmp_path, stub_json_adapter):
    """read() should accept both str and Path objects."""
    path = tmp_path / "pathlib.json"
    path.write_text(json.dumps({"name": "p", "value": 0.0}))
    result = stub_json_adapter.read(path)
    assert result == SampleRecord(name="p", value=0.0)
