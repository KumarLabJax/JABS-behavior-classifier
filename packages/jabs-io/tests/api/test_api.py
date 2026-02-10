"""Tests for the public load/save API."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jabs.core.enums import StorageFormat
from jabs.io.api import load, save
from tests.conftest import SampleRecord

# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


@patch("jabs.io.api.get_adapter")
@patch("jabs.io.api.get_storage_format")
def test_load_delegates_to_adapter_read(mock_get_format, mock_get_adapter):
    """load() resolves the adapter and delegates to its read method."""
    mock_get_format.return_value = StorageFormat.JSON
    mock_adapter = MagicMock()
    mock_adapter.read.return_value = SampleRecord(name="loaded", value=1.0)
    mock_get_adapter.return_value = mock_adapter

    result = load("/fake/path.json", SampleRecord)

    mock_get_format.assert_called_once_with(Path("/fake/path.json"))
    mock_get_adapter.assert_called_once_with(StorageFormat.JSON, SampleRecord)
    mock_adapter.read.assert_called_once_with(Path("/fake/path.json"))
    assert result == SampleRecord(name="loaded", value=1.0)


@patch("jabs.io.api.get_adapter")
@patch("jabs.io.api.get_storage_format")
def test_load_converts_str_path_to_pathlib(mock_get_format, mock_get_adapter):
    """load() converts a string path to a Path object before dispatching."""
    mock_get_format.return_value = StorageFormat.PARQUET
    mock_adapter = MagicMock()
    mock_get_adapter.return_value = mock_adapter

    load("relative/file.parquet", SampleRecord)

    called_path = mock_get_format.call_args[0][0]
    assert isinstance(called_path, Path)


@patch("jabs.io.api.get_adapter")
@patch("jabs.io.api.get_storage_format")
def test_load_accepts_pathlib_path(mock_get_format, mock_get_adapter):
    """load() accepts a pathlib.Path directly."""
    mock_get_format.return_value = StorageFormat.JSON
    mock_adapter = MagicMock()
    mock_get_adapter.return_value = mock_adapter

    load(Path("/some/file.json"), SampleRecord)

    called_path = mock_get_format.call_args[0][0]
    assert isinstance(called_path, Path)


@patch("jabs.io.api.get_adapter")
@patch("jabs.io.api.get_storage_format")
def test_load_passes_kwargs(mock_get_format, mock_get_adapter):
    """load() forwards extra keyword arguments to the adapter's read method."""
    mock_get_format.return_value = StorageFormat.JSON
    mock_adapter = MagicMock()
    mock_get_adapter.return_value = mock_adapter

    load("/fake/path.json", SampleRecord, columns=["name"])

    mock_adapter.read.assert_called_once_with(Path("/fake/path.json"), columns=["name"])


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


@patch("jabs.io.api.get_adapter")
@patch("jabs.io.api.get_domain_type")
@patch("jabs.io.api.get_storage_format")
def test_save_delegates_to_adapter_write(mock_get_format, mock_get_domain, mock_get_adapter):
    """save() resolves format and type, then delegates to the adapter's write method."""
    mock_get_format.return_value = StorageFormat.JSON
    mock_get_domain.return_value = SampleRecord
    mock_adapter = MagicMock()
    mock_get_adapter.return_value = mock_adapter

    data = SampleRecord(name="saved", value=2.0)
    save(data, "/fake/path.json")

    mock_get_format.assert_called_once_with(Path("/fake/path.json"))
    mock_get_domain.assert_called_once_with(data)
    mock_get_adapter.assert_called_once_with(StorageFormat.JSON, SampleRecord)
    mock_adapter.write.assert_called_once_with(data, Path("/fake/path.json"))


@patch("jabs.io.api.get_adapter")
@patch("jabs.io.api.get_domain_type")
@patch("jabs.io.api.get_storage_format")
def test_save_converts_str_path_to_pathlib(mock_get_format, mock_get_domain, mock_get_adapter):
    """save() converts a string path to a Path object before dispatching."""
    mock_get_format.return_value = StorageFormat.JSON
    mock_get_domain.return_value = SampleRecord
    mock_adapter = MagicMock()
    mock_get_adapter.return_value = mock_adapter

    save(SampleRecord(name="x", value=0.0), "output.json")

    called_path = mock_get_format.call_args[0][0]
    assert isinstance(called_path, Path)


@patch("jabs.io.api.get_adapter")
@patch("jabs.io.api.get_domain_type")
@patch("jabs.io.api.get_storage_format")
def test_save_passes_kwargs(mock_get_format, mock_get_domain, mock_get_adapter):
    """save() forwards extra keyword arguments to the adapter's write method."""
    mock_get_format.return_value = StorageFormat.PARQUET
    mock_get_domain.return_value = SampleRecord
    mock_adapter = MagicMock()
    mock_get_adapter.return_value = mock_adapter

    data = SampleRecord(name="x", value=0.0)
    save(data, "/fake/out.parquet", compression="snappy")

    mock_adapter.write.assert_called_once_with(
        data, Path("/fake/out.parquet"), compression="snappy"
    )


@patch("jabs.io.api.get_adapter")
@patch("jabs.io.api.get_domain_type")
@patch("jabs.io.api.get_storage_format")
def test_save_list_resolves_type_from_first_element(
    mock_get_format, mock_get_domain, mock_get_adapter
):
    """save() calls get_domain_type on the full list to resolve the element type."""
    mock_get_format.return_value = StorageFormat.JSON
    mock_get_domain.return_value = SampleRecord
    mock_adapter = MagicMock()
    mock_get_adapter.return_value = mock_adapter

    records = [SampleRecord(name="a", value=1.0), SampleRecord(name="b", value=2.0)]
    save(records, "/fake/path.json")

    # get_domain_type is called with the list, and returns the type of first element
    mock_get_domain.assert_called_once_with(records)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_load_unsupported_extension():
    """load() should raise ValueError for unknown file extensions."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load("/fake/path.xyz", SampleRecord)


def test_save_unsupported_extension():
    """save() should raise ValueError for unknown file extensions."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        save(SampleRecord(name="x", value=0.0), "/fake/path.xyz")
