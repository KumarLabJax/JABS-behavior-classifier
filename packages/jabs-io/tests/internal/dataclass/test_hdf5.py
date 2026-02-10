"""Tests for DataclassHDF5Adapter."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from jabs.io.internal.dataclass.hdf5 import DataclassHDF5Adapter
from tests.conftest import (
    NestedRecord,
    NumpyRecord,
    SampleRecord,
)


@pytest.fixture
def adapter():
    """Return a DataclassHDF5Adapter instance."""
    return DataclassHDF5Adapter()


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_type, expected",
    [
        (SampleRecord, True),
        (NumpyRecord, True),
        (dict, False),
        (int, False),
        (str, False),
        (list, False),
    ],
    ids=["dataclass", "dataclass-numpy", "dict", "int", "str", "list"],
)
def test_can_handle(data_type, expected):
    """can_handle returns True only for dataclass types."""
    assert DataclassHDF5Adapter.can_handle(data_type) is expected


# ---------------------------------------------------------------------------
# File I/O round-trips
# ---------------------------------------------------------------------------


def test_file_roundtrip_simple(tmp_path, adapter):
    """A simple scalar-only dataclass survives a file round-trip."""
    path = tmp_path / "simple.h5"
    record = SampleRecord(name="test", value=42.0)
    adapter.write(record, path)
    loaded = adapter.read(path, SampleRecord)
    assert loaded == record


def test_file_roundtrip_numpy(tmp_path, adapter):
    """A dataclass with numpy array fields survives a file round-trip."""
    path = tmp_path / "numpy.h5"
    record = NumpyRecord(
        name="arr",
        values=np.array([1.0, 2.0, 3.0]),
        labels=np.array([0, 1, 0]),
    )
    adapter.write(record, path)
    loaded = adapter.read(path, NumpyRecord)
    assert loaded.name == record.name
    np.testing.assert_array_equal(loaded.values, record.values)
    np.testing.assert_array_equal(loaded.labels, record.labels)


def test_file_roundtrip_nested_dataclass(tmp_path, adapter):
    """A dataclass containing another dataclass survives a file round-trip."""
    path = tmp_path / "nested.h5"
    record = NestedRecord(
        outer_name="outer",
        inner=SampleRecord(name="inner", value=9.0),
    )
    adapter.write(record, path)
    loaded = adapter.read(path, NestedRecord)
    assert loaded.outer_name == record.outer_name
    assert loaded.inner == record.inner


def test_file_roundtrip_optional_none(tmp_path, adapter):
    """Optional fields set to None are omitted and restored via defaults."""

    @dataclass
    class OptionalDC:
        label: str
        data: np.ndarray | None = None

    path = tmp_path / "opt_none.h5"
    record = OptionalDC(label="no-data")
    adapter.write(record, path)
    loaded = adapter.read(path, OptionalDC)
    assert loaded.label == "no-data"
    assert loaded.data is None


def test_file_roundtrip_optional_present(tmp_path, adapter):
    """Optional fields with values survive a file round-trip."""

    @dataclass
    class OptionalDC:
        label: str
        data: np.ndarray | None = None

    path = tmp_path / "opt_present.h5"
    arr = np.array([10, 20, 30])
    record = OptionalDC(label="has-data", data=arr)
    adapter.write(record, path)
    loaded = adapter.read(path, OptionalDC)
    assert loaded.label == "has-data"
    np.testing.assert_array_equal(loaded.data, arr)


def test_file_roundtrip_list_of_strings(tmp_path, adapter):
    """A list[str] field survives a file round-trip."""

    @dataclass
    class StringListDC:
        names: list[str]

    path = tmp_path / "strings.h5"
    record = StringListDC(names=["alpha", "beta", "gamma"])
    adapter.write(record, path)
    loaded = adapter.read(path, StringListDC)
    assert loaded.names == record.names


def test_file_roundtrip_dict(tmp_path, adapter):
    """A dict field is stored as a JSON attribute and round-trips."""

    @dataclass
    class DictDC:
        meta: dict[str, Any]

    path = tmp_path / "dict.h5"
    record = DictDC(meta={"key": "value", "num": 42})
    adapter.write(record, path)
    loaded = adapter.read(path, DictDC)
    assert loaded.meta == record.meta


def test_file_roundtrip_list_of_records(tmp_path, adapter):
    """A list of dataclasses survives a file round-trip via numbered subgroups."""
    path = tmp_path / "list.h5"
    records = [
        SampleRecord(name="a", value=1.0),
        SampleRecord(name="b", value=2.0),
        SampleRecord(name="c", value=3.0),
    ]
    adapter.write(records, path)
    loaded = adapter.read(path, SampleRecord)
    assert loaded == records


def test_file_roundtrip_behavior_prediction(tmp_path, adapter):
    """Full BehaviorPrediction round-trip as an integration test."""
    from jabs.core.types.prediction import BehaviorPrediction, ClassifierMetadata

    path = tmp_path / "pred.h5"
    pred = BehaviorPrediction(
        behavior="grooming",
        predicted_class=np.array([[0, 1, 0, 1]], dtype=np.int64),
        probabilities=np.array([[0.1, 0.9, 0.2, 0.8]]),
        classifier=ClassifierMetadata(
            classifier_file="model.ckpt",
            classifier_hash="abc123",
            app_version="1.0.0",
            prediction_date="2024-06-01T00:00:00Z",
        ),
        pose_file="poses.h5",
        pose_hash="def456",
        predicted_class_postprocessed=np.array([[0, 1, 1, 1]], dtype=np.int64),
        identity_to_track=np.array([[0, 0, 0, 0]], dtype=np.int64),
        external_identity_mapping=["mouse_0"],
        extra={"threshold": 0.5},
    )
    adapter.write(pred, path)
    loaded = adapter.read(path, BehaviorPrediction)

    assert loaded.behavior == pred.behavior
    assert loaded.pose_file == pred.pose_file
    assert loaded.pose_hash == pred.pose_hash
    assert loaded.classifier == pred.classifier
    np.testing.assert_array_equal(loaded.predicted_class, pred.predicted_class)
    np.testing.assert_array_equal(loaded.probabilities, pred.probabilities)
    np.testing.assert_array_equal(
        loaded.predicted_class_postprocessed, pred.predicted_class_postprocessed
    )
    np.testing.assert_array_equal(loaded.identity_to_track, pred.identity_to_track)
    assert loaded.external_identity_mapping == pred.external_identity_mapping
    assert loaded.extra == pred.extra


def test_read_without_data_type(tmp_path, adapter):
    """Reading without a data_type returns a raw dict."""
    path = tmp_path / "raw.h5"
    adapter.write(SampleRecord(name="x", value=0.0), path)
    loaded = adapter.read(path, data_type=None)
    assert isinstance(loaded, dict)
    assert loaded["name"] == "x"
