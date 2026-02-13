"""Tests for PredictionHDF5Adapter."""

import h5py
import numpy as np
import pytest

from jabs.core.types.prediction import BehaviorPrediction, ClassifierMetadata
from jabs.io.internal.prediction.hdf5 import PredictionHDF5Adapter


@pytest.fixture
def adapter():
    """Fixture to create a PredictionHDF5Adapter."""
    return PredictionHDF5Adapter()


def _make_prediction(
    behavior="grooming",
    n_identities=2,
    n_frames=10,
    postprocessed=False,
    identity_to_track=False,
    external_identity_mapping=None,
):
    """Helper to create a BehaviorPrediction with controllable optional fields."""
    rng = np.random.default_rng(42)
    pred_class = rng.integers(0, 2, size=(n_identities, n_frames), dtype=np.int64)
    probs = rng.random((n_identities, n_frames))

    pp = pred_class.copy() if postprocessed else None
    itt = np.zeros((n_identities, n_frames), dtype=np.int64) if identity_to_track else None

    return BehaviorPrediction(
        behavior=behavior,
        predicted_class=pred_class,
        probabilities=probs,
        classifier=ClassifierMetadata(
            classifier_file="model.ckpt",
            classifier_hash="abc123",
            app_version="1.0.0",
            prediction_date="2024-06-01T00:00:00Z",
        ),
        pose_file="poses.h5",
        pose_hash="def456",
        predicted_class_postprocessed=pp,
        identity_to_track=itt,
        external_identity_mapping=external_identity_mapping,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_roundtrip(tmp_path, adapter):
    """Write a BehaviorPrediction, read it back, assert equality."""
    path = tmp_path / "pred.h5"
    pred = _make_prediction(postprocessed=True, identity_to_track=True)

    adapter.write(pred, path)
    loaded = adapter.read(path, behavior="grooming")

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


# ---------------------------------------------------------------------------
# Multi-behavior
# ---------------------------------------------------------------------------


def test_multi_behavior(tmp_path, adapter):
    """Write two behaviors to the same file, read each independently."""
    path = tmp_path / "multi.h5"
    pred1 = _make_prediction(behavior="grooming", postprocessed=True)
    pred2 = _make_prediction(behavior="running")

    adapter.write(pred1, path)
    adapter.write(pred2, path)

    loaded1 = adapter.read(path, behavior="grooming")
    loaded2 = adapter.read(path, behavior="running")

    assert loaded1.behavior == "grooming"
    assert loaded2.behavior == "running"
    np.testing.assert_array_equal(loaded1.predicted_class, pred1.predicted_class)
    np.testing.assert_array_equal(loaded2.predicted_class, pred2.predicted_class)


def test_multi_behavior_list_write(tmp_path, adapter):
    """Writing a list of predictions stores all in one file."""
    path = tmp_path / "multi_list.h5"
    preds = [
        _make_prediction(behavior="grooming"),
        _make_prediction(behavior="running"),
    ]
    adapter.write(preds, path)

    loaded1 = adapter.read(path, behavior="grooming")
    loaded2 = adapter.read(path, behavior="running")
    assert loaded1.behavior == "grooming"
    assert loaded2.behavior == "running"


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


def test_optional_fields_none(tmp_path, adapter):
    """Write with None postprocessed/identity_to_track, verify omitted on read."""
    path = tmp_path / "opt_none.h5"
    pred = _make_prediction(postprocessed=False, identity_to_track=False)

    adapter.write(pred, path)
    loaded = adapter.read(path, behavior="grooming")

    assert loaded.predicted_class_postprocessed is None
    assert loaded.identity_to_track is None


def test_external_identity_mapping(tmp_path, adapter):
    """External identity mapping is written and read correctly."""
    path = tmp_path / "ext_map.h5"
    pred = _make_prediction(external_identity_mapping=["mouse_0", "mouse_1"])

    adapter.write(pred, path)
    loaded = adapter.read(path, behavior="grooming")

    assert loaded.external_identity_mapping == ["mouse_0", "mouse_1"]


def test_external_identity_mapping_none(tmp_path, adapter):
    """When no external identity mapping, it reads back as None."""
    path = tmp_path / "no_ext.h5"
    pred = _make_prediction(external_identity_mapping=None)

    adapter.write(pred, path)
    loaded = adapter.read(path, behavior="grooming")

    assert loaded.external_identity_mapping is None


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


def test_legacy_compat(tmp_path):
    """Create an HDF5 file with raw h5py matching legacy layout, verify read."""
    path = tmp_path / "legacy.h5"
    pred_class = np.array([[0, 1, 0]], dtype=np.int64)
    probs = np.array([[0.1, 0.9, 0.2]])

    with h5py.File(path, "w") as h5:
        h5.attrs["pose_file"] = "legacy_poses.h5"
        h5.attrs["pose_hash"] = "legacyhash"
        h5.attrs["version"] = 2

        pgroup = h5.create_group("predictions")
        pgroup.create_dataset(
            "external_identity_mapping",
            data=np.array(["id_0"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        bgroup = pgroup.create_group("grooming")
        bgroup.attrs["classifier_file"] = "old_model.ckpt"
        bgroup.attrs["classifier_hash"] = "oldhash"
        bgroup.attrs["app_version"] = "0.9.0"
        bgroup.attrs["prediction_date"] = "2023-01-01"
        bgroup.create_dataset("predicted_class", data=pred_class)
        bgroup.create_dataset("probabilities", data=probs)

    adapter = PredictionHDF5Adapter()
    loaded = adapter.read(path, behavior="grooming")

    assert loaded.behavior == "grooming"
    assert loaded.pose_file == "legacy_poses.h5"
    assert loaded.pose_hash == "legacyhash"
    assert loaded.classifier.classifier_file == "old_model.ckpt"
    assert loaded.classifier.app_version == "0.9.0"
    assert loaded.external_identity_mapping == ["id_0"]
    np.testing.assert_array_equal(loaded.predicted_class, pred_class)
    np.testing.assert_array_equal(loaded.probabilities, probs)
    assert loaded.predicted_class_postprocessed is None
    assert loaded.identity_to_track is None


# ---------------------------------------------------------------------------
# Read-all
# ---------------------------------------------------------------------------


def test_read_all(tmp_path, adapter):
    """Read without specifying behavior returns list of all behaviors."""
    path = tmp_path / "all.h5"
    preds = [
        _make_prediction(behavior="grooming", postprocessed=True),
        _make_prediction(behavior="running"),
    ]
    adapter.write(preds, path)

    loaded = adapter.read(path)
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    behaviors = {p.behavior for p in loaded}
    assert behaviors == {"grooming", "running"}


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_adapter_resolves():
    """The adapter resolves through the global registry."""
    from jabs.core.enums import StorageFormat
    from jabs.io.registry import get_adapter

    adapter = get_adapter(StorageFormat.HDF5, BehaviorPrediction)
    assert adapter is not None
    assert isinstance(adapter, PredictionHDF5Adapter)


def test_can_handle():
    """can_handle returns True only for BehaviorPrediction."""
    assert PredictionHDF5Adapter.can_handle(BehaviorPrediction) is True
    assert PredictionHDF5Adapter.can_handle(dict) is False
    assert PredictionHDF5Adapter.can_handle(ClassifierMetadata) is False
