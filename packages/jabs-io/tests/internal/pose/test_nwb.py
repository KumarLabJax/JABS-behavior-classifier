"""Tests for PoseNWBAdapter."""

import numpy as np
import pytest

from jabs.core.types import PoseData
from jabs.io.internal.pose.nwb import PoseNWBAdapter


@pytest.fixture
def adapter():
    """Return a PoseNWBAdapter instance."""
    return PoseNWBAdapter()


def _make_pose_data(
    num_identities=2,
    num_frames=10,
    num_keypoints=3,
    fps=30,
    cm_per_pixel=0.05,
    external_ids=None,
    with_bounding_boxes=False,
    with_static_objects=True,
    with_metadata=True,
    edges=None,
):
    rng = np.random.default_rng(42)
    points = rng.random((num_identities, num_frames, num_keypoints, 2)) * 100
    point_mask = rng.random((num_identities, num_frames, num_keypoints)) > 0.2
    identity_mask = rng.random((num_identities, num_frames)) > 0.1
    body_parts = [f"part_{i}" for i in range(num_keypoints)]
    bounding_boxes = (
        rng.random((num_identities, num_frames, 2, 2)) * 100 if with_bounding_boxes else None
    )
    static_objects = {"lixit": np.array([100.0, 200.0])} if with_static_objects else {}
    metadata = {"source": "test", "model_version": "1.0"} if with_metadata else {}
    if edges is None:
        edges = [(0, 1), (1, 2)]

    return PoseData(
        points=points,
        point_mask=point_mask,
        identity_mask=identity_mask,
        body_parts=body_parts,
        edges=edges,
        fps=fps,
        cm_per_pixel=cm_per_pixel,
        bounding_boxes=bounding_boxes,
        static_objects=static_objects,
        external_ids=external_ids,
        metadata=metadata,
    )


def _assert_pose_data_equal(a: PoseData, b: PoseData):
    """Assert two PoseData objects are equal within tolerance."""
    np.testing.assert_allclose(a.points, b.points, atol=1e-10)
    np.testing.assert_array_equal(a.point_mask, b.point_mask)
    np.testing.assert_array_equal(a.identity_mask, b.identity_mask)
    assert a.body_parts == b.body_parts
    assert a.edges == b.edges
    assert a.fps == b.fps
    assert a.cm_per_pixel == b.cm_per_pixel
    assert a.external_ids == b.external_ids
    assert a.metadata == b.metadata
    assert a.static_objects.keys() == b.static_objects.keys()
    for key in a.static_objects:
        np.testing.assert_allclose(a.static_objects[key], b.static_objects[key])
    if a.bounding_boxes is None:
        assert b.bounding_boxes is None
    else:
        np.testing.assert_allclose(a.bounding_boxes, b.bounding_boxes, atol=1e-10)


# ---------------------------------------------------------------------------
# Single-file roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_single_file(tmp_path, adapter):
    """Write multi-identity PoseData to one file, read back, assert equality."""
    path = tmp_path / "pose.nwb"
    data = _make_pose_data()

    adapter.write(data, path)
    loaded = adapter.read(path)

    _assert_pose_data_equal(data, loaded)


def test_roundtrip_with_bounding_boxes(tmp_path, adapter):
    """Bounding boxes survive roundtrip."""
    path = tmp_path / "pose_bb.nwb"
    data = _make_pose_data(with_bounding_boxes=True)

    adapter.write(data, path)
    loaded = adapter.read(path)

    _assert_pose_data_equal(data, loaded)


def test_roundtrip_bounding_boxes_none(tmp_path, adapter):
    """None bounding boxes read back as None."""
    path = tmp_path / "pose_no_bb.nwb"
    data = _make_pose_data(with_bounding_boxes=False)

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.bounding_boxes is None


def test_roundtrip_single_identity(tmp_path, adapter):
    """Single-identity PoseData roundtrips correctly."""
    path = tmp_path / "pose_single.nwb"
    data = _make_pose_data(num_identities=1)

    adapter.write(data, path)
    loaded = adapter.read(path)

    _assert_pose_data_equal(data, loaded)


def test_roundtrip_many_identities(tmp_path, adapter):
    """Five identities roundtrip correctly."""
    path = tmp_path / "pose_many.nwb"
    data = _make_pose_data(num_identities=5)

    adapter.write(data, path)
    loaded = adapter.read(path)

    _assert_pose_data_equal(data, loaded)


def test_external_ids_used_in_naming(tmp_path, adapter):
    """External IDs are used as PoseEstimation container names."""
    path = tmp_path / "pose_ext.nwb"
    data = _make_pose_data(external_ids=["mouse_a", "mouse_b"])

    adapter.write(data, path)
    loaded = adapter.read(path)

    _assert_pose_data_equal(data, loaded)


def test_external_ids_none(tmp_path, adapter):
    """Without external_ids, identity_N naming is used and roundtrips."""
    path = tmp_path / "pose_no_ext.nwb"
    data = _make_pose_data(external_ids=None)

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.external_ids is None
    _assert_pose_data_equal(data, loaded)


def test_cm_per_pixel_none(tmp_path, adapter):
    """None cm_per_pixel reads back as None."""
    path = tmp_path / "pose_no_cm.nwb"
    data = _make_pose_data(cm_per_pixel=None)

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.cm_per_pixel is None


def test_static_objects_roundtrip(tmp_path, adapter):
    """Static objects survive JSON serialization roundtrip."""
    path = tmp_path / "pose_static.nwb"
    data = _make_pose_data(with_static_objects=True)

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert "lixit" in loaded.static_objects
    np.testing.assert_allclose(loaded.static_objects["lixit"], data.static_objects["lixit"])


def test_empty_static_objects(tmp_path, adapter):
    """Empty static_objects dict roundtrips."""
    path = tmp_path / "pose_no_static.nwb"
    data = _make_pose_data(with_static_objects=False)

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.static_objects == {}


def test_edges_roundtrip(tmp_path, adapter):
    """Skeleton edges survive roundtrip."""
    path = tmp_path / "pose_edges.nwb"
    data = _make_pose_data(edges=[(0, 1), (1, 2)])

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.edges == [(0, 1), (1, 2)]


def test_empty_edges(tmp_path, adapter):
    """Empty edges list roundtrips."""
    path = tmp_path / "pose_no_edges.nwb"
    data = _make_pose_data(edges=[])

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.edges == []


# ---------------------------------------------------------------------------
# Per-identity files
# ---------------------------------------------------------------------------


def test_per_identity_files_creates_multiple_files(tmp_path, adapter):
    """per_identity_files=True creates one file per identity."""
    path = tmp_path / "pose.nwb"
    data = _make_pose_data(external_ids=["mouse_a", "mouse_b"])

    adapter.write(data, path, per_identity_files=True)

    assert (tmp_path / "pose_mouse_a.nwb").exists()
    assert (tmp_path / "pose_mouse_b.nwb").exists()
    assert not path.exists()  # base path should NOT be created


def test_per_identity_files_index_naming(tmp_path, adapter):
    """Without external_ids, per-identity files use index naming."""
    path = tmp_path / "pose.nwb"
    data = _make_pose_data(external_ids=None)

    adapter.write(data, path, per_identity_files=True)

    assert (tmp_path / "pose_identity_0.nwb").exists()
    assert (tmp_path / "pose_identity_1.nwb").exists()


def test_per_identity_roundtrip_single_file(tmp_path, adapter):
    """Reading any per-identity file auto-merges siblings and returns all identities."""
    path = tmp_path / "pose.nwb"
    data = _make_pose_data(external_ids=["mouse_a", "mouse_b"])

    adapter.write(data, path, per_identity_files=True)

    loaded = adapter.read(tmp_path / "pose_mouse_a.nwb")

    assert loaded.points.shape[0] == 2


def test_per_identity_auto_merge(tmp_path, adapter):
    """Reading any per-identity file auto-detects and merges all siblings."""
    path = tmp_path / "pose.nwb"
    data = _make_pose_data(
        external_ids=["mouse_a", "mouse_b"],
        with_bounding_boxes=True,
    )

    adapter.write(data, path, per_identity_files=True)

    # Read from either file, should get the full merged PoseData
    loaded = adapter.read(tmp_path / "pose_mouse_a.nwb")

    _assert_pose_data_equal(data, loaded)


def test_per_identity_auto_merge_index_naming(tmp_path, adapter):
    """Auto-merge works with index-based naming."""
    path = tmp_path / "pose.nwb"
    data = _make_pose_data(external_ids=None)

    adapter.write(data, path, per_identity_files=True)
    loaded = adapter.read(tmp_path / "pose_identity_0.nwb")

    _assert_pose_data_equal(data, loaded)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_adapter_resolves_from_registry():
    """The adapter resolves through the global registry."""
    from jabs.core.enums import StorageFormat
    from jabs.io.registry import get_adapter

    adapter = get_adapter(StorageFormat.NWB, PoseData)
    assert adapter is not None
    assert isinstance(adapter, PoseNWBAdapter)


def test_can_handle():
    """can_handle returns True only for PoseData."""
    assert PoseNWBAdapter.can_handle(PoseData) is True
    assert PoseNWBAdapter.can_handle(dict) is False
