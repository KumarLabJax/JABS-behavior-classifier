"""Tests for PoseNWBAdapter."""

import numpy as np
import pytest
from ndx_pose import PoseEstimation

from jabs.core.types import DynamicObjectData, PoseData
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
    with_subjects=False,
    with_dynamic_objects=False,
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
    # Static objects use 2-D arrays: shape (N, 2) where N is number of points.
    static_objects = {"lixit": np.array([[100.0, 200.0]])} if with_static_objects else {}
    metadata = {"source": "test", "model_version": "1.0"} if with_metadata else {}
    if edges is None:
        edges = [(0, 1), (1, 2)]

    names = external_ids or [f"subject_{i}" for i in range(num_identities)]
    subjects = {name: {"sex": "M", "genotype": "WT"} for name in names} if with_subjects else None

    if with_dynamic_objects:
        dyn_rng = np.random.default_rng(42)
        points_dyn = dyn_rng.random((5, 2, 1, 2)) * 100
        counts_dyn = np.array([2, 1, 0, 2, 1], dtype=np.int64)
        sample_indices_dyn = np.array([10, 25, 40, 60, 80], dtype=np.int64)
        dynamic_objects: dict[str, DynamicObjectData] = {
            "fecal_boli": DynamicObjectData(
                points=points_dyn,
                counts=counts_dyn,
                sample_indices=sample_indices_dyn,
            )
        }
    else:
        dynamic_objects = {}

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
        dynamic_objects=dynamic_objects,
        external_ids=external_ids,
        subjects=subjects,
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
    assert a.subjects == b.subjects
    assert a.metadata == b.metadata
    assert a.static_objects.keys() == b.static_objects.keys()
    for key in a.static_objects:
        np.testing.assert_allclose(a.static_objects[key], b.static_objects[key])
    assert set(a.dynamic_objects.keys()) == set(b.dynamic_objects.keys())
    for key in a.dynamic_objects:
        np.testing.assert_allclose(
            a.dynamic_objects[key].points, b.dynamic_objects[key].points, atol=1e-10
        )
        np.testing.assert_array_equal(a.dynamic_objects[key].counts, b.dynamic_objects[key].counts)
        np.testing.assert_array_equal(
            a.dynamic_objects[key].sample_indices, b.dynamic_objects[key].sample_indices
        )
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


def test_roundtrip_with_subjects(tmp_path, adapter):
    """subjects metadata survives roundtrip and is caught by _assert_pose_data_equal."""
    path = tmp_path / "pose_subjects.nwb"
    data = _make_pose_data(external_ids=["mouse_a", "mouse_b"], with_subjects=True)

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
    """Without external_ids, subject_N naming is used and roundtrips."""
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
    """Static objects survive NWB-native roundtrip via PoseEstimation containers."""
    path = tmp_path / "pose_static.nwb"
    data = _make_pose_data(with_static_objects=True)

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert "lixit" in loaded.static_objects
    np.testing.assert_allclose(loaded.static_objects["lixit"], data.static_objects["lixit"])


@pytest.mark.parametrize(
    ("obj_name", "points"),
    [
        ("corners", np.array([[10.0, 20.0], [300.0, 20.0], [10.0, 300.0], [300.0, 300.0]])),
        ("lixit", np.array([[62.0, 166.0]])),  # single-keypoint lixit (1, 2)
        (
            "lixit",
            np.array([[62.0, 166.0], [65.0, 160.0], [60.0, 172.0]]),
        ),  # 3-keypoint lixit (3, 2)
        (
            "food_hopper",
            np.array([[7.0, 291.0], [7.0, 528.0], [44.0, 296.0], [44.0, 518.0]]),
        ),  # (4, 2)
    ],
    ids=["corners-4pt", "lixit-1pt", "lixit-3pt", "food_hopper-4pt"],
)
def test_static_objects_multiple_points(tmp_path, adapter, obj_name, points):
    """Static objects with various point counts roundtrip correctly."""
    from jabs.core.types import PoseData

    path = tmp_path / "pose_static.nwb"
    base = _make_pose_data(with_static_objects=False)
    data = PoseData(
        points=base.points,
        point_mask=base.point_mask,
        identity_mask=base.identity_mask,
        body_parts=base.body_parts,
        edges=base.edges,
        fps=base.fps,
        cm_per_pixel=base.cm_per_pixel,
        static_objects={obj_name: points},
        metadata=base.metadata,
    )

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert obj_name in loaded.static_objects
    assert loaded.static_objects[obj_name].shape == points.shape
    np.testing.assert_allclose(loaded.static_objects[obj_name], points)


def test_static_objects_nwb_structure(tmp_path, adapter):
    """Static objects are stored as PoseEstimation containers in the behavior module."""
    from pynwb import NWBHDF5IO

    path = tmp_path / "pose_static_struct.nwb"
    data = _make_pose_data(with_static_objects=True)

    adapter.write(data, path)

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        behavior = nwb.processing["behavior"]

        # lixit should be a PoseEstimation container
        assert "lixit" in behavior.data_interfaces
        assert isinstance(behavior.data_interfaces["lixit"], PoseEstimation)

        # The Skeletons container should include a lixit skeleton
        skeletons_obj = behavior.data_interfaces["Skeletons"]
        skeleton_names = list(skeletons_obj.skeletons.keys())
        assert "lixit" in skeleton_names

        # The lixit PoseEstimation should have one series named lixit_0
        lixit_pe = behavior.data_interfaces["lixit"]
        assert "lixit_0" in lixit_pe.pose_estimation_series


def test_static_objects_not_in_jabs_metadata_json(tmp_path, adapter):
    """Static objects are no longer stored in the JSON scratch blob."""
    import json

    from pynwb import NWBHDF5IO

    path = tmp_path / "pose_no_json_static.nwb"
    data = _make_pose_data(with_static_objects=True)

    adapter.write(data, path)

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        jabs_meta = json.loads(str(nwb.scratch["jabs_metadata"].data))
        assert "static_objects" not in jabs_meta


def test_empty_static_objects(tmp_path, adapter):
    """Empty static_objects dict roundtrips."""
    path = tmp_path / "pose_no_static.nwb"
    data = _make_pose_data(with_static_objects=False)

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.static_objects == {}


def test_static_objects_1d_skipped_with_warning(tmp_path, adapter, caplog):
    """1-D static object arrays are skipped with a warning (not stored)."""
    import logging

    from jabs.core.types import PoseData

    base = _make_pose_data(with_static_objects=False)
    data = PoseData(
        points=base.points,
        point_mask=base.point_mask,
        identity_mask=base.identity_mask,
        body_parts=base.body_parts,
        edges=base.edges,
        fps=base.fps,
        cm_per_pixel=base.cm_per_pixel,
        static_objects={"bad_obj": np.array([1.0, 2.0])},  # 1-D, invalid
        metadata=base.metadata,
    )

    path = tmp_path / "pose_1d_static.nwb"
    with caplog.at_level(logging.WARNING):
        adapter.write(data, path)

    assert "bad_obj" in caplog.text
    loaded = adapter.read(path)
    assert "bad_obj" not in loaded.static_objects


def test_subjects_roundtrip(tmp_path, adapter):
    """Per-animal subjects metadata survives a single-file roundtrip."""
    import json

    from pynwb import NWBHDF5IO

    path = tmp_path / "pose_subjects.nwb"
    subjects = {
        "mouse_a": {"sex": "M", "genotype": "WT", "strain": "C57BL/6", "age": "P60"},
        "mouse_b": {"sex": "F", "genotype": "KO", "strain": "C57BL/6", "age": "P60"},
    }
    data = _make_pose_data(external_ids=["mouse_a", "mouse_b"])
    data = data.__class__(
        **{
            **{f: getattr(data, f) for f in data.__dataclass_fields__},
            "subjects": subjects,
        }
    )

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.subjects == subjects

    # Verify it's in the jabs_metadata JSON
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        meta = json.loads(str(nwb.scratch["jabs_metadata"].data))
        assert meta["subjects"] == subjects


def test_subjects_none_roundtrip(tmp_path, adapter):
    """subjects=None reads back as None."""
    path = tmp_path / "pose_no_subjects.nwb"
    data = _make_pose_data()

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.subjects is None


def test_subjects_per_identity_roundtrip(tmp_path, adapter):
    """Per-animal subjects metadata survives a per-identity file roundtrip."""
    path = tmp_path / "pose.nwb"
    subjects = {
        "mouse_a": {"sex": "M", "genotype": "WT"},
        "mouse_b": {"sex": "F", "genotype": "KO"},
    }
    data = _make_pose_data(external_ids=["mouse_a", "mouse_b"])
    data = data.__class__(
        **{
            **{f: getattr(data, f) for f in data.__dataclass_fields__},
            "subjects": subjects,
        }
    )

    adapter.write(data, path, per_identity_files=True)
    loaded = adapter.read(tmp_path / "pose_mouse_a.nwb")

    assert loaded.subjects == subjects


def test_bounding_boxes_per_identity_containers(tmp_path, adapter):
    """Bounding boxes are stored as one TimeSeries per identity, not a single combined array."""
    from pynwb import NWBHDF5IO

    path = tmp_path / "pose_bb_struct.nwb"
    data = _make_pose_data(
        num_identities=2,
        num_frames=10,
        external_ids=["mouse_a", "mouse_b"],
        with_bounding_boxes=True,
    )

    adapter.write(data, path)

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        behavior = nwb.processing["behavior"]

        assert "jabs_bounding_boxes_mouse_a" in behavior.data_interfaces
        assert "jabs_bounding_boxes_mouse_b" in behavior.data_interfaces
        # The old combined key must not be present
        assert "jabs_bounding_boxes" not in behavior.data_interfaces

        bb_a = behavior["jabs_bounding_boxes_mouse_a"]
        bb_b = behavior["jabs_bounding_boxes_mouse_b"]
        assert np.array(bb_a.data[:]).shape == (10, 2, 2)
        assert np.array(bb_b.data[:]).shape == (10, 2, 2)


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

    assert (tmp_path / "pose_subject_0.nwb").exists()
    assert (tmp_path / "pose_subject_1.nwb").exists()


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
    loaded = adapter.read(tmp_path / "pose_subject_0.nwb")

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


# ---------------------------------------------------------------------------
# Identity name sanitization
# ---------------------------------------------------------------------------


def test_sanitize_identity_name_alphanumeric():
    """Alphanumeric names and underscores/hyphens pass through unchanged."""
    assert PoseNWBAdapter._sanitize_identity_name("mouse_A-1") == "mouse_A-1"


def test_sanitize_identity_name_slash():
    """Forward slash is replaced with underscore."""
    assert PoseNWBAdapter._sanitize_identity_name("mouse/A") == "mouse_A"


def test_sanitize_identity_name_space():
    """Spaces within a name are replaced with underscores."""
    assert PoseNWBAdapter._sanitize_identity_name("mouse A") == "mouse_A"


def test_sanitize_identity_name_strips_whitespace():
    """Leading and trailing whitespace is stripped before substitution."""
    assert PoseNWBAdapter._sanitize_identity_name("  mouse  ") == "mouse"


def test_sanitize_identity_name_special_chars():
    """Dots, colons, and other special characters are replaced with underscores."""
    assert PoseNWBAdapter._sanitize_identity_name("mouse.A:1") == "mouse_A_1"


def test_sanitize_identity_name_empty_raises():
    """Empty string raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="empty"):
        PoseNWBAdapter._sanitize_identity_name("")


def test_sanitize_identity_name_whitespace_only_raises():
    """Whitespace-only string raises ValueError after stripping."""
    import pytest

    with pytest.raises(ValueError, match="empty"):
        PoseNWBAdapter._sanitize_identity_name("   ")


def test_write_sanitizes_external_ids(tmp_path, adapter):
    """External IDs with special characters are sanitized in the written NWB file."""
    path = tmp_path / "pose.nwb"
    data = _make_pose_data(external_ids=["mouse/A", "mouse/B"])

    adapter.write(data, path)
    loaded = adapter.read(path)

    # Data roundtrips correctly even though IDs were sanitized on disk
    assert loaded.external_ids == ["mouse/A", "mouse/B"]
    _assert_pose_data_equal(data, loaded)


def test_write_raises_on_collision_after_sanitization(tmp_path, adapter):
    """Write raises if two external IDs produce the same sanitized name."""
    import pytest

    path = tmp_path / "pose.nwb"
    # "mouse/A" and "mouse.A" both sanitize to "mouse_A"
    data = _make_pose_data(external_ids=["mouse/A", "mouse.A"])

    with pytest.raises(ValueError, match="not unique after sanitization"):
        adapter.write(data, path)


# ---------------------------------------------------------------------------
# Dynamic objects
# ---------------------------------------------------------------------------


def test_dynamic_objects_roundtrip(tmp_path, adapter):
    """Dynamic objects survive a single-file roundtrip."""
    path = tmp_path / "pose_dyn.nwb"
    data = _make_pose_data(with_dynamic_objects=True)

    adapter.write(data, path)
    loaded = adapter.read(path)

    _assert_pose_data_equal(data, loaded)


def test_dynamic_objects_per_identity_roundtrip(tmp_path, adapter):
    """Dynamic objects survive a per-identity file roundtrip."""
    path = tmp_path / "pose.nwb"
    data = _make_pose_data(external_ids=["mouse_a", "mouse_b"], with_dynamic_objects=True)

    adapter.write(data, path, per_identity_files=True)
    loaded = adapter.read(tmp_path / "pose_mouse_a.nwb")

    _assert_pose_data_equal(data, loaded)


def test_dynamic_objects_nwb_structure(tmp_path, adapter):
    """Dynamic objects are stored as PoseEstimation + Skeleton in the behavior module."""
    from pynwb import NWBHDF5IO

    path = tmp_path / "pose_dyn_struct.nwb"
    data = _make_pose_data(with_dynamic_objects=True)

    adapter.write(data, path)

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        behavior = nwb.processing["behavior"]

        # fecal_boli should be a PoseEstimation container
        assert "fecal_boli" in behavior.data_interfaces
        assert isinstance(behavior.data_interfaces["fecal_boli"], PoseEstimation)

        # The Skeletons container should include a fecal_boli skeleton
        skeletons_obj = behavior.data_interfaces["Skeletons"]
        assert "fecal_boli" in skeletons_obj.skeletons

        # Single-keypoint, 2 slots → series named fecal_boli_0 and fecal_boli_1
        fb_pe = behavior.data_interfaces["fecal_boli"]
        assert "fecal_boli_0" in fb_pe.pose_estimation_series
        assert "fecal_boli_1" in fb_pe.pose_estimation_series


def test_dynamic_objects_empty(tmp_path, adapter):
    """Empty dynamic_objects writes no extra containers and reads back as empty dict."""
    import json

    from pynwb import NWBHDF5IO

    path = tmp_path / "pose_no_dyn.nwb"
    data = _make_pose_data(with_dynamic_objects=False)

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert loaded.dynamic_objects == {}

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        meta = json.loads(str(nwb.scratch["jabs_metadata"].data))
        assert "dynamic_object_names" not in meta
        assert "dynamic_object_shapes" not in meta


def test_dynamic_objects_multi_keypoint(tmp_path, adapter):
    """Multi-keypoint dynamic objects use {name}_{slot}_{kp} node naming and roundtrip."""
    path = tmp_path / "pose_dyn_multi.nwb"
    rng = np.random.default_rng(7)
    # foo: 2 instances, 2 keypoints each → shape (4, 2, 2, 2)
    points_foo = rng.random((4, 2, 2, 2)) * 200
    counts_foo = np.array([2, 1, 2, 0], dtype=np.int64)
    sample_indices_foo = np.array([5, 15, 30, 50], dtype=np.int64)
    dyn_obj = DynamicObjectData(
        points=points_foo, counts=counts_foo, sample_indices=sample_indices_foo
    )
    base = _make_pose_data(with_dynamic_objects=False)
    data = PoseData(
        points=base.points,
        point_mask=base.point_mask,
        identity_mask=base.identity_mask,
        body_parts=base.body_parts,
        edges=base.edges,
        fps=base.fps,
        cm_per_pixel=base.cm_per_pixel,
        dynamic_objects={"foo": dyn_obj},
        metadata=base.metadata,
    )

    adapter.write(data, path)
    loaded = adapter.read(path)

    assert "foo" in loaded.dynamic_objects
    foo = loaded.dynamic_objects["foo"]
    np.testing.assert_allclose(foo.points, points_foo, atol=1e-10)
    np.testing.assert_array_equal(foo.counts, counts_foo)
    np.testing.assert_array_equal(foo.sample_indices, sample_indices_foo)

    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        foo_pe = nwb.processing["behavior"]["foo"]
        # 2 slots * 2 keypoints -> 4 series
        expected_names = {"foo_0_0", "foo_0_1", "foo_1_0", "foo_1_1"}
        assert set(foo_pe.pose_estimation_series.keys()) == expected_names
