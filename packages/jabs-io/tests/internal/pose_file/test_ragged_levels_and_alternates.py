"""ADR 0002 review round 4: group-level ragged payloads, and alternate encodings.

The ragged encoding used to assume every payload holds leaf items, so a
per-contour ``external_flag`` naming the same offsets as the contour points
could never validate: one ``group_offsets`` cannot end at both the point count
and the contour count. The encoding now declares its ``level``. Alternates are
derived copies of a component in another encoding, validated the way a
component is.
"""

import json

import h5py
import numpy as np

from jabs.io.internal.pose_file import validate
from jabs.io.internal.pose_file.schema import validate_manifest, validate_provenance

GROUP = "/jabs/segmentation/ragged/contour_offsets"
INSTANCE = "/jabs/segmentation/ragged/instance_offsets"

# Two frames, one slot. Frame 0 owns two contours, frame 1 owns one, and each
# contour has two points: 3 groups and 6 leaves.
GROUP_OFFSETS = [0, 2, 4, 6]
INSTANCE_OFFSETS = [0, 2, 3]


def _ragged(level=None, group=GROUP, instance=INSTANCE):
    encoding = {"kind": "ragged", "group_offsets": group, "instance_offsets": instance}
    if level is not None:
        encoding["level"] = level
    return encoding


def _points_entry(points=6, level="leaf"):
    return {
        "id": "jabs.segmentation.contours",
        "path": "/jabs/segmentation/ragged/contours",
        "axes": ["point", "coord"],
        "dtype": "int32",
        "shape": [points, 2],
        "units": "pixel",
        "coord_order": "xy",
        "encoding": _ragged(level),
        "missing": {"policy": "none"},
    }


def _flag_entry(contours=3, level="group", group=GROUP):
    return {
        "id": "jabs.segmentation.external_flag",
        "path": "/jabs/segmentation/ragged/external_flag",
        "axes": ["contour"],
        "dtype": "bool",
        "shape": [contours],
        "encoding": _ragged(level, group=group),
        "missing": {"policy": "none"},
    }


def _write(path, components, datasets, history=()):
    manifest = {
        "format": "jabs.pose-file",
        "schema_revision": 1,
        "created": "2026-11-11T00:00:00Z",
        "dimensions": {"frame": 2, "slot": 1, "identity": 1},
        "video": {"frame_count": 2, "width": 8, "height": 8, "fps": 30.0},
        "components": components,
    }
    with h5py.File(path, "w") as h5:
        h5.attrs["jabs_format"] = "jabs.pose-file"
        h5.attrs["schema_revision"] = np.int32(1)
        text = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("manifest", data=json.dumps(manifest), dtype=text)
        h5.create_dataset(
            "provenance",
            data=json.dumps({"records": {}, "history": list(history)}),
            dtype=text,
        )
        for name, data in datasets.items():
            h5.create_dataset(name, data=data)
    return path


def _offsets(group=GROUP_OFFSETS, instance=INSTANCE_OFFSETS):
    return {
        GROUP: np.array(group, dtype=np.uint64),
        INSTANCE: np.array(instance, dtype=np.uint64),
    }


def _errors(path):
    return {f.check for f in validate(path) if f.severity == "error"}


# --- group-level ragged payloads -------------------------------------------


def test_per_contour_flags_share_the_point_offsets(tmp_path):
    """The case gbeane reported on #445: points at leaf level, flags at group level."""
    path = _write(
        tmp_path / "f.h5",
        [_points_entry(), _flag_entry()],
        {
            "/jabs/segmentation/ragged/contours": np.zeros((6, 2), dtype=np.int32),
            "/jabs/segmentation/ragged/external_flag": np.array([1, 0, 1], dtype=bool),
            **_offsets(),
        },
    )
    assert _errors(path) == set()


def test_level_defaults_to_leaf(tmp_path):
    """A ragged encoding written before `level` existed keeps its meaning."""
    path = _write(
        tmp_path / "f.h5",
        [_points_entry(level=None), _flag_entry()],
        {
            "/jabs/segmentation/ragged/contours": np.zeros((6, 2), dtype=np.int32),
            "/jabs/segmentation/ragged/external_flag": np.array([1, 0, 1], dtype=bool),
            **_offsets(),
        },
    )
    assert _errors(path) == set()


def test_one_point_contours_are_unambiguous_because_the_level_is_declared(tmp_path):
    """With one point per contour the two payloads are the same length."""
    path = _write(
        tmp_path / "f.h5",
        [_points_entry(points=3), _flag_entry()],
        {
            "/jabs/segmentation/ragged/contours": np.zeros((3, 2), dtype=np.int32),
            "/jabs/segmentation/ragged/external_flag": np.array([1, 1, 0], dtype=bool),
            **_offsets(group=[0, 1, 2, 3]),
        },
    )
    assert _errors(path) == set()


def test_a_group_level_payload_must_have_one_value_per_group(tmp_path):
    """A per-contour payload with a value too many decodes the last instance wrongly."""
    path = _write(
        tmp_path / "f.h5",
        [_points_entry(), _flag_entry(contours=4)],
        {
            "/jabs/segmentation/ragged/contours": np.zeros((6, 2), dtype=np.int32),
            "/jabs/segmentation/ragged/external_flag": np.zeros(4, dtype=bool),
            **_offsets(),
        },
    )
    assert "ragged_offsets" in _errors(path)


def test_a_group_level_payload_needs_a_leaf_level_partner(tmp_path):
    """Without one, nothing checks where group_offsets ends, and nothing is associated."""
    path = _write(
        tmp_path / "f.h5",
        [_flag_entry()],
        {
            "/jabs/segmentation/ragged/external_flag": np.array([1, 0, 1], dtype=bool),
            **_offsets(),
        },
    )
    assert "ragged_group_pairing" in _errors(path)


def test_a_leaf_level_flag_array_still_fails_as_before(tmp_path):
    """Declaring the per-contour flags leaf-level is the bug, and it is still caught."""
    path = _write(
        tmp_path / "f.h5",
        [_points_entry(), _flag_entry(level="leaf")],
        {
            "/jabs/segmentation/ragged/contours": np.zeros((6, 2), dtype=np.int32),
            "/jabs/segmentation/ragged/external_flag": np.array([1, 0, 1], dtype=bool),
            **_offsets(),
        },
    )
    assert "ragged_offsets" in _errors(path)


# --- alternates -------------------------------------------------------------


def _dense_with_alternates(points_alternate=None, flag_alternate=None):
    contours = {
        "id": "jabs.segmentation.contours",
        "path": "/jabs/segmentation/contours",
        "axes": ["frame", "slot", "contour", "point", "coord"],
        "dtype": "int32",
        "shape": [2, 1, 2, 2, 2],
        "units": "pixel",
        "coord_order": "xy",
        "encoding": {"kind": "dense"},
        "missing": {"policy": "length", "length": "jabs.segmentation.contour_length"},
        "alternates": [
            points_alternate
            or {
                "name": "ragged",
                "path": "/jabs/segmentation/ragged/contours",
                "axes": ["point", "coord"],
                "dtype": "int32",
                "shape": [6, 2],
                "units": "pixel",
                "coord_order": "xy",
                "encoding": _ragged("leaf"),
                "missing": {"policy": "none"},
            }
        ],
    }
    length = {
        "id": "jabs.segmentation.contour_length",
        "path": "/jabs/segmentation/contour_length",
        "axes": ["frame", "slot", "contour"],
        "dtype": "uint32",
        "shape": [2, 1, 2],
        "encoding": {"kind": "dense"},
        "missing": {"policy": "none"},
    }
    flag = {
        "id": "jabs.segmentation.external_flag",
        "path": "/jabs/segmentation/external_flag",
        "axes": ["frame", "slot", "contour"],
        "dtype": "bool",
        "shape": [2, 1, 2],
        "encoding": {"kind": "dense"},
        "missing": {"policy": "none"},
        "alternates": [
            flag_alternate
            or {
                "name": "ragged",
                "path": "/jabs/segmentation/ragged/external_flag",
                "axes": ["contour"],
                "dtype": "bool",
                "shape": [3],
                "encoding": _ragged("group"),
                "missing": {"policy": "none"},
            }
        ],
    }
    datasets = {
        "/jabs/segmentation/contours": np.zeros((2, 1, 2, 2, 2), dtype=np.int32),
        "/jabs/segmentation/contour_length": np.array([[[2, 2]], [[2, 0]]], dtype=np.uint32),
        "/jabs/segmentation/external_flag": np.zeros((2, 1, 2), dtype=bool),
        "/jabs/segmentation/ragged/contours": np.zeros((6, 2), dtype=np.int32),
        "/jabs/segmentation/ragged/external_flag": np.array([1, 0, 1], dtype=bool),
        **_offsets(),
    }
    return [contours, length, flag], datasets


def test_a_dense_baseline_with_ragged_alternates_validates(tmp_path):
    """The shape the JABS Hub writes: dense baseline, ragged copy beside it."""
    components, datasets = _dense_with_alternates()
    path = _write(tmp_path / "f.h5", components, datasets)
    assert _errors(path) == set()


def test_an_alternate_payload_must_exist(tmp_path):
    """A declared alternate with no data would send a reader to nothing."""
    components, datasets = _dense_with_alternates()
    del datasets["/jabs/segmentation/ragged/contours"]
    path = _write(tmp_path / "f.h5", components, datasets)
    assert "alternate_path_exists" in _errors(path)


def test_an_alternate_payload_must_match_its_declared_shape(tmp_path):
    """An alternate is checked as a component is."""
    components, datasets = _dense_with_alternates()
    datasets["/jabs/segmentation/ragged/contours"] = np.zeros((5, 2), dtype=np.int32)
    path = _write(tmp_path / "f.h5", components, datasets)
    assert "dtype_shape_match" in _errors(path)


def test_an_alternates_offsets_are_checked_like_a_components(tmp_path):
    """Offsets that end short lose data whichever encoding declares them."""
    components, datasets = _dense_with_alternates()
    datasets[GROUP] = np.array([0, 2, 4, 5], dtype=np.uint64)
    path = _write(tmp_path / "f.h5", components, datasets)
    assert "ragged_offsets" in _errors(path)


def test_an_alternate_cannot_reuse_a_components_path(tmp_path):
    """Two declarations of one dataset would make dropping one destroy the other."""
    components, datasets = _dense_with_alternates()
    components[0]["alternates"][0]["path"] = "/jabs/segmentation/contour_length"
    path = _write(tmp_path / "f.h5", components, datasets)
    assert "alternate_unique" in _errors(path)


def test_alternate_names_are_unique_within_a_component(tmp_path):
    """A reader looks an alternate up by name, so a name must pick one."""
    components, datasets = _dense_with_alternates()
    twin = dict(components[0]["alternates"][0], path="/jabs/segmentation/ragged/twin")
    components[0]["alternates"].append(twin)
    datasets["/jabs/segmentation/ragged/twin"] = np.zeros((6, 2), dtype=np.int32)
    path = _write(tmp_path / "f.h5", components, datasets)
    assert "alternate_unique" in _errors(path)


# --- schema -----------------------------------------------------------------


def test_an_alternate_cannot_carry_its_own_id(tmp_path):
    """It shares the component's id; a second one would be a second identity."""
    components, _ = _dense_with_alternates()
    components[0]["alternates"][0]["id"] = "jabs.segmentation.other"
    manifest = {
        "format": "jabs.pose-file",
        "schema_revision": 1,
        "dimensions": {"frame": 2, "slot": 1, "identity": 1},
        "video": {"frame_count": 2, "width": 8, "height": 8, "fps": 30.0},
        "components": components,
    }
    assert validate_manifest(manifest) != []


def test_an_unknown_ragged_level_is_rejected():
    """The level is a closed choice: leaf or group."""
    manifest = {
        "format": "jabs.pose-file",
        "schema_revision": 1,
        "dimensions": {"frame": 2, "slot": 1, "identity": 1},
        "video": {"frame_count": 2, "width": 8, "height": 8, "fps": 30.0},
        "components": [_points_entry(level="middle")],
    }
    assert validate_manifest(manifest) != []


def test_an_encode_entry_must_name_what_it_added():
    """An encode entry is how a reader learns the alternates are derived."""
    entry = {
        "operation": "encode",
        "tool": "jabs-io",
        "version": "0.1.0",
        "time": "2026-11-11T00:00:00Z",
    }
    assert validate_provenance({"records": {}, "history": [entry]}) != []
    entry["synthesized"] = ["jabs.segmentation.contours:alternate:ragged"]
    assert validate_provenance({"records": {}, "history": [entry]}) == []
