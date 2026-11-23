"""Tests for the embedding descriptor, its fingerprint and embedding component ids."""

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file import read_manifest, read_pose_file, validate, write_pose_file
from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.embeddings import (
    EmbeddingDescriptor,
    EmbeddingInfo,
    EmbeddingModel,
    Projection,
    TemporalWindow,
    canonical_json,
    compute_fingerprint,
    is_embedding_id,
    list_embeddings,
    paired_field_differences,
    parse_embedding_id,
)

from .conftest import build_embedding_component, build_pose_file_with_embeddings

DATA = Path(__file__).parents[2] / "data" / "pose-format"

_WEIGHTS = "sha256:" + "0" * 64


def _descriptor(**overrides):
    fields = {
        "name": "trunk",
        "kind": "instance",
        "dim": 4,
        "model": EmbeddingModel("vit-carry-spal", _WEIGHTS),
        "tap": "trunk.final",
        "pooling": "mask_mean",
        "window": TemporalWindow(9, 9),
    }
    fields.update(overrides)
    return EmbeddingDescriptor(**fields)


def test_fingerprint_test_vector():
    """The fingerprint of the reference descriptor matches the published test vector."""
    d = _descriptor()
    assert d.fingerprint == (
        "sha256:df9397378684bc85bac5df57e6137f5f69fb0dd3abdc99b33d0e2570c5c17696"
    )
    fingerprint_input = {
        k: v for k, v in d.to_json().items() if k not in {"name", "kind", "fingerprint"}
    }
    assert canonical_json(fingerprint_input) == (
        b'{"dim":4,"model":{"name":"vit-carry-spal","weights":"sha256:'
        + b"0" * 64
        + b'"},"normalization":"none","pooling":"mask_mean","projection":null,'
        b'"tap":"trunk.final","window":{"after":9,"before":9}}'
    )


def test_fingerprint_ignores_name_and_kind():
    """Name and kind are not part of the fingerprint."""
    assert _descriptor(name="other", kind="frame").fingerprint == _descriptor().fingerprint


@pytest.mark.parametrize(
    "overrides",
    [
        {"dim": 5},
        {"model": EmbeddingModel("vit-carry-spal", "sha256:" + "1" * 64)},
        {"tap": "trunk.block31"},
        {"pooling": "mean"},
        {"window": TemporalWindow(8, 9)},
        {"normalization": "l2"},
        {"projection": Projection("pca", 768, "sha256:" + "a" * 64)},
    ],
    ids=["dim", "weights", "tap", "pooling", "window", "normalization", "projection"],
)
def test_fingerprint_changes_with_each_paired_field(overrides):
    """Changing any fingerprinted field changes the fingerprint."""
    assert _descriptor(**overrides).fingerprint != _descriptor().fingerprint


def test_canonical_json_is_key_order_independent_and_utf8():
    """Canonical JSON sorts keys, drops whitespace and encodes non-ASCII as UTF-8."""
    expected = '{"a":"é","b":1}'.encode()
    assert canonical_json({"b": 1, "a": "é"}) == expected
    assert canonical_json({"a": "é", "b": 1}) == expected


@pytest.mark.parametrize(
    "descriptor",
    [
        _descriptor(),
        _descriptor(
            kind="frame",
            normalization="l2",
            projection=Projection("pca", 768, "sha256:" + "a" * 64),
        ),
    ],
    ids=["no-projection", "projection"],
)
def test_json_round_trip(descriptor):
    """A descriptor survives its manifest form, which carries the fingerprint."""
    data = descriptor.to_json()
    assert EmbeddingDescriptor.from_json(data) == descriptor
    assert data["fingerprint"] == descriptor.fingerprint
    assert compute_fingerprint(data) == descriptor.fingerprint


def test_to_json_records_absent_projection_as_null():
    """An absent projection is recorded explicitly as None."""
    assert _descriptor().to_json()["projection"] is None


def test_from_json_ignores_stored_fingerprint():
    """from_json does not trust or check the stored fingerprint."""
    data = _descriptor().to_json()
    data["fingerprint"] = "sha256:" + "f" * 64
    assert EmbeddingDescriptor.from_json(data).fingerprint == _descriptor().fingerprint


@pytest.mark.parametrize(
    ("component_id", "expected"),
    [
        ("jabs.embedding.trunk.instance", ("trunk", "instance")),
        ("jabs.embedding.blk-31.frame", ("blk-31", "frame")),
        ("jabs.embedding.trunk", None),
        ("jabs.embedding.trunk.extra", None),
        ("jabs.embedding.a.b.instance", None),
        ("jabs.pose.points", None),
    ],
)
def test_parse_embedding_id(component_id, expected):
    """Only exactly jabs.embedding.<name>.(instance|frame) parses."""
    assert parse_embedding_id(component_id) == expected


@pytest.mark.parametrize(
    ("component_id", "expected"),
    [
        ("jabs.embedding.trunk", True),
        ("jabs.pose.points", False),
        ("jabs.embeddings.trunk.instance", False),
    ],
)
def test_is_embedding_id(component_id, expected):
    """Any id under jabs.embedding. is in the namespace, well-formed or not."""
    assert is_embedding_id(component_id) is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "Trunk"},
        {"name": "trunk-"},
        {"name": "a.b"},
        {"name": ""},
        {"kind": "clip"},
        {"dim": 0},
        {"window": TemporalWindow(-1, 9)},
        {"window": TemporalWindow(9, -1)},
        {"model": EmbeddingModel("vit-carry-spal", "sha256:" + "0" * 63)},
        {"model": EmbeddingModel("vit-carry-spal", "sha256:" + "A" * 64)},
        {"model": EmbeddingModel("vit-carry-spal", "md5:" + "0" * 64)},
        {"model": EmbeddingModel("", _WEIGHTS)},
        {"tap": ""},
        {"pooling": ""},
        {"normalization": ""},
        {"projection": Projection("pca", 0, "sha256:" + "a" * 64)},
        {"projection": Projection("", 768, "sha256:" + "a" * 64)},
        {"projection": Projection("pca", 768, "")},
    ],
    ids=[
        "name-uppercase",
        "name-trailing-dash",
        "name-dotted",
        "name-empty",
        "kind",
        "dim",
        "window-before",
        "window-after",
        "weights-short",
        "weights-uppercase",
        "weights-algorithm",
        "model-name",
        "tap",
        "pooling",
        "normalization",
        "projection-input-dim",
        "projection-kind",
        "projection-basis",
    ],
)
def test_descriptor_rejects(overrides):
    """Invalid descriptors cannot be constructed."""
    with pytest.raises(ValueError):
        _descriptor(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dim": 4.0},
        {"dim": True},
        {"dim": "4"},
        {"window": TemporalWindow(9.0, 9)},  # type: ignore[arg-type]
        {"window": TemporalWindow(9, 9.0)},  # type: ignore[arg-type]
        {"window": TemporalWindow(False, 9)},
        {"window": TemporalWindow(9, np.bool_(True))},  # type: ignore[arg-type]
        {"projection": Projection("pca", 768.0, "b")},  # type: ignore[arg-type]
        {"projection": Projection("pca", True, "b")},
        {"name": 1},
        {"kind": None},
        {"model": EmbeddingModel(1, _WEIGHTS)},  # type: ignore[arg-type]
        {"model": EmbeddingModel("vit", 1)},  # type: ignore[arg-type]
        {"tap": 1},
        {"pooling": b"mask_mean"},
        {"normalization": None},
        {"projection": Projection(1, 768, "b")},  # type: ignore[arg-type]
        {"projection": Projection("pca", 768, None)},  # type: ignore[arg-type]
    ],
    ids=[
        "dim-float",
        "dim-bool",
        "dim-str",
        "window-before-float",
        "window-after-float",
        "window-before-bool",
        "window-after-numpy-bool",
        "projection-input-dim-float",
        "projection-input-dim-bool",
        "name",
        "kind",
        "model-name",
        "model-weights",
        "tap",
        "pooling",
        "normalization",
        "projection-kind",
        "projection-basis",
    ],
)
def test_descriptor_rejects_wrong_types(overrides):
    """Integers must be real ints and strings real strings, so the fingerprint is unambiguous."""
    with pytest.raises(ValueError):
        _descriptor(**overrides)


def test_descriptor_coerces_numpy_integers():
    """NumPy integers are stored as plain ints, so the descriptor serializes and hashes."""
    descriptor = _descriptor(
        dim=np.int64(4),
        window=TemporalWindow(np.int32(9), np.uint16(9)),  # type: ignore[arg-type]
        projection=Projection("pca", np.int64(768), "b"),  # type: ignore[arg-type]
    )
    integers = (
        descriptor.dim,
        descriptor.window.before,
        descriptor.window.after,
        descriptor.projection.input_dim,
    )
    assert all(type(value) is int for value in integers)
    assert descriptor == _descriptor(
        window=TemporalWindow(9, 9), projection=Projection("pca", 768, "b")
    )
    assert json.loads(canonical_json(descriptor.to_json()))["dim"] == 4


def test_descriptor_is_frozen():
    """Descriptors are immutable."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        _descriptor().dim = 8  # type: ignore[misc]


_INSTANCE = "jabs.embedding.trunk.instance"
_FRAME = "jabs.embedding.trunk.frame"


def _errors(path) -> list:
    """Return a file's error-severity findings."""
    return [f for f in validate(path) if f.severity == "error"]


def _entry(manifest: dict, component_id: str) -> dict:
    """Return one component's manifest entry."""
    return next(c for c in manifest["components"] if c["id"] == component_id)


def test_embedding_round_trip(tmp_path):
    """Embedding components and their descriptors survive a write and read."""
    pose_file = build_pose_file_with_embeddings()
    path = tmp_path / "embeddings.h5"
    write_pose_file(pose_file, path)

    assert _errors(path) == []
    read = read_pose_file(path)
    manifest = read_manifest(path)
    for component_id in (_INSTANCE, _FRAME):
        original = pose_file.component(component_id)
        restored = read.component(component_id)
        assert restored.embedding == original.embedding
        assert restored.data.dtype == np.float16
        np.testing.assert_array_equal(restored.data, original.data)
        assert (
            _entry(manifest, component_id)["embedding"]["fingerprint"]
            == original.embedding.fingerprint
        )
    assert read.component("jabs.pose.points").embedding is None
    # An unoccupied slot's instance row is all-NaN, and stays so on disk.
    original_rows = np.isnan(np.asarray(pose_file.component(_INSTANCE).data)).all(axis=-1)
    restored_rows = np.isnan(read.component(_INSTANCE).data).all(axis=-1)
    assert original_rows.any()
    np.testing.assert_array_equal(restored_rows, original_rows)


def _as_two_blocks(array: npt.NDArray[np.float16]) -> BlockSource:
    """Serve an array as two row blocks."""
    split = array.shape[0] // 2

    def blocks() -> Iterator[tuple[int, npt.NDArray[np.float16]]]:
        yield 0, array[:split]
        yield split, array[split:]

    return BlockSource(array.shape, array.dtype, blocks)


def test_embedding_from_block_source(tmp_path):
    """An embedding component streamed from two float16 blocks writes and reads back."""
    pose_file = build_pose_file_with_embeddings()
    original = pose_file.component(_INSTANCE)
    rows = np.asarray(original.data)
    streamed = dataclasses.replace(original, data=_as_two_blocks(rows))
    pose_file = dataclasses.replace(
        pose_file,
        components=tuple(streamed if c.id == _INSTANCE else c for c in pose_file.components),
    )
    path = tmp_path / "streamed.h5"
    write_pose_file(pose_file, path)

    assert _errors(path) == []
    restored = read_pose_file(path).component(_INSTANCE)
    assert restored.embedding == original.embedding
    assert restored.data.dtype == np.float16
    np.testing.assert_array_equal(restored.data, rows)


def test_paired_field_differences_names_each_differing_field():
    """Only PAIRED_FIELDS count; pooling, name and kind may differ between the kinds."""
    instance = _descriptor().to_json()
    frame = _descriptor(kind="frame", pooling="grid_mean").to_json()
    assert paired_field_differences(instance, frame) == []
    other = _descriptor(
        kind="frame", tap="trunk.other", window=TemporalWindow(4, 9), normalization="l2"
    ).to_json()
    assert paired_field_differences(instance, other) == ["tap", "window", "normalization"]


def test_a_pose_file_with_a_mismatched_pair_cannot_be_constructed():
    """The pairing rule is checked when the PoseFile is built, before any write."""
    sample = build_pose_file_with_embeddings()
    mismatched = build_embedding_component("trunk", "frame", tap="trunk.other")
    components = tuple(mismatched if c.id == _FRAME else c for c in sample.components)
    with pytest.raises(ValueError, match=r"jabs\.embedding\.trunk.*\['tap'\]"):
        dataclasses.replace(sample, components=components)


def _written_with_manifest(tmp_path, mutate) -> list:
    """Write the embedding sample, mutate its manifest, and return the file's errors."""
    path = tmp_path / "mutated.h5"
    write_pose_file(build_pose_file_with_embeddings(), path)
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5["manifest"][()])
        mutate(manifest)
        del h5["manifest"]
        h5.create_dataset(
            "manifest", data=json.dumps(manifest), dtype=h5py.string_dtype(encoding="utf-8")
        )
    return _errors(path)


@pytest.mark.parametrize(
    "field,value", [("name", "other"), ("kind", "frame")], ids=["name", "kind"]
)
def test_a_descriptor_naming_another_embedding_is_an_error(tmp_path, field, value):
    """The descriptor's name and kind must agree with the id; the fingerprint omits both."""
    errors = _written_with_manifest(
        tmp_path, lambda m: _entry(m, _INSTANCE)["embedding"].update({field: value})
    )
    assert {f.check for f in errors} == {"embedding_descriptor_matches"}


def _project(manifest: dict, instance_input_dim: float) -> None:
    """Give both trunk embeddings a projection, the instance one with the given width.

    768.0 equals 768 in Python, so the pair still agrees, and the frame
    descriptor is re-fingerprinted, so only the type rule fires.
    """
    for component_id, input_dim in ((_INSTANCE, instance_input_dim), (_FRAME, 768)):
        descriptor = _entry(manifest, component_id)["embedding"]
        descriptor["projection"] = {"kind": "pca", "input_dim": input_dim, "basis": "b"}
        descriptor["fingerprint"] = compute_fingerprint(descriptor)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda m: _entry(m, _INSTANCE)["embedding"].update(dim=8.0), id="dim"),
        pytest.param(
            lambda m: _entry(m, _INSTANCE)["embedding"]["window"].update(before=9.0),
            id="window-before",
        ),
        pytest.param(
            lambda m: _entry(m, _INSTANCE)["embedding"]["window"].update(after=9.0),
            id="window-after",
        ),
        pytest.param(lambda m: _project(m, 768.0), id="projection-input-dim"),
    ],
)
def test_a_float_where_an_integer_belongs_is_an_error(tmp_path, mutate):
    """JSON Schema counts 9.0 as an integer, but it hashes differently from 9.

    The stale fingerprint is not reported as well: the descriptor has no
    canonical form to hash until its types are right.
    """
    errors = _written_with_manifest(tmp_path, mutate)
    assert [f.check for f in errors] == ["embedding_descriptor_types"]
    assert _INSTANCE in errors[0].message


def test_an_embedding_without_the_nan_policy_is_an_error(tmp_path):
    """A missing vector is an all-NaN row, so no other missing policy fits an embedding."""
    errors = _written_with_manifest(
        tmp_path, lambda m: _entry(m, _INSTANCE).update(missing={"policy": "none"})
    )
    assert {f.check for f in errors} == {"embedding_storage"}


def test_an_empty_shape_is_reported_not_raised(tmp_path):
    """With no feature axis, dim is compared against nothing rather than indexed out of range."""
    errors = _written_with_manifest(tmp_path, lambda m: _entry(m, _INSTANCE).update(shape=[]))
    checks = {f.check for f in errors}
    assert "embedding_descriptor_matches" in checks
    assert "axes_arity" in checks


def test_list_embeddings_from_fixture():
    """Each embedding in a file is listed once, sorted by name then kind."""
    infos = list_embeddings(read_manifest(DATA / "valid-embedding.h5"))
    assert [(i.name, i.kind, i.dim) for i in infos] == [
        ("blk31", "instance", 4),
        ("trunk", "frame", 8),
        ("trunk", "instance", 8),
    ]
    assert [i.component_id for i in infos] == [
        "jabs.embedding.blk31.instance",
        "jabs.embedding.trunk.frame",
        "jabs.embedding.trunk.instance",
    ]
    assert all(isinstance(i, EmbeddingInfo) for i in infos)
    assert all(i.fingerprint.startswith("sha256:") for i in infos)
    # The two trunk components pool differently, so their vectors are not comparable.
    assert infos[1].fingerprint != infos[2].fingerprint


def test_list_embeddings_skips_malformed_embedding_ids():
    """Only ids of the exact form jabs.embedding.<name>.<kind> are embeddings to list."""
    descriptor = _descriptor().to_json()
    manifest = {
        "components": [
            {"id": "jabs.pose.points"},
            {"id": "jabs.embedding.Bad.instance", "embedding": descriptor},
            {"id": "jabs.embedding.trunk.instance", "embedding": descriptor},
        ]
    }
    assert list_embeddings(manifest) == [
        EmbeddingInfo(
            name="trunk",
            kind="instance",
            dim=4,
            fingerprint=descriptor["fingerprint"],
            component_id="jabs.embedding.trunk.instance",
        )
    ]
