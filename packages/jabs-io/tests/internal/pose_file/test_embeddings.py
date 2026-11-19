"""Tests for the embedding descriptor, its fingerprint and embedding component ids."""

import dataclasses
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file import read_manifest, read_pose_file, validate, write_pose_file
from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.embeddings import (
    EmbeddingDescriptor,
    EmbeddingModel,
    Projection,
    TemporalWindow,
    canonical_json,
    compute_fingerprint,
    is_embedding_id,
    parse_embedding_id,
)

from .conftest import build_pose_file_with_embeddings

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
    [("jabs.embedding.trunk", True), ("jabs.pose.points", False)],
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
    ],
)
def test_descriptor_rejects(overrides):
    """Invalid descriptors cannot be constructed."""
    with pytest.raises(ValueError):
        _descriptor(**overrides)


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
