"""Tests for the embedding descriptor, its fingerprint and embedding component ids."""

import dataclasses

import pytest

from jabs.io.internal.pose_file.embeddings import (
    EmbeddingDescriptor,
    EmbeddingModel,
    Projection,
    TemporalWindow,
    canonical_json,
    compute_fingerprint,
    parse_embedding_id,
)

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
