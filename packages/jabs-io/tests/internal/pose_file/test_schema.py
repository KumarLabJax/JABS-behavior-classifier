"""Tests for pose file schema loading and validation."""

import jsonschema
import pytest

from jabs.io.internal.pose_file.schema import (
    FORMAT_ID,
    MANIFEST_SCHEMA,
    PROVENANCE_SCHEMA,
    SCHEMA_REVISION,
    validate_manifest,
    validate_provenance,
)

from .conftest import build_embedding_component


def test_schemas_are_valid_draft_2020_12():
    """Both shipped schemas must themselves be valid schemas."""
    jsonschema.Draft202012Validator.check_schema(MANIFEST_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(PROVENANCE_SCHEMA)


def test_format_constants():
    """The format id and revision are fixed by ADR 0002."""
    assert FORMAT_ID == "jabs.pose-file"
    assert SCHEMA_REVISION == 1


def test_minimal_manifest_validates(minimal_manifest):
    """A one-component manifest is valid."""
    assert validate_manifest(minimal_manifest) == []


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda m: m.pop("video"), id="no-video"),
        pytest.param(lambda m: m["components"][0].pop("missing"), id="no-missing"),
        pytest.param(lambda m: m["components"][0].pop("coord_order"), id="coord-without-order"),
        pytest.param(
            lambda m: m["components"][0].update(axes=["sample", "coord"]),
            id="sample-without-sparse",
        ),
        pytest.param(
            lambda m: m["components"][0].update(sparse={"index": "jabs.x.frame_index"}),
            id="sparse-without-sample",
        ),
        pytest.param(
            lambda m: m["components"][0].update(
                encoding={"kind": "ragged", "instance_offsets": "/x"}
            ),
            id="ragged-without-group-offsets",
        ),
        pytest.param(lambda m: m["components"][0].update(id="jabs"), id="single-segment-id"),
        pytest.param(lambda m: m["components"][0].update(id="Jabs.Pose"), id="uppercase-id"),
    ],
)
def test_manifest_rejects(minimal_manifest, mutate):
    """Each mutation makes the manifest describe a file no reader could decode."""
    mutate(minimal_manifest)
    assert validate_manifest(minimal_manifest) != []


def test_convert_history_requires_source_and_synthesized():
    """A convert entry must say where it came from and what it invented."""
    prov = {
        "records": {},
        "history": [
            {
                "operation": "convert",
                "tool": "jabs-io",
                "version": "0.1.0",
                "time": "2026-09-03T00:00:00Z",
            }
        ],
    }
    assert validate_provenance(prov) != []
    prov["history"][0]["source"] = {"format": "pose_est_v6"}
    prov["history"][0]["synthesized"] = ["skeletons.jabs.mouse12"]
    assert validate_provenance(prov) == []


def _embedding_entry() -> dict:
    """A schema-valid manifest entry for an instance embedding component."""
    return {
        "id": "jabs.embedding.trunk.instance",
        "path": "/jabs/embedding/trunk/instance",
        "axes": ["frame", "slot", "feature"],
        "dtype": "float16",
        "shape": [4, 1, 8],
        "encoding": {"kind": "dense"},
        "missing": {"policy": "nan"},
        "embedding": build_embedding_component().embedding.to_json(),
    }


def test_schema_accepts_an_embedding_entry(minimal_manifest):
    """An embedding entry with its descriptor validates."""
    minimal_manifest["components"].append(_embedding_entry())
    assert validate_manifest(minimal_manifest) == []


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.pop("embedding"), id="missing"),
        pytest.param(lambda d: d["embedding"].pop("normalization"), id="no-normalization"),
        pytest.param(lambda d: d["embedding"].pop("projection"), id="no-projection"),
        pytest.param(lambda d: d["embedding"].pop("fingerprint"), id="no-fingerprint"),
        pytest.param(lambda d: d["embedding"].update(extra=1), id="extra-key"),
        pytest.param(lambda d: d["embedding"].update(dim=True), id="boolean-dim"),
        pytest.param(lambda d: d["embedding"].update(dim=0), id="zero-dim"),
        pytest.param(lambda d: d["embedding"].update(kind="clip"), id="kind"),
        pytest.param(lambda d: d["embedding"].update(name="Trunk"), id="name"),
        pytest.param(lambda d: d["embedding"].update(tap=""), id="empty-tap"),
        pytest.param(
            lambda d: d["embedding"].update(fingerprint="sha256:" + "A" * 64), id="fingerprint"
        ),
        pytest.param(
            lambda d: d["embedding"]["model"].update(weights="md5:" + "0" * 64), id="weights"
        ),
        pytest.param(lambda d: d["embedding"]["model"].update(x=1), id="model-extra"),
        pytest.param(lambda d: d["embedding"]["window"].update(before=-1), id="window-negative"),
        pytest.param(lambda d: d["embedding"]["window"].update(x=1), id="window-extra"),
        pytest.param(
            lambda d: d["embedding"].update(projection={"kind": "pca", "input_dim": 768}),
            id="projection-incomplete",
        ),
        pytest.param(
            lambda d: d["embedding"].update(
                projection={"kind": "pca", "input_dim": 0, "basis": "b"}
            ),
            id="projection-input-dim",
        ),
    ],
)
def test_schema_requires_descriptor_on_embedding_ids(minimal_manifest, mutate):
    """An embedding id needs a complete, well-formed descriptor."""
    entry = _embedding_entry()
    mutate(entry)
    minimal_manifest["components"].append(entry)
    assert validate_manifest(minimal_manifest) != []


# The largest integer an IEEE 754 double holds exactly, so JCS writes it as Python does.
_MAX_SAFE_INTEGER = 2**53 - 1


def _set_projection(entry: dict, input_dim: int) -> None:
    """Give an embedding entry a projection with the given input width."""
    entry["embedding"]["projection"] = {"kind": "pca", "input_dim": input_dim, "basis": "b"}


@pytest.mark.parametrize(
    "set_integer",
    [
        pytest.param(lambda d, v: d["embedding"].update(dim=v), id="dim"),
        pytest.param(lambda d, v: d["embedding"]["window"].update(before=v), id="window-before"),
        pytest.param(lambda d, v: d["embedding"]["window"].update(after=v), id="window-after"),
        pytest.param(_set_projection, id="projection-input-dim"),
    ],
)
def test_schema_bounds_descriptor_integers_at_2_53(minimal_manifest, set_integer):
    """Descriptor integers stop at 2^53 - 1, so every one serializes the same under JCS."""
    entry = _embedding_entry()
    set_integer(entry, _MAX_SAFE_INTEGER)
    minimal_manifest["components"].append(entry)
    assert validate_manifest(minimal_manifest) == []
    set_integer(entry, _MAX_SAFE_INTEGER + 1)
    assert validate_manifest(minimal_manifest) != []


def test_schema_accepts_a_projection(minimal_manifest):
    """A complete projection object validates."""
    entry = _embedding_entry()
    entry["embedding"]["projection"] = {"kind": "pca", "input_dim": 768, "basis": "b"}
    minimal_manifest["components"].append(entry)
    assert validate_manifest(minimal_manifest) == []


def test_schema_forbids_descriptor_elsewhere(minimal_manifest):
    """A component outside jabs.embedding.* may not carry a descriptor."""
    minimal_manifest["components"][0]["embedding"] = _embedding_entry()["embedding"]
    assert validate_manifest(minimal_manifest) != []
