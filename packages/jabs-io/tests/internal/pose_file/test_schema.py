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
