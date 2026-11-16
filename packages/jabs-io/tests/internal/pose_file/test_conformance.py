"""The fixture corpus that any independent implementation can test against.

Regenerate the fixtures with
``uv run python packages/jabs-io/tests/data/pose-format/generate.py``.
"""

import ast
import importlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.reader import read_component, read_pose_file
from jabs.io.internal.pose_file.validate import validate

# The package re-exports the function under the module's name.
validate_module = importlib.import_module("jabs.io.internal.pose_file.validate")

DATA = Path(__file__).parents[2] / "data" / "pose-format"

VALID = [
    "valid-minimal.h5",
    "valid-full.h5",
    "valid-sparse.h5",
    "valid-ragged.h5",
    "valid-ragged-alternate.h5",
    "valid-rle.h5",
]
INVALID = {
    "invalid-alternate-duplicate.h5": "alternate_unique",
    "invalid-alternate-missing.h5": "alternate_path_exists",
    "invalid-attachment-missing.h5": "attachment_path_exists",
    "invalid-attachment-outside.h5": "attachment_namespace",
    "invalid-attachment-twice.h5": "attachment_unique",
    "invalid-axes-arity.h5": "axes_arity",
    "invalid-axis-length.h5": "axis_length",
    "invalid-coord-axis-length.h5": "coord_axis_length",
    "invalid-coord-declarations.h5": "manifest_schema",
    "invalid-dangling-mask.h5": "mask_reference",
    "invalid-duplicate-component-id.h5": "component_id_unique",
    "invalid-frame-axis-length.h5": "frame_axis_length",
    "invalid-frame-count.h5": "dimensions_match_video",
    "invalid-identity-exceeds-slot.h5": "identity_le_slot",
    "invalid-keypoint-axis.h5": "keypoint_axis_length",
    "invalid-length-exceeds-axis.h5": "mask_reference",
    "invalid-keypoint-without-skeleton.h5": "manifest_schema",
    "invalid-manifest-root-mismatch.h5": "manifest_matches_root",
    "invalid-manifest-schema.h5": "manifest_schema",
    "invalid-mask-axes.h5": "mask_reference",
    "invalid-mask-dtype.h5": "mask_reference",
    "invalid-missing-payload.h5": "component_path_exists",
    "invalid-namespace.h5": "namespace_well_formed",
    "invalid-nan-policy.h5": "manifest_schema",
    "invalid-not-a-pose-file.h5": "root_attrs",
    "invalid-not-hdf5.h5": "file_readable",
    "invalid-provenance-reference.h5": "provenance_reference",
    "invalid-provenance-schema.h5": "provenance_schema",
    "invalid-ragged-group-length.h5": "ragged_offsets",
    "invalid-ragged-group-terminal.h5": "ragged_offsets",
    "invalid-ragged-instance-count.h5": "ragged_offsets",
    "invalid-ragged-level.h5": "ragged_offsets",
    "invalid-ragged-order.h5": "ragged_offsets",
    "invalid-ragged-pairing.h5": "ragged_group_pairing",
    "invalid-ragged-signed-offsets.h5": "ragged_offsets",
    "invalid-rle-coverage.h5": "rle_coverage",
    "invalid-rle-dimensions.h5": "rle_needs_dimensions",
    "invalid-rle-offsets.h5": "rle_offsets",
    "invalid-rle-shape.h5": "manifest_schema",
    "invalid-sample-without-sparse.h5": "manifest_schema",
    "invalid-schema-revision-missing.h5": "root_attrs",
    "invalid-shape-mismatch.h5": "dtype_shape_match",
    "invalid-skeleton-edge.h5": "skeleton_edge_range",
    "invalid-skeleton-reference.h5": "skeleton_reference",
    "invalid-sparse-index-dtype.h5": "sparse_index_valid",
    "invalid-sparse-index-length.h5": "sparse_index_length",
    "invalid-sparse-index-order.h5": "sparse_index_valid",
    "invalid-sparse-index-range.h5": "sparse_index_valid",
    "invalid-timestamp.h5": "timestamp_parseable",
}

# Files that pass every structural check and fail only the opt-in deep check.
DEEP_INVALID = {
    "invalid-alternate-disagrees.h5": "alternate_decodes",
}


@pytest.mark.parametrize("name", VALID)
def test_valid_fixtures_have_no_errors(name):
    """Every valid fixture conforms."""
    errors = [f for f in validate(DATA / name) if f.severity == "error"]
    assert errors == [], f"{name} should be valid, got {[f.message for f in errors]}"


@pytest.mark.parametrize("name,expected_check", sorted(INVALID.items()))
def test_invalid_fixtures_report_their_check(name, expected_check):
    """Every invalid fixture fails the specific rule it was built to break, and no other.

    A fixture that broke two rules would tell a porter less about which check
    they got wrong.
    """
    errors = {f.check for f in validate(DATA / name) if f.severity == "error"}
    assert errors == {expected_check}, f"{name} should report only {expected_check}; got {errors}"


# Any schema violation reports manifest_schema, so these fixtures are pinned to
# the rule each was built to break by the schema error's message.
SCHEMA_MESSAGES = {
    "invalid-manifest-schema.h5": "'video' is a required property",
    "invalid-coord-declarations.h5": "'units' is a required property",
    "invalid-nan-policy.h5": "dtype: 'bool' is not one of ['float32', 'float64']",
    "invalid-keypoint-without-skeleton.h5": "'skeleton' is a required property",
    "invalid-sample-without-sparse.h5": "'sparse' is a required property",
    "invalid-rle-shape.h5": "axes: ['run'] was expected",
}


@pytest.mark.parametrize("name,message", sorted(SCHEMA_MESSAGES.items()))
def test_schema_fixtures_break_the_rule_they_name(name, message):
    """Each schema fixture fails for its own rule, not for some other schema error."""
    assert INVALID[name] == "manifest_schema"
    messages = [f.message for f in validate(DATA / name) if f.severity == "error"]
    assert len(messages) == 1 and message in messages[0], messages


def test_every_schema_fixture_is_pinned():
    """A new manifest_schema fixture needs its message here too."""
    schema_fixtures = {n for n, check in INVALID.items() if check == "manifest_schema"}
    assert schema_fixtures == set(SCHEMA_MESSAGES)


# Checks validate() cannot emit, because the manifest schema enforces the same
# rule first and a schema failure stops the structural pass. Their ADR rows are
# covered by fixtures that report manifest_schema.
SHADOWED_BY_SCHEMA = {"coord_declarations", "sample_sparse_pairing"}

# Check names validate.py builds rather than spells out.
COMPUTED_CHECKS = {
    "ragged_offsets",
    "rle_offsets",
    "manifest_schema",
    "provenance_schema",
    "timestamp_parseable",
}


def _error_checks_in_source() -> set[str]:
    """Every check name validate.py passes to an error Finding."""
    source = Path(validate_module.__file__).read_text()
    names = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Finding"):
            continue
        if len(node.args) < 2 or ast.unparse(node.args[0]) != "ERROR":
            continue
        check = node.args[1]
        branches = [check.body, check.orelse] if isinstance(check, ast.IfExp) else [check]
        names |= {b.value for b in branches if isinstance(b, ast.Constant)}
    return names | COMPUTED_CHECKS


# The functions in validate.py that pass an error Finding a check name held in
# a variable. The scan above cannot read those names, so COMPUTED_CHECKS lists
# them by hand. Pinning the functions means a new one fails the guard until its
# names are added there too.
COMPUTED_CHECK_SITES = {"_check_monotonic", "_check_offsets", "_record_schema_errors"}


def _computed_check_sites() -> set[str]:
    """The functions that build an error check name rather than spell it out."""
    tree = ast.parse(Path(validate_module.__file__).read_text())
    sites = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Finding"):
                continue
            if len(node.args) < 2 or ast.unparse(node.args[0]) != "ERROR":
                continue
            check = node.args[1]
            branches = [check.body, check.orelse] if isinstance(check, ast.IfExp) else [check]
            if not all(isinstance(b, ast.Constant) for b in branches):
                sites.add(function.name)
    return sites


def test_every_computed_check_name_is_listed():
    """A check name built at runtime is invisible to the scan, so its function is pinned."""
    assert _computed_check_sites() == COMPUTED_CHECK_SITES


def test_every_error_check_has_an_invalid_fixture():
    """ADR 0002: a set of deliberately invalid files, one per error check.

    A new check cannot ship without a fixture that fails it.
    """
    covered = set(INVALID.values()) | set(DEEP_INVALID.values())
    missing = _error_checks_in_source() - covered - SHADOWED_BY_SCHEMA
    assert missing == set(), f"error checks with no invalid fixture: {sorted(missing)}"


def test_the_schema_shadowed_checks_really_are_shadowed():
    """If one becomes reachable it needs a fixture, and this list must shrink."""
    assert _error_checks_in_source() >= SHADOWED_BY_SCHEMA
    reported = {f.check for name in INVALID for f in validate(DATA / name)}
    assert not (SHADOWED_BY_SCHEMA & reported)


# The valid fixtures that carry both points and confidence.
KEYPOINT_FIXTURES = ["valid-full.h5", "valid-sparse.h5", "valid-ragged-alternate.h5"]


@pytest.mark.parametrize("name", KEYPOINT_FIXTURES)
def test_nan_points_mean_no_coordinate_never_low_confidence(name):
    """ADR 0002: NaN in points only where confidence is 0; a low-confidence point keeps its coordinates.

    A fixture that blanks every point below the threshold teaches the very
    thing the rule forbids, to whoever ports the reader next.
    """
    pose_file = read_pose_file(DATA / name)
    points = pose_file.component("jabs.pose.points").data
    confidence = pose_file.component("jabs.pose.confidence").data
    point_valid = pose_file.component("jabs.pose.point_valid").data
    no_coordinate = np.isnan(points).any(axis=-1)

    assert not (no_coordinate & (confidence > 0)).any()
    # The fixture must exercise both sides of the rule, or the assertion above is vacuous.
    assert no_coordinate.any()
    assert (~point_valid & (confidence > 0) & ~no_coordinate).any()


@pytest.mark.parametrize("name", VALID)
def test_valid_fixtures_pass_the_deep_check(name):
    """Every alternate in a valid fixture decodes to its baseline."""
    errors = [f for f in validate(DATA / name, deep=True) if f.severity == "error"]
    assert errors == [], f"{name} should pass the deep check, got {[f.message for f in errors]}"


@pytest.mark.parametrize("name,expected_check", sorted(DEEP_INVALID.items()))
def test_deep_invalid_fixtures_fail_only_when_asked(name, expected_check):
    """Structurally valid; the deep check alone finds what is wrong."""
    assert [f for f in validate(DATA / name) if f.severity == "error"] == []
    findings = validate(DATA / name, deep=True)
    assert any(f.check == expected_check and f.severity == "error" for f in findings)


def test_fixture_corpus_is_complete():
    """The corpus on disk covers everything the suite names, and nothing more."""
    present = {p.name for p in DATA.glob("*.h5")}
    assert present == set(VALID) | set(INVALID) | set(DEEP_INVALID)


def test_full_fixture_carries_a_foreign_component():
    """Extensibility is demonstrated, not just asserted in prose."""
    pose_file = read_pose_file(DATA / "valid-full.h5")
    foreign = pose_file.component("org.example.lab.whisker_angle")
    assert foreign.axes == ("frame", "slot")
    assert foreign.path == "/org.example.lab/whisker_angle"
    assert foreign.units == "radian"


def test_a_foreign_component_can_be_windowed_without_understanding_it():
    """The point of declared axes: generic tooling can subset unknown data."""
    window = read_component(
        DATA / "valid-full.h5", "org.example.lab.whisker_angle", frames=slice(2, 5)
    )
    assert window.shape[0] == 3


def test_undeclared_attachment_is_only_a_warning():
    """An opaque payload nobody declared does not make the file invalid."""
    findings = validate(DATA / "valid-full.h5")
    assert [f for f in findings if f.severity == "error"] == []
    assert any(f.check == "attachment_undeclared" for f in findings)


def test_sparse_fixture_maps_samples_to_frames():
    """A sparse component's index says which frames it describes."""
    pose_file = read_pose_file(DATA / "valid-sparse.h5")
    index = pose_file.component("jabs.dynamic_objects.fecal_boli.frame_index")
    counts = pose_file.component("jabs.dynamic_objects.fecal_boli.counts")
    assert index.axes == ("sample",)
    assert index.units == "frame"
    np.testing.assert_array_equal(index.data, np.array([0, 3, 6], dtype=np.uint32))
    assert counts.sparse_index == index.id
    assert counts.data.shape == index.data.shape


def _decode_ragged(h5: h5py.File, contours: dict, flags: dict, frames: int, slots: int):
    """Decode a leaf/group ragged pair exactly as ADR 0002 spells it out.

    Leaf level: group g spans payload[group_offsets[g] : group_offsets[g + 1]],
    and instance (f, s) owns groups [instance_offsets[i], instance_offsets[i + 1]).
    Group level: instance (f, s) owns payload[instance_offsets[i] :
    instance_offsets[i + 1]] directly.
    """
    points = h5[contours["path"]][()]
    flag_values = h5[flags["path"]][()]
    group = h5[contours["encoding"]["group_offsets"]][()].astype(np.int64)
    instance = h5[contours["encoding"]["instance_offsets"]][()].astype(np.int64)
    decoded = {}
    for f in range(frames):
        for s in range(slots):
            i = f * slots + s
            first, last = instance[i], instance[i + 1]
            outlines = [points[group[g] : group[g + 1]].tolist() for g in range(first, last)]
            decoded[f, s] = (outlines, flag_values[first:last].tolist())
    return decoded


def _decode_dense(pose_file, frames: int, slots: int):
    data = pose_file.component("jabs.segmentation.contours").data
    count = pose_file.component("jabs.segmentation.contour_count").data
    length = pose_file.component("jabs.segmentation.contour_length").data
    flag = pose_file.component("jabs.segmentation.external_flag").data
    decoded = {}
    for f in range(frames):
        for s in range(slots):
            n = int(count[f, s])
            outlines = [data[f, s, c, : int(length[f, s, c])].tolist() for c in range(n)]
            decoded[f, s] = (outlines, flag[f, s, :n].tolist())
    return decoded


def test_the_ragged_alternate_decodes_to_its_dense_baseline():
    """The fixture an independent decoder checks itself against.

    It includes an instance whose every contour has one point, where the leaf
    and group payloads of that instance are the same length: a decoder that
    guesses the level instead of reading it gets that instance wrong.
    """
    path = DATA / "valid-ragged-alternate.h5"
    pose_file = read_pose_file(path)
    frames, slots = pose_file.dimensions["frame"], pose_file.dimensions["slot"]
    with h5py.File(path, "r") as h5:
        manifest = json.loads(h5["manifest"][()])
        by_id = {c["id"]: c for c in manifest["components"]}
        contours = by_id["jabs.segmentation.contours"]["alternates"][0]
        flags = by_id["jabs.segmentation.external_flag"]["alternates"][0]
        assert (contours["encoding"]["level"], flags["encoding"]["level"]) == ("leaf", "group")
        ragged = _decode_ragged(h5, contours, flags, frames, slots)
    assert ragged == _decode_dense(pose_file, frames, slots)
    one_point = [
        k for k, (outlines, _) in ragged.items() if outlines and all(len(o) == 1 for o in outlines)
    ]
    assert one_point, "the fixture must keep an instance of one-point contours"


def test_the_ragged_alternate_is_recorded_as_an_encode():
    """Adding an alternate writes a derived file, and its history says so."""
    pose_file = read_pose_file(DATA / "valid-ragged-alternate.h5")
    encode = [e for e in pose_file.provenance.history if e.operation == "encode"]
    assert len(encode) == 1
    assert "jabs.segmentation.contours:alternate:ragged" in encode[0].synthesized
