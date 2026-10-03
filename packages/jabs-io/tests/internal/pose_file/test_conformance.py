"""The fixture corpus that any independent implementation can test against.

Regenerate the fixtures with
``uv run python packages/jabs-io/tests/data/pose-format/generate.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from jabs.io.internal.pose_file.reader import read_component, read_pose_file
from jabs.io.internal.pose_file.validate import validate

DATA = Path(__file__).parents[2] / "data" / "pose-format"

VALID = ["valid-minimal.h5", "valid-full.h5", "valid-sparse.h5"]
INVALID = {
    "invalid-shape-mismatch.h5": "dtype_shape_match",
    "invalid-missing-payload.h5": "component_path_exists",
    "invalid-dangling-mask.h5": "mask_reference",
    "invalid-keypoint-axis.h5": "keypoint_axis_length",
    "invalid-not-a-pose-file.h5": "root_attrs",
}


@pytest.mark.parametrize("name", VALID)
def test_valid_fixtures_have_no_errors(name):
    """Every valid fixture conforms."""
    errors = [f for f in validate(DATA / name) if f.severity == "error"]
    assert errors == [], f"{name} should be valid, got {[f.message for f in errors]}"


@pytest.mark.parametrize("name,expected_check", sorted(INVALID.items()))
def test_invalid_fixtures_report_their_check(name, expected_check):
    """Every invalid fixture fails the specific rule it was built to break."""
    findings = validate(DATA / name)
    assert any(f.check == expected_check and f.severity == "error" for f in findings), (
        f"{name} should report {expected_check}; got {sorted({f.check for f in findings})}"
    )


def test_fixture_corpus_is_complete():
    """The corpus on disk covers everything the suite names."""
    present = {p.name for p in DATA.glob("*.h5")}
    assert present >= set(VALID) | set(INVALID)


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
