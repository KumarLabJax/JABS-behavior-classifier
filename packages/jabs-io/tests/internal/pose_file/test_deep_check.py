"""The opt-in deep check: each alternate decodes to its baseline, per (frame, slot).

Every other check reads shapes, offsets and the manifest. An alternate can pass
all of them and still say something different from its baseline: a point
moved, a flag flipped, a contour handed to the wrong slot. Only decoding both
and comparing them catches that, which costs as much as reading the payload,
so ADR 0002 makes it opt-in.

The validator decodes in vectorized frame blocks. The conformance suite decodes
the same fixture one instance at a time, exactly as the ADR spells it out, so
the two are independent implementations checking each other.
"""

import importlib
import json
import shutil
from pathlib import Path

import h5py
import pytest

from jabs.io.internal.pose_file import validate

# The package re-exports the function under the module's name, so reach the
# module itself to patch it.
validate_module = importlib.import_module("jabs.io.internal.pose_file.validate")

DATA = Path(__file__).parents[2] / "data" / "pose-format"
ALTERNATE = DATA / "valid-ragged-alternate.h5"

POINTS = "/jabs/segmentation/ragged/contours"
FLAGS = "/jabs/segmentation/ragged/external_flag"
GROUP = "/jabs/segmentation/ragged/contour_offsets"
INSTANCE = "/jabs/segmentation/ragged/instance_offsets"


def _copy(tmp_path: Path, source: Path = ALTERNATE) -> Path:
    target = tmp_path / source.name
    shutil.copy(source, target)
    return target


def _checks(findings, severity):
    return [f for f in findings if f.severity == severity]


def _deep_errors(path):
    return [f for f in validate(path, deep=True) if f.check == "alternate_decodes"]


def _flip(path: Path, dataset: str, index) -> None:
    with h5py.File(path, "r+") as h5:
        values = h5[dataset][()]
        values[index] = ~values[index] if values.dtype == bool else values[index] + 1
        h5[dataset][...] = values


def _rewrite_manifest(path: Path, mutate) -> None:
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5["manifest"][()])
        mutate(manifest)
        del h5["manifest"]
        h5.create_dataset(
            "manifest", data=json.dumps(manifest), dtype=h5py.string_dtype(encoding="utf-8")
        )


def test_a_faithful_alternate_passes_the_deep_check():
    """The fixture's alternate is the baseline, re-encoded."""
    findings = validate(ALTERNATE, deep=True)
    assert _checks(findings, "error") == []
    assert not any(f.check == "alternate_deep_unchecked" for f in findings)


def test_the_deep_check_is_opt_in(tmp_path):
    """Without deep=True a disagreeing alternate is not looked at."""
    path = _copy(tmp_path)
    _flip(path, POINTS, (0, 0))
    assert _checks(validate(path), "error") == []
    assert _deep_errors(path)


def test_a_moved_point_disagrees(tmp_path):
    """One coordinate of the last point, off by one."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        last = h5[POINTS].shape[0] - 1
    _flip(path, POINTS, (last, 1))
    errors = _deep_errors(path)
    assert len(errors) == 1
    assert "jabs.segmentation.contours:alternate:ragged" in errors[0].message


def test_a_flipped_flag_disagrees(tmp_path):
    """A hole reported as an outline, or the reverse."""
    path = _copy(tmp_path)
    _flip(path, FLAGS, 0)
    errors = _deep_errors(path)
    assert len(errors) == 1
    assert "jabs.segmentation.external_flag:alternate:ragged" in errors[0].message


def test_a_contour_handed_to_the_next_slot_disagrees(tmp_path):
    """Offsets still well formed and every point in place, but one instance owns one group fewer.

    No structural check can see this: the offsets start at 0, never
    decrease, and end where they must.
    """
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        offsets = h5[INSTANCE][()]
        # The first instance boundary that can move down one group and stay
        # non-decreasing.
        boundary = next(
            i
            for i in range(1, len(offsets) - 1)
            if offsets[i] > offsets[i - 1] and offsets[i] <= offsets[i + 1]
        )
        offsets[boundary] -= 1
        h5[INSTANCE][...] = offsets
    assert _checks(validate(path), "error") == []
    errors = _deep_errors(path)
    assert errors
    assert "frame" in errors[0].message and "slot" in errors[0].message


def test_a_regrouped_contour_disagrees(tmp_path):
    """The same points split differently between two contours of one instance."""
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        offsets = h5[GROUP][()]
        boundary = next(
            i
            for i in range(1, len(offsets) - 1)
            if offsets[i] - offsets[i - 1] > 1 and offsets[i + 1] > offsets[i]
        )
        offsets[boundary] -= 1
        h5[GROUP][...] = offsets
    assert _checks(validate(path), "error") == []
    assert _deep_errors(path)


def test_a_disagreement_in_a_later_frame_block_is_found(tmp_path, monkeypatch):
    """The decode reads frame blocks so a full-size baseline never sits in memory at once.

    One frame per block makes every boundary a block boundary.
    """
    monkeypatch.setattr(validate_module, "DEEP_CHECK_BLOCK_BYTES", 1)
    assert _checks(validate(ALTERNATE, deep=True), "error") == []
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        last = h5[POINTS].shape[0] - 1
    _flip(path, POINTS, (last, 0))
    assert _deep_errors(path)


def test_an_alternate_this_build_cannot_decode_is_a_warning(tmp_path):
    """Only the ragged segmentation alternate is defined; anything else is reported, not passed."""
    path = _copy(tmp_path)

    def rename(manifest):
        for spec in manifest["components"]:
            for alternate in spec.get("alternates", []):
                alternate["name"] = "tiled"

    _rewrite_manifest(path, rename)
    findings = validate(path, deep=True)
    assert _checks(findings, "error") == []
    unchecked = [f for f in findings if f.check == "alternate_deep_unchecked"]
    assert len(unchecked) == 2
    assert all(f.severity == "warning" for f in unchecked)


def test_a_file_with_errors_is_not_decoded(tmp_path):
    """Decoding trusts offsets and shapes the structural checks have already passed."""
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        offsets = h5[INSTANCE][()]
        offsets[-1] += 7
        h5[INSTANCE][...] = offsets
    findings = validate(path, deep=True)
    assert _checks(findings, "error")
    assert not any(f.check == "alternate_decodes" for f in findings)
    assert any(f.check == "alternate_deep_unchecked" for f in findings)


@pytest.mark.parametrize("name", sorted(p.name for p in DATA.glob("*.h5")))
def test_the_deep_check_never_raises(name):
    """Every fixture, valid or not, comes back as findings."""
    validate(DATA / name, deep=True)


def test_files_without_alternates_report_nothing_extra():
    """No alternates, nothing to decode, nothing to say."""
    assert validate(DATA / "valid-full.h5", deep=True) == validate(DATA / "valid-full.h5")


def test_the_baseline_wins_so_the_finding_names_the_alternate(tmp_path):
    """ADR 0002: a reader finding them disagreeing trusts the baseline."""
    path = _copy(tmp_path)
    _flip(path, POINTS, (0, 0))
    message = _deep_errors(path)[0].message
    assert "alternate" in message and "baseline" in message
