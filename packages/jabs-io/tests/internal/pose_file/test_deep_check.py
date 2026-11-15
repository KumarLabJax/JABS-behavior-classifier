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
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file import Finding, validate

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


def _checks(findings: list[Finding], severity: str) -> list[Finding]:
    """The findings of one severity."""
    return [f for f in findings if f.severity == severity]


def _deep_errors(path: Path) -> list[Finding]:
    """The deep check's disagreement errors for one file."""
    return [f for f in validate(path, deep=True) if f.check == "alternate_decodes"]


def _flip(path: Path, dataset: str, index: int | tuple[int, ...]) -> None:
    """Change one value: invert a boolean, or add one to a number."""
    with h5py.File(path, "r+") as h5:
        values = h5[dataset][()]
        values[index] = ~values[index] if values.dtype == bool else values[index] + 1
        h5[dataset][...] = values


def _rewrite_manifest(path: Path, mutate: Callable[[dict], object]) -> None:
    """Replace a file's manifest with a mutated copy."""
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


# --- Layouts the deep check must refuse to decode rather than raise on -------
#
# The deep check decodes through the catalog's segmentation layout. Every
# structural check can pass on a file that lays segmentation out some other
# way, so each of these must come back as findings, never as an exception.

DENSE = "/jabs/segmentation/contours"
LENGTH = "/jabs/segmentation/contour_length"
COUNT = "/jabs/segmentation/contour_count"


def _entry(manifest: dict, component_id: str) -> dict:
    """The manifest entry for one component."""
    return next(c for c in manifest["components"] if c["id"] == component_id)


def _replace(path: Path, dataset: str, data: npt.NDArray[np.generic]) -> None:
    """Replace a dataset's contents, shape and dtype at the same path."""
    with h5py.File(path, "r+") as h5:
        del h5[dataset]
        h5.create_dataset(dataset, data=data)


def _unchecked_without_raising(path: Path) -> list[Finding]:
    """Run the deep check; it must not raise, and must not claim a disagreement."""
    findings = validate(path, deep=True)
    assert not any(f.check == "alternate_decodes" for f in findings)
    return [f for f in findings if f.check == "alternate_deep_unchecked"]


def test_contours_without_a_length_policy_are_not_decoded(tmp_path):
    """Review finding 1: a contour_length past the point axis, unchecked because no policy names it.

    The alternate is rebuilt to agree with the oversized length, so the group
    sizes match and only the points comparison could see the problem - by
    comparing arrays of different lengths.
    """
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        point_axis = h5[DENSE].shape[3]
        group = h5[GROUP][()]
        first = int(group[1] - group[0])
        lengths = h5[LENGTH][()]
        lengths[0, 0, 0] = point_axis + 2
        h5[LENGTH][...] = lengths
        extra = point_axis + 2 - first
        points = h5[POINTS][()]
        padded = np.concatenate(
            [points[: group[1]], np.zeros((extra, 2), points.dtype), points[group[1] :]]
        )
        del h5[POINTS]
        h5.create_dataset(POINTS, data=padded)
        group[1:] += extra
        h5[GROUP][...] = group

    def drop_policy(manifest):
        contours = _entry(manifest, "jabs.segmentation.contours")
        contours["missing"] = {"policy": "none"}
        contours["alternates"][0]["shape"] = [int(padded.shape[0]), 2]

    _rewrite_manifest(path, drop_policy)
    assert _checks(validate(path), "error") == []
    unchecked = _unchecked_without_raising(path)
    assert any("contour_length" in f.message for f in unchecked)


def test_contours_of_the_wrong_rank_are_not_decoded(tmp_path):
    """Review finding 2: a (frame, slot) contours array has no contour or point axis."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        frames, slots = h5[COUNT].shape
    flat = np.zeros((frames, slots), dtype=np.int32)
    _replace(path, DENSE, flat)

    def flatten(manifest):
        contours = _entry(manifest, "jabs.segmentation.contours")
        contours.update(axes=["frame", "slot"], shape=[frames, slots], missing={"policy": "none"})
        for key in ("units", "coord_order"):
            contours.pop(key)

    _rewrite_manifest(path, flatten)
    assert _checks(validate(path), "error") == []
    assert _unchecked_without_raising(path)


def test_an_alternate_whose_trailing_axes_differ_is_not_decoded(tmp_path):
    """Review finding 3: a one-dimensional alternate cannot hold (x, y) points."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        n = h5[POINTS].shape[0]
    _replace(path, POINTS, np.zeros(n, dtype=np.int32))

    def flatten(manifest):
        alternate = _entry(manifest, "jabs.segmentation.contours")["alternates"][0]
        alternate.update(axes=["point"], shape=[n])
        for key in ("units", "coord_order"):
            alternate.pop(key)

    _rewrite_manifest(path, flatten)
    assert _checks(validate(path), "error") == []
    assert _unchecked_without_raising(path)


def test_flags_paired_with_other_offsets_are_not_decoded_through_the_contours(tmp_path):
    """Review finding 4: the flags alternate's offsets belong to a different leaf payload.

    Pairing only requires *some* leaf-level partner, so this is structurally
    valid, but decoding the flags through the contours' offsets compares the
    wrong groups.
    """
    path = _copy(tmp_path)
    other_group, other_instance = "/org.example.lab/group", "/org.example.lab/instance"
    with h5py.File(path, "r+") as h5:
        h5.create_dataset(other_group, data=h5[GROUP][()])
        h5.create_dataset(other_instance, data=h5[INSTANCE][()])
        h5.create_dataset("/org.example.lab/outline", data=h5[POINTS][()])
        n = h5[POINTS].shape[0]

    def repoint(manifest):
        manifest["components"].append(
            {
                "id": "org.example.lab.outline",
                "path": "/org.example.lab/outline",
                "axes": ["point", "coord"],
                "dtype": "int32",
                "shape": [n, 2],
                "units": "pixel",
                "coord_order": "xy",
                "encoding": {
                    "kind": "ragged",
                    "level": "leaf",
                    "group_offsets": other_group,
                    "instance_offsets": other_instance,
                },
                "missing": {"policy": "none"},
            }
        )
        flags = _entry(manifest, "jabs.segmentation.external_flag")["alternates"][0]
        flags["encoding"].update(group_offsets=other_group, instance_offsets=other_instance)

    _rewrite_manifest(path, repoint)
    assert _checks(validate(path), "error") == []
    unchecked = _unchecked_without_raising(path)
    assert any("external_flag" in f.message for f in unchecked)


def test_a_bad_contour_count_skips_only_its_own_block(tmp_path, monkeypatch):
    """Review finding 5: one undecodable block must not hide a disagreement in a later one."""
    monkeypatch.setattr(validate_module, "DEEP_CHECK_BLOCK_BYTES", 1)
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        contour_axis = h5[DENSE].shape[2]
        counts = h5[COUNT][()]
        counts[0, 0] = contour_axis + 1
        h5[COUNT][...] = counts
        last = h5[POINTS].shape[0] - 1
    _flip(path, POINTS, (last, 0))
    findings = validate(path, deep=True)
    assert any(f.check == "alternate_deep_unchecked" and "[0, 1)" in f.message for f in findings)
    assert any(f.check == "alternate_decodes" for f in findings)


def test_nan_in_a_float_baseline_matches_nan_in_the_alternate(tmp_path):
    """Review finding 6: NaN != NaN must not read as a disagreement."""
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        dense = h5[DENSE][()].astype(np.float32)
        points = h5[POINTS][()].astype(np.float32)
        dense[0, 0, 0, 0, 0] = np.nan
        points[0, 0] = np.nan
        for name, data in ((DENSE, dense), (POINTS, points)):
            del h5[name]
            h5.create_dataset(name, data=data)

    def to_float(manifest):
        contours = _entry(manifest, "jabs.segmentation.contours")
        contours["dtype"] = "float32"
        contours["alternates"][0]["dtype"] = "float32"

    _rewrite_manifest(path, to_float)
    assert _checks(validate(path, deep=True), "error") == []


def test_length_values_are_checked_block_by_block(tmp_path, monkeypatch):
    """Review finding 7: the length bound is read in blocks, and a late block still counts."""
    monkeypatch.setattr(validate_module, "VALUE_CHECK_BLOCK_BYTES", 1)
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        point_axis = h5[DENSE].shape[3]
        lengths = h5[LENGTH][()]
        lengths[-1, -1, -1] = point_axis + 1
        h5[LENGTH][...] = lengths
    errors = [f for f in validate(path) if f.severity == "error"]
    assert [f.check for f in errors] == ["mask_reference"]
