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

import dataclasses
import importlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file import (
    Finding,
    PoseFile,
    VideoInfo,
    validate,
    write_pose_file,
)

from .conftest import build_embedding_component, build_pose_file_with_embeddings

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
    """Copy a fixture into the test's directory, so the test can change it."""
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


def test_a_faithful_alternate_passes_the_deep_check() -> None:
    """The fixture's alternate is the baseline, re-encoded."""
    findings = validate(ALTERNATE, deep=True)
    assert _checks(findings, "error") == []
    assert not any(f.check == "alternate_deep_unchecked" for f in findings)


def test_the_deep_check_is_opt_in(tmp_path: Path) -> None:
    """Without deep=True a disagreeing alternate is not looked at."""
    path = _copy(tmp_path)
    _flip(path, POINTS, (0, 0))
    assert _checks(validate(path), "error") == []
    assert _deep_errors(path)


def test_a_moved_point_disagrees(tmp_path: Path) -> None:
    """One coordinate of the last point, off by one."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        last = h5[POINTS].shape[0] - 1
    _flip(path, POINTS, (last, 1))
    errors = _deep_errors(path)
    assert len(errors) == 1
    assert "jabs.segmentation.contours:alternate:ragged" in errors[0].message


def test_a_flipped_flag_disagrees(tmp_path: Path) -> None:
    """A hole reported as an outline, or the reverse."""
    path = _copy(tmp_path)
    _flip(path, FLAGS, 0)
    errors = _deep_errors(path)
    assert len(errors) == 1
    assert "jabs.segmentation.external_flag:alternate:ragged" in errors[0].message


def test_a_contour_handed_to_the_next_slot_disagrees(tmp_path: Path) -> None:
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


def test_a_regrouped_contour_disagrees(tmp_path: Path) -> None:
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


def test_a_disagreement_in_a_later_frame_block_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_an_alternate_this_build_cannot_decode_is_a_warning(tmp_path: Path) -> None:
    """Only the ragged segmentation alternate is defined; anything else is reported, not passed."""
    path = _copy(tmp_path)

    def rename(manifest: dict) -> None:
        for spec in manifest["components"]:
            for alternate in spec.get("alternates", []):
                alternate["name"] = "tiled"

    _rewrite_manifest(path, rename)
    findings = validate(path, deep=True)
    assert _checks(findings, "error") == []
    unchecked = [f for f in findings if f.check == "alternate_deep_unchecked"]
    assert len(unchecked) == 2
    assert all(f.severity == "warning" for f in unchecked)


def test_a_file_with_errors_is_not_decoded(tmp_path: Path) -> None:
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
def test_the_deep_check_never_raises(name: str) -> None:
    """Every fixture, valid or not, comes back as findings."""
    validate(DATA / name, deep=True)


def test_files_without_alternates_report_nothing_extra() -> None:
    """No alternates, nothing to decode, nothing to say."""
    assert validate(DATA / "valid-full.h5", deep=True) == validate(DATA / "valid-full.h5")


def test_the_baseline_wins_so_the_finding_names_the_alternate(tmp_path: Path) -> None:
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


def test_contours_without_a_length_policy_are_not_decoded(tmp_path: Path) -> None:
    """A contour_length past the point axis, unchecked because no policy names it.

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

    def drop_policy(manifest: dict) -> None:
        contours = _entry(manifest, "jabs.segmentation.contours")
        contours["missing"] = {"policy": "none"}
        contours["alternates"][0]["shape"] = [int(padded.shape[0]), 2]

    _rewrite_manifest(path, drop_policy)
    assert _checks(validate(path), "error") == []
    unchecked = _unchecked_without_raising(path)
    assert any("contour_length" in f.message for f in unchecked)


def test_contours_of_the_wrong_rank_are_not_decoded(tmp_path: Path) -> None:
    """A (frame, slot) contours array has no contour or point axis."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        frames, slots = h5[COUNT].shape
    flat = np.zeros((frames, slots), dtype=np.int32)
    _replace(path, DENSE, flat)

    def flatten(manifest: dict) -> None:
        contours = _entry(manifest, "jabs.segmentation.contours")
        contours.update(axes=["frame", "slot"], shape=[frames, slots], missing={"policy": "none"})
        for key in ("units", "coord_order"):
            contours.pop(key)

    _rewrite_manifest(path, flatten)
    assert _checks(validate(path), "error") == []
    assert _unchecked_without_raising(path)


def test_an_alternate_whose_trailing_axes_differ_is_not_decoded(tmp_path: Path) -> None:
    """A one-dimensional alternate cannot hold (x, y) points."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        n = h5[POINTS].shape[0]
    _replace(path, POINTS, np.zeros(n, dtype=np.int32))

    def flatten(manifest: dict) -> None:
        alternate = _entry(manifest, "jabs.segmentation.contours")["alternates"][0]
        alternate.update(axes=["point"], shape=[n])
        for key in ("units", "coord_order"):
            alternate.pop(key)

    _rewrite_manifest(path, flatten)
    assert _checks(validate(path), "error") == []
    assert _unchecked_without_raising(path)


def test_flags_paired_with_other_offsets_are_not_decoded_through_the_contours(
    tmp_path: Path,
) -> None:
    """The flags alternate's offsets belong to a different leaf payload.

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

    def repoint(manifest: dict) -> None:
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


def test_nan_in_a_float_baseline_matches_nan_in_the_alternate(tmp_path: Path) -> None:
    """NaN != NaN must not read as a disagreement."""
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        dense = h5[DENSE][()].astype(np.float32)
        points = h5[POINTS][()].astype(np.float32)
        dense[0, 0, 0, 0, 0] = np.nan
        points[0, 0] = np.nan
        for name, data in ((DENSE, dense), (POINTS, points)):
            del h5[name]
            h5.create_dataset(name, data=data)

    def to_float(manifest: dict) -> None:
        contours = _entry(manifest, "jabs.segmentation.contours")
        contours["dtype"] = "float32"
        contours["alternates"][0]["dtype"] = "float32"

    _rewrite_manifest(path, to_float)
    assert _checks(validate(path, deep=True), "error") == []


def test_length_values_are_checked_block_by_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The length bound is read in blocks, and a late block still counts."""
    monkeypatch.setattr(validate_module, "VALUE_CHECK_BLOCK_BYTES", 1)
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        point_axis = h5[DENSE].shape[3]
        lengths = h5[LENGTH][()]
        lengths[-1, -1, -1] = point_axis + 1
        h5[LENGTH][...] = lengths
    errors = [f for f in validate(path) if f.severity == "error"]
    assert [f.check for f in errors] == ["length_bound"]


# --- The bounds the decode relies on ------------------------------------------
#
# The decode slices through contour_count and contour_length. The structural
# pass bounds both, and the deep check only runs on a file with no errors, so
# a bad count or length is an error before anything is decoded.


def _retype(path: Path, dataset: str, component_id: str, data: npt.NDArray[np.generic]) -> None:
    """Replace a baseline dataset and declare its new dtype in the manifest."""
    _replace(path, dataset, data)
    dtype = "string" if data.dtype.kind == "O" else str(data.dtype)

    def declare(manifest: dict) -> None:
        _entry(manifest, component_id)["dtype"] = dtype

    _rewrite_manifest(path, declare)


def _errors(findings: list[Finding]) -> list[str]:
    """The check names of the error findings."""
    return [f.check for f in findings if f.severity == "error"]


@pytest.mark.parametrize(
    ("index", "value"),
    [((-1, -1), "past"), ((0, 0), "negative")],
    ids=["past-the-axis-in-the-last-block", "negative"],
)
def test_a_contour_count_outside_the_contour_axis_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: tuple[int, int], value: str
) -> None:
    """A count past the contour axis, or below zero, makes a reader slice past the data.

    One frame per block, so a count in the last frame is only seen if every
    block is read.
    """
    monkeypatch.setattr(validate_module, "VALUE_CHECK_BLOCK_BYTES", 1)
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        contour_axis = h5[DENSE].shape[2]
        counts = h5[COUNT][()].astype(np.int32)
    counts[index] = contour_axis + 1 if value == "past" else -1
    _retype(path, COUNT, "jabs.segmentation.contour_count", counts)
    assert _errors(validate(path)) == ["contour_count_bound"]
    assert _errors(validate(path, deep=True)) == ["contour_count_bound"]


@pytest.mark.parametrize(
    "convert",
    [
        lambda v: np.array([[str(x) for x in row] for row in v], dtype=h5py.string_dtype()),
        lambda v: v.astype(np.float32) + 0.5,
    ],
    ids=["string", "float"],
)
def test_a_contour_count_that_is_not_an_integer_is_an_error(
    tmp_path: Path, convert: Callable[[npt.NDArray[np.generic]], npt.NDArray[np.generic]]
) -> None:
    """A float count of 1.5 would be truncated to one contour; a string one cannot be compared."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        values = h5[COUNT][()]
    _retype(path, COUNT, "jabs.segmentation.contour_count", convert(values))
    assert _errors(validate(path)) == ["contour_count_bound"]
    assert not any(f.check == "alternate_decodes" for f in validate(path, deep=True))


def test_integer_baseline_flags_are_not_decoded(tmp_path: Path) -> None:
    """Flags are compared as booleans, and nothing structural requires the baseline's to be."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        values = h5["/jabs/segmentation/external_flag"][()]
    _retype(
        path,
        "/jabs/segmentation/external_flag",
        "jabs.segmentation.external_flag",
        values.astype(np.uint8),
    )
    assert _errors(validate(path)) == []
    unchecked = _unchecked_without_raising(path)
    assert any("jabs.segmentation.external_flag has dtype uint8" in f.message for f in unchecked)


def test_an_integer_flag_alternate_is_not_decoded(tmp_path: Path) -> None:
    """A uint8 flag alternate holding 2 is neither True nor False, so it is not compared."""
    path = _copy(tmp_path)
    with h5py.File(path, "r") as h5:
        flags = h5[FLAGS][()].astype(np.uint8)
    flags[0] = 2
    _replace(path, FLAGS, flags)

    def declare(manifest: dict) -> None:
        _entry(manifest, "jabs.segmentation.external_flag")["alternates"][0]["dtype"] = "uint8"

    _rewrite_manifest(path, declare)
    assert _errors(validate(path)) == []
    unchecked = _unchecked_without_raising(path)
    assert any("must be bool, not uint8" in f.message for f in unchecked)


def test_a_flag_disagreement_names_both_values(tmp_path: Path) -> None:
    """The message gives the alternate's value and the baseline's, which differ."""
    path = _copy(tmp_path)
    _flip(path, FLAGS, 0)
    (error,) = _deep_errors(path)
    assert "True" in error.message and "False" in error.message


# --- Embedding rows: each row is a whole vector or all-NaN ------------------

EMBEDDING_INSTANCE = "jabs.embedding.trunk.instance"


def _row_errors(path: Path, check: str) -> list[Finding]:
    """The deep check's errors of one embedding row rule."""
    return [f for f in validate(path, deep=True) if f.severity == "error" and f.check == check]


def test_deep_check_reads_float16_in_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows are read a block at a time, and a bad row in the last, partial block is named.

    With ten frames and three per block, frame 9 is alone in the fourth block,
    so a message naming frame 0 would mean the block offset was dropped, and a
    check that stopped at the last whole block would miss it.
    """
    # One frame of the D=8 float16 frame embedding costs 8 * (2 + 1) = 24 bytes
    # (payload plus NaN mask), so a 72-byte budget reads three frames at a time.
    monkeypatch.setattr(validate_module, "DEEP_CHECK_BLOCK_BYTES", 72)
    component = build_embedding_component("trunk", "frame", frames=10)
    rows = np.array(component.data, copy=True)
    assert rows.dtype == np.float16
    rows[9, 2] = np.nan
    pose_file = PoseFile(
        dimensions={"frame": 10, "slot": 1, "identity": 1},
        video=VideoInfo(frame_count=10),
        components=(dataclasses.replace(component, data=rows),),
    )
    path = tmp_path / "blocks.h5"
    write_pose_file(pose_file, path)
    assert _checks(validate(path), "error") == []

    payload_path = component.path
    reads: list[object] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(self: h5py.Dataset, key: object) -> object:
        if self.name == payload_path:
            reads.append(key)
        return original_getitem(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    errors = _row_errors(path, "embedding_rows_whole")

    assert len(errors) == 1
    assert "jabs.embedding.trunk.frame" in errors[0].message
    assert "frame 9" in errors[0].message
    # Only the row check reads the payload, and only in bounded frame slices.
    assert all(isinstance(key, tuple) and isinstance(key[0], slice) for key in reads), reads
    spans = [range(10)[key[0]] for key in reads]
    assert all(len(span) <= 3 for span in spans), reads
    assert sorted(frame for span in spans for frame in span) == list(range(10))


def test_deep_check_slices_the_feature_axis_when_one_frame_exceeds_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared shape wider than the budget is read a feature slice at a time, not whole.

    One frame of the D=8 float16 frame embedding costs 24 bytes with its mask;
    a 9-byte budget fits three features, so each read covers one frame and at
    most three features, and a NaN in the last feature is still counted.
    """
    monkeypatch.setattr(validate_module, "DEEP_CHECK_BLOCK_BYTES", 9)
    component = build_embedding_component("trunk", "frame", frames=4)
    rows = np.array(component.data, copy=True)
    rows[2, 7] = np.nan
    pose_file = PoseFile(
        dimensions={"frame": 4, "slot": 1, "identity": 1},
        video=VideoInfo(frame_count=4),
        components=(dataclasses.replace(component, data=rows),),
    )
    path = tmp_path / "wide.h5"
    write_pose_file(pose_file, path)

    payload_path = component.path
    reads: list[tuple] = []
    original_getitem = h5py.Dataset.__getitem__

    def recording_getitem(self: h5py.Dataset, key: object) -> object:
        if self.name == payload_path:
            reads.append(key)
        return original_getitem(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", recording_getitem)
    errors = _row_errors(path, "embedding_rows_whole")

    assert len(errors) == 1
    assert "frame 2" in errors[0].message
    assert "1 of 8" in errors[0].message
    assert reads, "the payload was never read"
    for key in reads:
        frames, features = range(4)[key[0]], range(8)[key[-1]]
        assert len(frames) == 1 and len(features) <= 3, key


def test_deep_check_reports_an_unreadable_embedding_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row block that cannot be read is an error finding, not an exception."""
    pose_file = build_pose_file_with_embeddings()
    path = tmp_path / "unreadable.h5"
    write_pose_file(pose_file, path)
    assert _checks(validate(path, deep=True), "error") == []

    payload_path = pose_file.component(EMBEDDING_INSTANCE).path
    original_getitem = h5py.Dataset.__getitem__

    def failing_getitem(self: h5py.Dataset, key: object) -> object:
        if self.name == payload_path:
            raise OSError("Can't synchronously read data (corrupt chunk)")
        return original_getitem(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", failing_getitem)
    errors = _checks(validate(path, deep=True), "error")

    assert [f.check for f in errors] == ["file_readable"]
    assert EMBEDDING_INSTANCE in errors[0].message
    assert "corrupt chunk" in errors[0].message


def _with_instance_rows(
    pose_file: PoseFile, edit: Callable[[npt.NDArray[np.float16]], None]
) -> PoseFile:
    """The pose file with its trunk.instance rows edited in place on a copy."""
    rows = np.array(pose_file.component(EMBEDDING_INSTANCE).data, copy=True)
    edit(rows)
    return dataclasses.replace(
        pose_file,
        components=tuple(
            dataclasses.replace(c, data=rows) if c.id == EMBEDDING_INSTANCE else c
            for c in pose_file.components
        ),
    )


def _fill_empty_slot(rows: npt.NDArray[np.float16]) -> None:
    """Store a vector at slot (0, 1), which the builder leaves unoccupied."""
    rows[0, 1] = 1.0


def test_instance_embedding_without_slot_occupied_is_not_checked(tmp_path: Path) -> None:
    """Without jabs.pose.slot_occupied, nothing says which slots are empty.

    So no row is held to the unoccupied-slot rule.
    """
    with_vector = _with_instance_rows(build_pose_file_with_embeddings(), _fill_empty_slot)
    without_occupancy = dataclasses.replace(
        with_vector,
        components=tuple(c for c in with_vector.components if c.id != "jabs.pose.slot_occupied"),
    )
    path = tmp_path / "no-occupancy.h5"
    write_pose_file(without_occupancy, path)

    assert _checks(validate(path, deep=True), "error") == []


def test_a_vector_in_an_unoccupied_slot_is_an_error(tmp_path: Path) -> None:
    """The same rows, with slot_occupied present, break the unoccupied-slot rule."""
    pose_file = _with_instance_rows(build_pose_file_with_embeddings(), _fill_empty_slot)
    path = tmp_path / "occupancy.h5"
    write_pose_file(pose_file, path)

    assert _checks(validate(path), "error") == []
    errors = _row_errors(path, "embedding_unoccupied_nan")
    assert len(errors) == 1
    assert EMBEDDING_INSTANCE in errors[0].message
    assert "frame 0, slot 1" in errors[0].message


def test_many_bad_rows_give_one_finding_naming_the_first(tmp_path: Path) -> None:
    """One finding per component and rule, naming the first offending row."""

    def damage(rows: npt.NDArray[np.float16]) -> None:
        rows[2, 1, 0] = np.nan
        rows[3, 0, 5] = np.nan

    pose_file = _with_instance_rows(build_pose_file_with_embeddings(), damage)
    path = tmp_path / "many.h5"
    write_pose_file(pose_file, path)

    errors = _row_errors(path, "embedding_rows_whole")
    assert len(errors) == 1
    assert "frame 2, slot 1" in errors[0].message


def test_embedding_rows_are_checked_only_when_asked(tmp_path: Path) -> None:
    """Reading every vector costs as much as the payload, so it waits for deep=True."""
    path = _copy(tmp_path, DATA / "invalid-embedding-partial-nan.h5")
    assert not any(f.check == "embedding_rows_whole" for f in validate(path))
    assert _row_errors(path, "embedding_rows_whole")


def test_an_unusable_slot_occupied_skips_the_occupancy_rule(tmp_path: Path) -> None:
    """A slot_occupied of the wrong shape is a structural error, not a crash here."""
    path = _copy(tmp_path, DATA / "invalid-embedding-unoccupied.h5")
    _replace(path, "/jabs/pose/slot_occupied", np.ones((2, 2), dtype=bool))
    findings = validate(path, deep=True)
    assert _checks(findings, "error")
    assert not any(f.check == "embedding_unoccupied_nan" for f in findings)


def test_an_embedding_of_the_wrong_rank_is_not_read(tmp_path: Path) -> None:
    """The structural pass reports the shape; the row check skips what it cannot index."""
    path = _copy(tmp_path, DATA / "invalid-embedding-partial-nan.h5")
    _replace(path, "/jabs/embedding/trunk/instance", np.full((8,), np.nan, dtype=np.float16))
    findings = validate(path, deep=True)
    assert _checks(findings, "error")
    assert not any(f.check == "embedding_rows_whole" for f in findings)
