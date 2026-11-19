"""The writer writes ragged alternates, and can deep-check before publishing.

A ``ragged`` set -- the shared offsets and every alternate of one name -- is
written as one run of datasets immediately before the first component that
carries it, so the set occupies one byte range (spec 0029-A §2.6). With
``deep=True`` the temporary file is decoded and compared against its dense
baseline before it replaces the destination.
"""

import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file import (
    Alternate,
    RaggedIndex,
    read_pose_file,
    validate,
    write_pose_file,
)
from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.types import Component, PoseFile

from .conftest import build_sample_pose_file, ragged_segmentation

CREATED = "2026-11-18T00:00:00Z"
PREFIX = "/jabs/segmentation/ragged"
POINTS = f"{PREFIX}/contours"
FLAGS = f"{PREFIX}/external_flag"
GROUP = f"{PREFIX}/contour_offsets"
INSTANCE = f"{PREFIX}/instance_offsets"
RAGGED_SET = (POINTS, FLAGS, GROUP, INSTANCE)
CONTOURS_ID = "jabs.segmentation.contours"
FLAG_ID = "jabs.segmentation.external_flag"
SENTINEL = b"the previous file"

Segmentation = dict[str, npt.NDArray[np.generic]]


def _empty_segmentation(frames: int = 4, slots: int = 2) -> Segmentation:
    """Segmentation in which no instance has a contour, so the payload is empty."""
    return {
        "contours": np.full((frames, slots, 2, 3, 2), -1, dtype=np.int32),
        "contour_count": np.zeros((frames, slots), dtype=np.uint32),
        "contour_length": np.zeros((frames, slots, 2), dtype=np.uint32),
        "external_flag": np.zeros((frames, slots, 2), dtype=bool),
        "points": np.zeros((0, 2), dtype=np.int32),
        "flags": np.zeros((0,), dtype=bool),
        "contour_offsets": np.zeros((1,), dtype=np.uint64),
        "instance_offsets": np.zeros((frames * slots + 1,), dtype=np.uint64),
    }


def _index(seg: Segmentation, **overrides: Any) -> RaggedIndex:
    """The shared offsets of the segmentation's ragged alternates."""
    fields: dict[str, Any] = {
        "group_offsets_path": GROUP,
        "instance_offsets_path": INSTANCE,
        "group_offsets": seg["contour_offsets"],
        "instance_offsets": seg["instance_offsets"],
    }
    return RaggedIndex(**(fields | overrides))


def _points_alternate(seg: Segmentation, index: RaggedIndex, **overrides: Any) -> Alternate:
    """The leaf-level ragged alternate of the contours."""
    fields: dict[str, Any] = {
        "name": "ragged",
        "path": POINTS,
        "axes": ("point", "coord"),
        "data": seg["points"],
        "missing": {"policy": "none"},
        "index": index,
        "units": "pixel",
        "coord_order": "xy",
    }
    return Alternate(**(fields | overrides))


def _flags_alternate(seg: Segmentation, index: RaggedIndex, **overrides: Any) -> Alternate:
    """The group-level ragged alternate of the external flags."""
    fields: dict[str, Any] = {
        "name": "ragged",
        "path": FLAGS,
        "axes": ("contour",),
        "data": seg["flags"],
        "missing": {"policy": "none"},
        "index": index,
        "level": "group",
    }
    return Alternate(**(fields | overrides))


def _segmentation_file(
    seg: Segmentation | None = None,
    points_alternates: Callable[[Segmentation, RaggedIndex], tuple[Alternate, ...]] | None = None,
    flag_alternates: Callable[[Segmentation, RaggedIndex], tuple[Alternate, ...]] | None = None,
    index: RaggedIndex | None = None,
) -> PoseFile:
    """The sample pose file, then dense segmentation carrying the ragged alternates.

    Args:
        seg: The segmentation arrays; defaults to ``ragged_segmentation()``.
        points_alternates: Builds the contours' alternates from the arrays and
            the shared index; defaults to the leaf-level ``ragged`` alternate.
        flag_alternates: The same for the external flags; defaults to the
            group-level ``ragged`` alternate.
        index: The index the builders are handed; defaults to ``_index(seg)``.

    Returns:
        A pose file whose segmentation the deep check can decode.
    """
    seg = ragged_segmentation() if seg is None else seg
    index = _index(seg) if index is None else index
    points_alternates = points_alternates or (lambda s, i: (_points_alternate(s, i),))
    flag_alternates = flag_alternates or (lambda s, i: (_flags_alternate(s, i),))
    sample = build_sample_pose_file()
    segmentation = (
        Component(
            id=CONTOURS_ID,
            axes=("frame", "slot", "contour", "point", "coord"),
            data=seg["contours"],
            missing={"policy": "length", "length": "jabs.segmentation.contour_length"},
            units="pixel",
            coord_order="xy",
            provenance="jabs.pose",
            alternates=points_alternates(seg, index),
        ),
        Component(
            id="jabs.segmentation.contour_count",
            axes=("frame", "slot"),
            data=seg["contour_count"],
            missing={"policy": "none"},
            provenance="jabs.pose",
        ),
        Component(
            id="jabs.segmentation.contour_length",
            axes=("frame", "slot", "contour"),
            data=seg["contour_length"],
            missing={"policy": "none"},
            provenance="jabs.pose",
        ),
        Component(
            id=FLAG_ID,
            axes=("frame", "slot", "contour"),
            data=seg["external_flag"],
            missing={"policy": "none"},
            provenance="jabs.pose",
            alternates=flag_alternates(seg, index),
        ),
    )
    return dataclasses.replace(sample, components=(*sample.components, *segmentation))


def _errors(path: Path) -> list[str]:
    """The deep validator's errors, as ``check: message`` lines."""
    return [f"{f.check}: {f.message}" for f in validate(path, deep=True) if f.severity == "error"]


def _extents(dataset: h5py.Dataset) -> list[tuple[int, int]]:
    """Each ``(start, end)`` byte range a dataset's raw data occupies in the file."""
    if dataset.chunks is None:
        offset = dataset.id.get_offset()
        if offset is None:
            return []
        return [(offset, offset + dataset.id.get_storage_size())]
    extents = []
    for i in range(dataset.id.get_num_chunks()):
        info = dataset.id.get_chunk_info(i)
        extents.append((info.byte_offset, info.byte_offset + info.size))
    return extents


def _all_extents(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Every dataset's byte ranges, by path."""
    extents: dict[str, list[tuple[int, int]]] = {}
    with h5py.File(path, "r") as h5:

        def note(name: str, node: object) -> None:
            if isinstance(node, h5py.Dataset):
                extents["/" + name] = _extents(node)

        h5.visititems(note)
    return extents


def _span(extents: dict[str, list[tuple[int, int]]], paths: tuple[str, ...]) -> tuple[int, int]:
    """The smallest byte range holding every extent of the named datasets."""
    ranges = [r for p in paths for r in extents[p]]
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def test_alternates_are_written_and_pass_the_deep_check(tmp_path: Path) -> None:
    """The alternates decode to their dense baseline."""
    path = tmp_path / "alternates.h5"
    write_pose_file(_segmentation_file(), path, created=CREATED)
    findings = validate(path, deep=True)
    assert [f for f in findings if f.severity == "error"] == []
    assert not any(f.check == "alternate_deep_unchecked" for f in findings)
    seg = ragged_segmentation()
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5[POINTS][()], seg["points"])
        np.testing.assert_array_equal(h5[FLAGS][()], seg["flags"])
        np.testing.assert_array_equal(h5[GROUP][()], seg["contour_offsets"])
        np.testing.assert_array_equal(h5[INSTANCE][()], seg["instance_offsets"])


def test_block_sourced_alternates_are_streamed(tmp_path: Path) -> None:
    """An alternate's payload and its offsets may be block sources too."""
    seg = ragged_segmentation()

    def streamed(array: npt.NDArray[np.generic]) -> BlockSource:
        def blocks() -> Iterator[tuple[int, npt.NDArray[np.generic]]]:
            for start in range(0, array.shape[0], 3):
                yield start, array[start : start + 3]

        return BlockSource(array.shape, array.dtype, blocks)

    index = _index(
        seg,
        group_offsets=streamed(seg["contour_offsets"]),
        instance_offsets=streamed(seg["instance_offsets"]),
    )
    pose_file = _segmentation_file(
        seg,
        points_alternates=lambda s, _: (_points_alternate(s, index, data=streamed(s["points"])),),
        flag_alternates=lambda s, _: (_flags_alternate(s, index, data=streamed(s["flags"])),),
    )
    path = tmp_path / "streamed.h5"
    write_pose_file(pose_file, path, created=CREATED, deep=True)
    assert _errors(path) == []
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5[POINTS][()], seg["points"])
        np.testing.assert_array_equal(h5[GROUP][()], seg["contour_offsets"])


def test_ragged_layout(tmp_path: Path) -> None:
    """Leaf payloads are chunked and compressed; offsets and group payloads are not."""
    path = tmp_path / "layout.h5"
    write_pose_file(_segmentation_file(), path, created=CREATED)
    n = ragged_segmentation()["points"].shape[0]
    with h5py.File(path, "r") as h5:
        payload = h5[POINTS]
        assert payload.chunks == (min(65536, n), 2)
        assert payload.compression == "gzip"
        assert payload.compression_opts == 1
        assert payload.shuffle is True
        for contiguous in (GROUP, INSTANCE, FLAGS):
            assert h5[contiguous].chunks is None, contiguous
            assert h5[contiguous].compression is None, contiguous
    assert not any(f.check == "layout_matches_file" for f in validate(path))


def test_the_ragged_set_occupies_one_byte_range(tmp_path: Path) -> None:
    """No other dataset's storage starts inside the ragged set's span (spec 0029-A §2.6)."""
    path = tmp_path / "span.h5"
    write_pose_file(_segmentation_file(), path, created=CREATED)
    extents = _all_extents(path)
    start, end = _span(extents, RAGGED_SET)
    intruders = [
        (name, extent)
        for name, ranges in extents.items()
        if name not in RAGGED_SET
        for extent in ranges
        if start <= extent[0] < end
    ]
    assert intruders == []


def test_ragged_set_is_written_before_its_first_carrier(tmp_path: Path) -> None:
    """The ragged set follows the keypoint datasets in the file's storage order."""
    path = tmp_path / "order.h5"
    write_pose_file(_segmentation_file(), path, created=CREATED)
    extents = _all_extents(path)
    start, end = _span(extents, RAGGED_SET)
    keypoint_ends = [
        end
        for name, ranges in extents.items()
        if name.startswith("/jabs/pose/")
        for _, end in ranges
    ]
    assert keypoint_ends
    assert max(keypoint_ends) <= start
    # Not after everything: components written after the first carrier follow
    # the set, which tells "before the carrier" apart from "at the end".
    later = _span(
        extents, ("/jabs/segmentation/contour_count", "/jabs/segmentation/contour_length")
    )
    assert end <= later[0]


def test_an_alternate_with_no_points_writes(tmp_path: Path) -> None:
    """With no contours anywhere, the payload is empty, contiguous, and still decodes."""
    path = tmp_path / "empty.h5"
    write_pose_file(_segmentation_file(_empty_segmentation()), path, created=CREATED)
    with h5py.File(path, "r") as h5:
        assert h5[POINTS].shape == (0, 2)
        assert h5[POINTS].chunks is None
        assert h5[POINTS].compression is None
    assert _errors(path) == []


def test_deep_write_refuses_a_disagreeing_alternate(tmp_path: Path) -> None:
    """A deep write that fails leaves the destination as it was, and no temporary."""
    seg = ragged_segmentation()
    seg["points"] = seg["points"].copy()
    seg["points"][0, 0] += 1
    pose_file = _segmentation_file(seg)
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)

    with pytest.raises(ValueError, match="alternate_decodes"):
        write_pose_file(pose_file, path, created=CREATED, deep=True)
    assert path.read_bytes() == SENTINEL
    assert sorted(tmp_path.iterdir()) == [path]

    write_pose_file(pose_file, path, created=CREATED)
    assert path.read_bytes() != SENTINEL


def test_deep_write_refuses_an_alternate_it_cannot_decode(tmp_path: Path) -> None:
    """``deep`` promises a decoded alternate, so one it cannot decode is not published.

    The validator reports an undecodable alternate as a warning; the deep write
    refuses it, because publishing it would let a stale alternate through.
    Without ``deep`` the same file publishes.
    """
    seg = ragged_segmentation()
    stale = np.full_like(seg["points"], 9999)
    pose_file = _segmentation_file(
        seg,
        points_alternates=lambda s, i: (_points_alternate(s, i, name="other", data=stale),),
        flag_alternates=lambda s, i: (_flags_alternate(s, i, name="other"),),
    )
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)

    with pytest.raises(ValueError, match="alternate_deep_unchecked"):
        write_pose_file(pose_file, path, created=CREATED, deep=True)
    assert path.read_bytes() == SENTINEL
    assert sorted(tmp_path.iterdir()) == [path]

    write_pose_file(pose_file, path, created=CREATED)
    assert path.read_bytes() != SENTINEL
    assert any(f.check == "alternate_deep_unchecked" for f in validate(path, deep=True))


def test_a_ragged_baseline_is_still_refused(tmp_path: Path) -> None:
    """Alternates are written; a ragged component encoding is still not."""
    sample = build_sample_pose_file()
    ragged = dataclasses.replace(
        sample.components[-1],
        encoding={
            "kind": "ragged",
            "level": "leaf",
            "group_offsets": GROUP,
            "instance_offsets": INSTANCE,
        },
    )
    pose_file = dataclasses.replace(sample, components=(*sample.components[:-1], ragged))
    with pytest.raises(NotImplementedError):
        write_pose_file(pose_file, tmp_path / "ragged.h5", created=CREATED)


def _offsets_at_a_component_path() -> PoseFile:
    """Both alternates share offsets whose group array sits at the contour counts' path."""
    seg = ragged_segmentation()
    return _segmentation_file(
        seg, index=_index(seg, group_offsets_path="/jabs/segmentation/contour_count")
    )


def _payload_at_an_offset_path() -> PoseFile:
    """The flags' payload sits at the shared group offsets' path."""
    return _segmentation_file(
        flag_alternates=lambda s, i: (_flags_alternate(s, i, path=GROUP),),
    )


def _payload_under_a_component() -> PoseFile:
    """The flags' payload sits beneath a component's dataset."""
    return _segmentation_file(
        flag_alternates=lambda s, i: (
            _flags_alternate(s, i, path="/jabs/segmentation/contours/ragged"),
        ),
    )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        pytest.param(_offsets_at_a_component_path, "collide", id="offsets-at-a-component-path"),
        pytest.param(
            _payload_at_an_offset_path, "collide", id="alternate-payload-at-an-offset-path"
        ),
        pytest.param(
            _payload_under_a_component, "collide", id="alternate-payload-under-a-component"
        ),
    ],
)
def test_colliding_paths_are_refused(
    tmp_path: Path, build: Callable[[], PoseFile], match: str
) -> None:
    """Every path in the file -- components, alternates, offsets -- must be distinct."""
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)
    pose_file = build()
    with pytest.raises(ValueError, match=match):
        write_pose_file(pose_file, path)
    assert path.read_bytes() == SENTINEL


def test_one_name_names_one_pair_of_offsets() -> None:
    """Alternates sharing a name across components must name the same offset datasets.

    Refused when the file is constructed, so a file the writer cannot write
    cannot exist.
    """
    with pytest.raises(ValueError, match="one encoding"):
        _segmentation_file(
            flag_alternates=lambda s, _: (
                _flags_alternate(
                    s,
                    _index(
                        s,
                        group_offsets_path=f"{PREFIX}/other_offsets",
                        instance_offsets_path=f"{PREFIX}/other_instances",
                    ),
                ),
            )
        )


def test_two_names_sharing_one_index_write_its_offsets_once(tmp_path: Path) -> None:
    """One RaggedIndex under two names is one pair of offset datasets, written once."""
    pose_file = _segmentation_file(
        flag_alternates=lambda s, i: (
            _flags_alternate(s, i),
            _flags_alternate(s, i, name="copy", path=f"{PREFIX}/copy"),
        ),
    )
    path = tmp_path / "shared.h5"
    write_pose_file(pose_file, path, created=CREATED)
    seg = ragged_segmentation()
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5[f"{PREFIX}/copy"][()], seg["flags"])
        np.testing.assert_array_equal(h5[GROUP][()], seg["contour_offsets"])
    # Read back, the two names share one index again, so the file rewrites.
    flags = read_pose_file(path).component(FLAG_ID).alternates
    assert flags[0].index is flags[1].index
