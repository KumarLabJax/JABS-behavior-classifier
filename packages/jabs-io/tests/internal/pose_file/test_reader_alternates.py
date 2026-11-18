"""The reader keeps alternates, so a read-then-write keeps them.

Before this, ``read_pose_file`` skipped alternates, and writing back what it
read dropped them without a word. Alternates of one name share their offsets,
so the reader hands them one :class:`RaggedIndex`, which is what lets the
writer write those offsets once.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file import validate, write_pose_file
from jabs.io.internal.pose_file.reader import PoseFileError, read_component, read_pose_file
from jabs.io.internal.pose_file.types import Component, PoseFile

DATA = Path(__file__).parents[2] / "data" / "pose-format"
FIXTURE = DATA / "valid-ragged-alternate.h5"
CONTOURS_ID = "jabs.segmentation.contours"
FLAG_ID = "jabs.segmentation.external_flag"
PREFIX = "/jabs/segmentation/ragged"
CREATED = "2026-11-18T00:00:00Z"


def _component(pose_file: PoseFile, component_id: str) -> Component:
    """The component with this id."""
    return next(c for c in pose_file.components if c.id == component_id)


def _rewrite_manifest(path: Path, mutate: Callable[[dict], object]) -> None:
    """Replace a file's manifest with a mutated copy."""
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5["manifest"][()])
        mutate(manifest)
        del h5["manifest"]
        h5.create_dataset(
            "manifest", data=json.dumps(manifest), dtype=h5py.string_dtype(encoding="utf-8")
        )


def test_read_keeps_alternates() -> None:
    """Both alternates come back, at their levels, sharing one index."""
    pose_file = read_pose_file(FIXTURE)
    contours = _component(pose_file, CONTOURS_ID)
    flags = _component(pose_file, FLAG_ID)

    assert len(contours.alternates) == 1
    points = contours.alternates[0]
    assert points.name == "ragged"
    assert points.level == "leaf"
    with h5py.File(FIXTURE, "r") as h5:
        np.testing.assert_array_equal(points.data, h5[f"{PREFIX}/contours"][()])
        np.testing.assert_array_equal(
            points.index.group_offsets, h5[f"{PREFIX}/contour_offsets"][()]
        )
        np.testing.assert_array_equal(
            points.index.instance_offsets, h5[f"{PREFIX}/instance_offsets"][()]
        )
    assert points.index.group_offsets.dtype == np.uint64

    assert len(flags.alternates) == 1
    assert flags.alternates[0].level == "group"
    assert flags.alternates[0].index is points.index


def test_read_then_write_keeps_alternates(tmp_path: Path) -> None:
    """Writing back what was read keeps both alternates and writes the offsets once."""
    out = tmp_path / "round-trip.h5"
    write_pose_file(read_pose_file(FIXTURE), out, created=CREATED)

    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []
    with h5py.File(out, "r") as h5:
        manifest = json.loads(h5["manifest"][()])
        offsets: list[str] = []

        def _collect(name: str, node: object) -> None:
            """Note every offsets dataset."""
            if isinstance(node, h5py.Dataset) and name.endswith(
                ("contour_offsets", "instance_offsets")
            ):
                offsets.append(name)

        h5.visititems(_collect)
    declared = {
        spec["id"]: [alt["name"] for alt in spec.get("alternates", [])]
        for spec in manifest["components"]
    }
    assert declared[CONTOURS_ID] == ["ragged"]
    assert declared[FLAG_ID] == ["ragged"]
    assert sorted(offsets) == [
        "jabs/segmentation/ragged/contour_offsets",
        "jabs/segmentation/ragged/instance_offsets",
    ]


def test_read_component_ignores_alternates() -> None:
    """Reading one component still returns its dense baseline, nothing more."""
    payload = read_component(FIXTURE, CONTOURS_ID)
    assert isinstance(payload, np.ndarray)
    with h5py.File(FIXTURE, "r") as h5:
        np.testing.assert_array_equal(payload, h5["/jabs/segmentation/contours"][()])


def test_an_undecodable_alternate_kind_is_refused(tmp_path: Path) -> None:
    """An alternate the reader cannot decode is refused, never silently dropped."""
    path = tmp_path / "rle-alternate.h5"
    shutil.copy(FIXTURE, path)

    def _make_rle(manifest: dict) -> None:
        """Re-declare the contours' alternate as a schema-valid rle encoding."""
        spec = next(s for s in manifest["components"] if s["id"] == CONTOURS_ID)
        alternate = spec["alternates"][0]
        alternate.pop("units")
        alternate.pop("coord_order")
        alternate.update(
            axes=["run"],
            dtype="uint64",
            shape=[25],
            encoding={"kind": "rle", "instance_offsets": f"{PREFIX}/instance_offsets"},
        )

    _rewrite_manifest(path, _make_rle)
    with pytest.raises(NotImplementedError, match=f"{CONTOURS_ID}:alternate:ragged"):
        read_pose_file(path)


def test_missing_offsets_are_a_pose_file_error(tmp_path: Path) -> None:
    """Offsets the manifest names but the file lacks are the file disagreeing with itself."""
    path = tmp_path / "no-offsets.h5"
    shutil.copy(FIXTURE, path)
    with h5py.File(path, "r+") as h5:
        del h5[f"{PREFIX}/contour_offsets"]
    with pytest.raises(PoseFileError, match=f"{CONTOURS_ID}:alternate:ragged"):
        read_pose_file(path)
