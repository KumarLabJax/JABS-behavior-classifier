"""Legacy static and dynamic objects become uniform object components."""

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file.convert import LegacyConversionError, inspect_legacy
from jabs.io.internal.pose_file.convert.objects import object_components
from jabs.io.internal.pose_file.types import Component

from .conftest import generator, legacy


def _objects(path: Path) -> dict[str, Component]:
    """Return one legacy file's object components, keyed by id."""
    return {c.id: c for c in object_components(path, inspect_legacy(path))}


def _stored(path: Path, name: str) -> npt.NDArray[np.generic]:
    """Read one whole legacy dataset by its path."""
    with h5py.File(path, "r") as h5:
        return h5[name][...]


def test_static_objects() -> None:
    """corners are unflipped; lixit and food_hopper are flipped; every one is O x P x 2."""
    path = legacy("v7-dynamic-objects.h5")
    objects = _objects(path)
    for name in ("corners", "lixit", "food_hopper"):
        component = objects[f"jabs.static_objects.{name}"]
        assert component.axes == ("object", "point", "coord")
        assert (component.dtype, component.units, component.coord_order) == (
            "float32",
            "pixel",
            "xy",
        )
        assert dict(component.missing) == {"policy": "none"}
        assert component.description

    corners = objects["jabs.static_objects.corners"].data
    assert corners.shape == (4, 1, 2)
    np.testing.assert_array_equal(
        corners[:, 0], _stored(path, "static_objects/corners").astype(np.float32)
    )
    lixit = objects["jabs.static_objects.lixit"].data
    assert lixit.shape == (2, 3, 2)
    np.testing.assert_array_equal(lixit, _stored(path, "static_objects/lixit")[..., ::-1])
    hopper = objects["jabs.static_objects.food_hopper"].data
    assert hopper.shape == (4, 1, 2)
    np.testing.assert_array_equal(
        hopper[:, 0], _stored(path, "static_objects/food_hopper")[:, ::-1].astype(np.float32)
    )


def test_two_dimensional_lixit_gets_one_point(tmp_path: Path) -> None:
    """A (L, 2) lixit becomes (L, 1, 2), flipped."""
    gen = generator()
    lixit = np.array([[120, 380], [640, 702]], dtype=np.float32)

    def add_lixit(h5: h5py.File) -> None:
        """Store a single-keypoint lixit."""
        h5.create_dataset("static_objects/lixit", data=lixit)

    path = tmp_path / "lixit.h5"
    gen.write_legacy(path, **gen.SMALL, version=gen.V7, seg=False, edits=add_lixit)
    data = _objects(path)["jabs.static_objects.lixit"].data
    assert data.shape == (2, 1, 2)
    np.testing.assert_array_equal(data[:, 0], lixit[:, ::-1])


def test_dynamic_objects() -> None:
    """fecal_boli without axis_order is flipped, indexed by its samples, and gets P=1."""
    path = legacy("v7-dynamic-objects.h5")
    objects = _objects(path)
    prefix = "jabs.dynamic_objects.fecal_boli"
    index, points, counts = (objects[f"{prefix}.{s}"] for s in ("frame_index", "points", "counts"))

    assert index.axes == ("sample",)
    assert index.sparse_index == index.id
    assert index.dtype == "uint32"
    np.testing.assert_array_equal(
        index.data, _stored(path, "dynamic_objects/fecal_boli/sample_indices")
    )

    assert counts.axes == ("sample",)
    assert counts.sparse_index == index.id
    assert counts.dtype == "uint32"
    np.testing.assert_array_equal(counts.data, _stored(path, "dynamic_objects/fecal_boli/counts"))

    assert points.axes == ("sample", "object", "point", "coord")
    assert (points.dtype, points.units, points.coord_order) == ("float32", "pixel", "xy")
    assert points.sparse_index == index.id
    assert dict(points.missing) == {"policy": "length", "length": counts.id}
    stored = _stored(path, "dynamic_objects/fecal_boli/points")
    assert points.data.shape == (*stored.shape[:2], 1, 2)
    np.testing.assert_array_equal(points.data[:, :, 0], stored[..., ::-1].astype(np.float32))


def test_dynamic_objects_in_xy_are_not_flipped(tmp_path: Path) -> None:
    """A dynamic object stored with axis_order "xy" keeps its order."""
    gen = generator()
    stored = np.array([[[[210, 610], [230, 655]]], [[[205, 612], [228, 650]]]], dtype=np.uint16)

    def add_boli(h5: h5py.File) -> None:
        """Store a 4-D fecal_boli in (x, y) order."""
        boli = h5.create_group("dynamic_objects/fecal_boli")
        boli.create_dataset("points", data=stored)
        boli["points"].attrs["axis_order"] = "xy"
        boli.create_dataset("counts", data=np.array([1, 1], dtype=np.uint8))
        boli.create_dataset("sample_indices", data=np.array([0, 30], dtype=np.uint32))

    path = tmp_path / "xy.h5"
    gen.write_legacy(path, **gen.SMALL, version=gen.V7, seg=False, edits=add_boli)
    points = _objects(path)["jabs.dynamic_objects.fecal_boli.points"].data
    assert points.shape == stored.shape
    np.testing.assert_array_equal(points, stored.astype(np.float32))


def test_no_objects() -> None:
    """A v8 file with only corners yields corners alone."""
    ids = set(_objects(legacy("v8-with-centers.h5")))
    assert ids == {"jabs.static_objects.corners"}


@pytest.mark.parametrize(
    ("member", "stored", "match"),
    [
        pytest.param(
            "sample_indices",
            np.array([0.0, 30.0]),
            r"sample_indices has dtype float64",
            id="float",
        ),
        pytest.param(
            "counts", np.array([1, -1], dtype=np.int8), r"counts holds -1", id="negative"
        ),
        pytest.param(
            "sample_indices",
            np.array([0, 2**32], dtype=np.uint64),
            # Inspection refuses it first, as a frame past the video's last.
            r"sample_indices holds frame 4294967296",
            id="above-uint32",
        ),
    ],
)
def test_dynamic_object_indexes_that_would_wrap_are_refused(
    tmp_path: Path, member: str, stored: npt.NDArray[np.generic], match: str
) -> None:
    """``sample_indices`` and ``counts`` that are not uint32-sized integers are refused by name."""
    gen = generator()

    def add_boli(h5: h5py.File) -> None:
        """Store a fecal_boli object with one bad index dataset."""
        boli = h5.create_group("dynamic_objects/fecal_boli")
        boli.create_dataset("points", data=np.zeros((2, 3, 2), dtype=np.uint16))
        members = {
            "counts": np.array([1, 1], dtype=np.uint8),
            "sample_indices": np.array([0, 30], dtype=np.uint32),
        }
        members[member] = stored
        for name, data in members.items():
            boli.create_dataset(name, data=data)

    path = tmp_path / "boli.h5"
    gen.write_legacy(path, **gen.SMALL, version=gen.V7, seg=False, edits=add_boli)
    with pytest.raises(LegacyConversionError, match=match) as caught:
        _objects(path)
    assert f"dynamic_objects/fecal_boli/{member}" in str(caught.value)
    assert "boli.h5" in str(caught.value)
