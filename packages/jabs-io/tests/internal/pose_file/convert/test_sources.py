"""Legacy pose and identity data become block-sourced components, placed by the plan."""

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.convert import (
    LegacyLayout,
    PlacementPlan,
    inspect_legacy,
    plan_placement,
)
from jabs.io.internal.pose_file.convert.inspect import BLOCK_FRAMES
from jabs.io.internal.pose_file.convert.objects import object_components
from jabs.io.internal.pose_file.convert.sources import pose_components
from jabs.io.internal.pose_file.reader import read_pose_file
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12, jabs_mouse12
from jabs.io.internal.pose_file.types import Component, PoseFile, VideoInfo
from jabs.io.internal.pose_file.writer import write_pose_file

from .conftest import generator, legacy


def _setup(path: Path) -> tuple[LegacyLayout, PlacementPlan]:
    """Inspect and plan one legacy file, closing it before anything else reads it."""
    layout = inspect_legacy(path)
    with h5py.File(path, "r") as h5:
        return layout, plan_placement(h5, layout)


def _components(path: Path) -> dict[str, Component]:
    """Return one legacy file's pose components, keyed by id."""
    layout, plan = _setup(path)
    components = pose_components(path, layout, plan, block_frames=BLOCK_FRAMES)
    return {c.id: c for c in components}


def _materialize(component: Component) -> npt.NDArray[np.generic]:
    """Return a component's whole payload, assembling a block source's blocks."""
    data = component.data
    if not isinstance(data, BlockSource):
        return np.asarray(data)
    out = np.empty(data.shape, dtype=data.dtype)
    for start, block in data.blocks():
        assert block.dtype == data.dtype
        out[start : start + block.shape[0]] = block
    return out


def _arrays(path: Path) -> dict[str, npt.NDArray[np.generic]]:
    """Return every pose component's payload of one legacy file, keyed by id."""
    return {cid: _materialize(c) for cid, c in _components(path).items()}


def _legacy(path: Path, name: str) -> npt.NDArray[np.generic]:
    """Read one whole ``poseest`` dataset."""
    with h5py.File(path, "r") as h5:
        return h5["poseest"][name][...]


def _edge_location(name: str) -> tuple[int, int, int | None]:
    """Return a single-frame, single-instance EDGE case's (frame, instance, keypoint)."""
    case = generator().EDGE[name]
    assert len(case.frames) == len(case.instances) == 1
    return case.frames[0], case.instances[0], case.keypoint


def test_components_in_order() -> None:
    """The pose components come in the brief's order, then identity."""
    ids = list(_components(legacy("v8-with-centers.h5")))
    assert ids == [
        "jabs.pose.points",
        "jabs.pose.confidence",
        "jabs.pose.point_valid",
        "jabs.pose.slot_occupied",
        "jabs.pose.bbox",
        "jabs.pose.tracklet_id",
        "jabs.identity.embeddings",
        "jabs.identity.centers",
    ]


def test_declarations() -> None:
    """Each component declares the catalog's axes, dtype, units and missing policy."""
    components = _components(legacy("v8-with-centers.h5"))
    points = components["jabs.pose.points"]
    assert points.axes == ("frame", "slot", "keypoint", "coord")
    assert (points.dtype, points.units, points.coord_order) == ("float32", "pixel", "xy")
    assert points.skeleton == JABS_MOUSE12
    assert dict(points.missing) == {"policy": "nan"}
    for cid in ("jabs.pose.confidence", "jabs.pose.point_valid", "jabs.pose.slot_occupied"):
        assert dict(components[cid].missing) == {"policy": "none"}
    assert components["jabs.pose.confidence"].dtype == "float32"
    assert components["jabs.pose.point_valid"].dtype == "bool"
    assert components["jabs.pose.slot_occupied"].dtype == "bool"
    masked = {"policy": "mask", "mask": "jabs.pose.slot_occupied"}
    for cid in ("jabs.pose.bbox", "jabs.pose.tracklet_id", "jabs.identity.embeddings"):
        assert dict(components[cid].missing) == masked
    assert components["jabs.pose.tracklet_id"].dtype == "uint32"
    assert components["jabs.identity.embeddings"].dtype == "float32"
    centers = components["jabs.identity.centers"]
    assert centers.axes == ("identity", "embedding")
    assert centers.dtype == "float32"
    assert all(c.provenance is None for c in components.values())
    for cid, component in components.items():
        if cid != "jabs.identity.centers":
            assert isinstance(component.data, BlockSource), cid


def test_points_are_flipped_and_placed() -> None:
    """A placed instance's points are its legacy (y, x) points flipped, as float32."""
    path = legacy("v8-edge-cases.h5")
    _, plan = _setup(path)
    points = _arrays(path)["jabs.pose.points"]
    stored = _legacy(path, "points")
    frame, instance = 5, 0
    slot = int(plan.slot_of[frame, instance])
    assert slot >= 0
    assert points.dtype == np.float32
    np.testing.assert_array_equal(
        points[frame, slot], stored[frame, instance, :, ::-1].astype(np.float32)
    )


def test_nan_only_at_zero_confidence() -> None:
    """NaN marks exactly no coordinate: zero confidence or a NaN source coordinate.

    The brief states this as ``isnan == (confidence == 0)``; the controller's
    ruling adds a source ``NaN`` with confidence above 0, which the corpus
    carries at ``nan_with_confidence``, so the expectation covers both.
    """
    path = legacy("v8-edge-cases.h5")
    _, plan = _setup(path)
    arrays = _arrays(path)
    points, confidence = arrays["jabs.pose.points"], arrays["jabs.pose.confidence"]
    point_valid = arrays["jabs.pose.point_valid"]
    stored = _legacy(path, "points")

    source_nan = np.zeros(confidence.shape, dtype=bool)
    frame, instance = np.nonzero(plan.slot_of >= 0)
    source_nan[frame, plan.slot_of[frame, instance]] = np.isnan(stored[frame, instance]).any(-1)
    occupied = arrays["jabs.pose.slot_occupied"]
    expected = (confidence == 0) | source_nan
    np.testing.assert_array_equal(np.isnan(points).any(-1)[occupied], expected[occupied])
    # Both coordinates go together.
    np.testing.assert_array_equal(np.isnan(points).any(-1), np.isnan(points).all(-1))

    frame, instance, keypoint = _edge_location("nan_with_confidence")
    slot = int(plan.slot_of[frame, instance])
    assert np.isnan(points[frame, slot, keypoint]).all()
    assert confidence[frame, slot, keypoint] > 0.3
    assert point_valid[frame, slot, keypoint]

    frame, instance, keypoint = _edge_location("low_confidence")
    slot = int(plan.slot_of[frame, instance])
    assert 0 < confidence[frame, slot, keypoint] <= 0.3
    assert np.isfinite(points[frame, slot, keypoint]).all()
    assert not point_valid[frame, slot, keypoint]


def test_float32_point_three_is_valid() -> None:
    """A stored float32 0.3 is 0.30000001192, above the threshold in float64."""
    path = legacy("v8-edge-cases.h5")
    _, plan = _setup(path)
    arrays = _arrays(path)
    frame, instance, keypoint = _edge_location("threshold_exact")
    slot = int(plan.slot_of[frame, instance])
    assert arrays["jabs.pose.confidence"][frame, slot, keypoint] == np.float32(0.3)
    assert arrays["jabs.pose.point_valid"][frame, slot, keypoint]


def test_uint16_points_are_nan_only_at_zero_confidence(tmp_path: Path) -> None:
    """Integer legacy points become float32, NaN where confidence is 0 (Review Focus 5)."""
    gen = generator()
    frame, instance, keypoint = 7, 1, 4

    def zero_one_confidence(h5: h5py.File) -> None:
        """Zero one keypoint's confidence, leaving its stored coordinate."""
        h5["poseest/confidence"][frame, instance, keypoint] = 0

    path = tmp_path / "v7.h5"
    gen.write_legacy(path, **gen.SMALL, version=gen.V7, seg=False, edits=zero_one_confidence)
    stored = _legacy(path, "points")
    assert stored.dtype == np.uint16
    assert (stored[frame, instance, keypoint] != 0).all()

    _, plan = _setup(path)
    arrays = _arrays(path)
    points, confidence = arrays["jabs.pose.points"], arrays["jabs.pose.confidence"]
    assert points.dtype == np.float32
    slot = int(plan.slot_of[frame, instance])
    assert np.isnan(points[frame, slot, keypoint]).all()
    occupied = arrays["jabs.pose.slot_occupied"]
    np.testing.assert_array_equal(np.isnan(points).any(-1)[occupied], (confidence == 0)[occupied])
    np.testing.assert_array_equal(
        points[frame, slot, keypoint + 1],
        stored[frame, instance, keypoint + 1, ::-1].astype(np.float32),
    )


def test_empty_slots() -> None:
    """An empty slot has NaN points, zero confidence, false masks, NaN bbox and zeros."""
    gen = generator()
    path = legacy("v8-edge-cases.h5")
    _, plan = _setup(path)
    arrays = _arrays(path)
    # The tracklet_gap instance is empty, so its identity slot holds nothing.
    frame = gen.EDGE["tracklet_gap"].frames[0]
    (instance,) = gen.EDGE["tracklet_gap"].instances
    assert plan.slot_of[frame, instance] == -1
    slot = gen.clean_embed_id(instance, gen.EDGE_IDENTITIES) - 1
    assert slot not in plan.slot_of[frame]

    assert np.isnan(arrays["jabs.pose.points"][frame, slot]).all()
    assert (arrays["jabs.pose.confidence"][frame, slot] == 0).all()
    assert not arrays["jabs.pose.point_valid"][frame, slot].any()
    assert not arrays["jabs.pose.slot_occupied"][frame, slot]
    assert np.isnan(arrays["jabs.pose.bbox"][frame, slot]).all()
    assert arrays["jabs.pose.tracklet_id"][frame, slot] == 0
    assert (arrays["jabs.identity.embeddings"][frame, slot] == 0).all()


def test_zero_confidence_instance_is_unoccupied_but_keeps_its_bbox() -> None:
    """A placed instance with every confidence 0 is unoccupied, with its own bbox kept."""
    path = legacy("v8-edge-cases.h5")
    _, plan = _setup(path)
    arrays = _arrays(path)
    frame, instance, _ = _edge_location("zero_confidence_contours")
    slot = int(plan.slot_of[frame, instance])
    assert slot >= 0
    assert not arrays["jabs.pose.slot_occupied"][frame, slot]
    np.testing.assert_array_equal(
        arrays["jabs.pose.bbox"][frame, slot], _legacy(path, "bbox")[frame, instance]
    )


def test_bbox_carried_unflipped_only_when_generated() -> None:
    """bbox is the legacy (x, y) bbox at the placed slot, and absent when untrusted."""
    path = legacy("v8-with-centers.h5")
    _, plan = _setup(path)
    bbox = _arrays(path)["jabs.pose.bbox"]
    stored = _legacy(path, "bbox")
    frame, instance = np.nonzero(plan.slot_of >= 0)
    np.testing.assert_array_equal(
        bbox[frame, plan.slot_of[frame, instance]], stored[frame, instance]
    )
    assert bbox.dtype == np.float32

    assert "jabs.pose.bbox" not in _components(legacy("v8-without-centers.h5"))


def test_tracklets_and_identity() -> None:
    """Tracklets come from the plan; embeddings and centers from the legacy file."""
    path = legacy("v8-with-centers.h5")
    _, plan = _setup(path)
    arrays = _arrays(path)
    assert plan.tracklet_id is not None
    np.testing.assert_array_equal(arrays["jabs.pose.tracklet_id"], plan.tracklet_id)
    frame, instance = np.nonzero(plan.slot_of >= 0)
    np.testing.assert_array_equal(
        arrays["jabs.identity.embeddings"][frame, plan.slot_of[frame, instance]],
        _legacy(path, "identity_embeds")[frame, instance],
    )
    np.testing.assert_array_equal(
        arrays["jabs.identity.centers"],
        _legacy(path, "instance_id_center").astype(np.float32),
    )


def test_without_centers_omits_identity_and_tracklets() -> None:
    """A file without tracklets, embeddings or centers yields none of them."""
    ids = set(_components(legacy("v8-without-centers.h5")))
    assert ids == {
        "jabs.pose.points",
        "jabs.pose.confidence",
        "jabs.pose.point_valid",
        "jabs.pose.slot_occupied",
    }


def test_blocks_are_multiples_of_900(tmp_path: Path) -> None:
    """Every frame-axis block but the last holds a multiple of 900 frames."""
    gen = generator()
    path = tmp_path / "long.h5"
    gen.write_legacy(path, frames=2000, instances=3, identities=3, version=gen.V8, seg=False)
    components = _components(path)
    for cid, component in components.items():
        if component.axes[0] != "frame":
            continue
        assert isinstance(component.data, BlockSource), cid
        rows = [block.shape[0] for _, block in component.data.blocks()]
        assert sum(rows) == 2000, cid
        assert len(rows) > 1, cid
        assert all(n % 900 == 0 for n in rows[:-1]), (cid, rows)


def test_refuses_block_frames_off_the_900_grid() -> None:
    """A block size that is not a multiple of 900 is refused."""
    path = legacy("v8-with-centers.h5")
    layout, plan = _setup(path)
    with pytest.raises(ValueError, match="900"):
        pose_components(path, layout, plan, block_frames=450)


@pytest.mark.parametrize(
    "name",
    ["v8-with-centers.h5", "v8-without-centers.h5", "v8-edge-cases.h5", "v7-dynamic-objects.h5"],
)
def test_components_write_after_the_handle_closes(tmp_path: Path, name: str) -> None:
    """The components write, pass the deep check and read back, the file opened per source."""
    path = legacy(name)
    layout, plan = _setup(path)
    objects = object_components(path, layout)
    components = (*pose_components(path, layout, plan, block_frames=BLOCK_FRAMES), *objects)
    pose_file = PoseFile(
        dimensions={"frame": layout.frames, "slot": plan.slots, "identity": layout.identity},
        video=VideoInfo(frame_count=layout.frames),
        skeletons={JABS_MOUSE12: jabs_mouse12()},
        components=components,
    )
    out = tmp_path / "out.h5"
    write_pose_file(pose_file, out, deep=True)
    written = {c.id: c for c in read_pose_file(out).components}
    expected = {**_arrays(path), **{c.id: np.asarray(c.data) for c in objects}}
    assert set(written) == set(expected)
    for cid, array in expected.items():
        np.testing.assert_array_equal(np.asarray(written[cid].data), array, err_msg=cid)
