"""``convert_legacy`` converts a ``pose_est_v2`` file: one animal, no instance axis.

ADR 0002's conversion table: the single animal becomes one slot and one
identity, ``slot_occupied`` is any confidence above 0, and ``point_valid`` is
confidence above 0.3 compared in float64, as for v4 and later.
"""

import tracemalloc
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file.convert import (
    LegacyConversionError,
    convert_legacy,
    inspect_legacy,
)
from jabs.io.internal.pose_file.convert.convert import DEFAULT_MAX_MEMORY, block_frames_for
from jabs.io.internal.pose_file.convert.memory import estimate_memory
from jabs.io.internal.pose_file.validate import validate

from .conftest import SAMPLES, generator, legacy, requires_samples

CREATED = "2026-11-20T00:00:00+00:00"
POSE_COMPONENTS = {
    "jabs.pose.points",
    "jabs.pose.confidence",
    "jabs.pose.point_valid",
    "jabs.pose.slot_occupied",
}


def _legacy_data(path: Path) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.float32]]:
    """Read a v2 file's ``points`` and ``confidence``."""
    with h5py.File(path, "r") as h5:
        return h5["poseest/points"][()], h5["poseest/confidence"][()]


def _component(path: Path, component_id: str) -> npt.NDArray[np.generic]:
    """Read one written component's dense data."""
    with h5py.File(path, "r") as h5:
        return h5[component_id.replace(".", "/")][()]


@pytest.fixture
def converted(tmp_path: Path) -> Path:
    """The synthetic v2 file, converted."""
    out = tmp_path / "out.h5"
    convert_legacy(legacy("v2-single.h5"), out, created=CREATED)
    return out


def test_inspect_reads_v2_as_one_instance_and_identity() -> None:
    """A v2 file has no instance axis: one legacy instance, one identity, nothing else."""
    layout = inspect_legacy(legacy("v2-single.h5"))

    assert layout.instance_axis is False
    assert (layout.frames, layout.instances, layout.keypoints) == (generator().V2_FRAMES, 1, 12)
    assert layout.identity == 1
    assert layout.version == ()
    assert not (layout.has_segmentation or layout.has_tracklets or layout.has_embeddings)
    assert not layout.bbox_generated
    assert layout.static_objects == layout.dynamic_objects == layout.dropped == ()


def test_inspect_reads_v4_and_later_with_an_instance_axis() -> None:
    """Files from v4 on keep the instance axis."""
    assert inspect_legacy(legacy("v8-with-centers.h5")).instance_axis is True


def test_converted_v2_is_deep_valid(converted: Path) -> None:
    """The output passes the deep check, with one slot and one identity and only pose data."""
    assert [f for f in validate(converted, deep=True) if f.severity == "error"] == []
    with h5py.File(converted, "r") as h5:
        components = set()
        h5.visititems(
            lambda name, obj: components.add(name.replace("/", "."))
            if isinstance(obj, h5py.Dataset) and name.startswith("jabs/")
            else None
        )
    assert components == POSE_COMPONENTS


def test_report_dimensions(tmp_path: Path) -> None:
    """The dimensions are the frames, one slot and one identity; nothing is dropped or split."""
    report = convert_legacy(legacy("v2-single.h5"), tmp_path / "out.h5", created=CREATED)

    assert report.dimensions == {"frame": generator().V2_FRAMES, "slot": 1, "identity": 1}
    assert report.tail_slots == report.tracklet_splits == report.longterm_mismatches == 0
    assert report.dropped == ()


def test_points_are_flipped_to_xy(converted: Path) -> None:
    """Points become ``(x, y)`` wherever confidence is above 0, and NaN where it is 0."""
    points, confidence = _legacy_data(legacy("v2-single.h5"))
    written = _component(converted, "jabs.pose.points")

    assert written.shape == (*points.shape[:1], 1, 12, 2)
    has = confidence > 0
    np.testing.assert_array_equal(written[:, 0][has], points[..., ::-1][has].astype(np.float32))
    assert np.isnan(written[:, 0][~has]).all()


def test_point_valid_is_exactly_confidence_above_threshold(converted: Path) -> None:
    """``point_valid`` is confidence above 0.3 in float64, so float32 0.3 is valid."""
    _, confidence = _legacy_data(legacy("v2-single.h5"))
    point_valid = _component(converted, "jabs.pose.point_valid")

    np.testing.assert_array_equal(point_valid[:, 0], confidence.astype(np.float64) > 0.3)
    assert point_valid[(*generator().V2_THRESHOLD[:1], 0, generator().V2_THRESHOLD[1])]
    np.testing.assert_array_equal(_component(converted, "jabs.pose.confidence")[:, 0], confidence)


def test_all_zero_frame_is_unoccupied(converted: Path) -> None:
    """The frame with every confidence 0 is the only unoccupied one, and has no points."""
    empty = generator().V2_EMPTY_FRAME
    occupied = _component(converted, "jabs.pose.slot_occupied")

    assert occupied.shape == (generator().V2_FRAMES, 1)
    np.testing.assert_array_equal(np.flatnonzero(~occupied[:, 0]), [empty])
    assert np.isnan(_component(converted, "jabs.pose.points")[empty]).all()


def test_v2_with_a_v4_dataset_is_refused(tmp_path: Path) -> None:
    """A file with no instance axis holding a per-instance dataset is refused, naming it."""
    src = tmp_path / "v2.h5"
    generator().write_v2(src)
    with h5py.File(src, "a") as h5:
        h5["poseest"].create_dataset("id_mask", data=np.zeros(generator().V2_FRAMES, dtype=bool))

    with pytest.raises(LegacyConversionError, match="poseest/id_mask"):
        inspect_legacy(src)


def test_v2_confidence_shape_must_match_points(tmp_path: Path) -> None:
    """``confidence`` must be ``(frame, 12)``, as ``points`` is ``(frame, 12, 2)``."""
    src = tmp_path / "v2.h5"
    generator().write_v2(src)
    with h5py.File(src, "a") as h5:
        del h5["poseest/confidence"]
        h5["poseest"].create_dataset("confidence", data=np.ones((generator().V2_FRAMES, 11)))

    with pytest.raises(LegacyConversionError, match=r"poseest/confidence has shape \(120, 11\)"):
        inspect_legacy(src)


def test_v2_huge_frames_are_refused_before_allocating(tmp_path: Path) -> None:
    """A v2 file declaring billions of frames is refused by the memory estimate."""
    src = tmp_path / "huge.h5"
    frames = 2_000_000_000
    with h5py.File(src, "w") as h5:
        pose = h5.create_group("poseest")
        for name, shape, dtype in (
            ("points", (frames, 12, 2), np.uint16),
            ("confidence", (frames, 12), np.float32),
        ):
            pose.create_dataset(name, shape=shape, dtype=dtype, chunks=(900, *shape[1:]))

    tracemalloc.start()
    try:
        with pytest.raises(LegacyConversionError, match="estimated"):
            convert_legacy(src, tmp_path / "out.h5")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16 * 2**20


def test_v2_estimate_covers_the_measured_peak(tmp_path: Path) -> None:
    """The estimate counts one instance, not twelve, and is above the real peak, within 4x."""
    src = tmp_path / "v2.h5"
    generator().write_v2(src, frames=36_000)
    block_bytes = 4 * 2**20
    layout = inspect_legacy(src)
    with h5py.File(src, "r") as h5:
        estimate = estimate_memory(
            h5, identity=layout.identity, block_frames=block_frames_for(layout, block_bytes)
        )

    tracemalloc.start()
    try:
        convert_legacy(src, tmp_path / "out.h5", block_bytes=block_bytes)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak <= estimate <= 4 * peak
    assert estimate < DEFAULT_MAX_MEMORY / 100


@requires_samples
def test_public_v2_sample_converts(tmp_path: Path) -> None:
    """The repository's public v2 sample converts to a deep-valid single-slot file."""
    source = SAMPLES / "sample_pose_est_v2.h5"
    out = tmp_path / "out.h5"
    report = convert_legacy(source, out, created=CREATED)

    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []
    frames = _legacy_data(source)[0].shape[0]
    assert report.dimensions == {"frame": frames, "slot": 1, "identity": 1}
