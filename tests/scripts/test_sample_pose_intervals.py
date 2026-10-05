"""Tests for sample_pose_intervals clipping logic."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.scripts.cli.sample_pose_intervals import _sample_one


def _write_pose_h5(
    path: Path,
    frame_count: int,
    dynamic_objects: dict | None = None,
) -> None:
    """Write a minimal pose v6 HDF5 file for testing.

    Args:
        path: Destination path.
        frame_count: Number of frames to write.
        dynamic_objects: Optional dict mapping object name to a dict with
            keys ``sample_indices``, ``counts``, ``points``, and optionally
            ``axis_order``.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "poseest/points",
            data=np.zeros((frame_count, 1, 12, 2), dtype=np.float32),
        )
        f.create_dataset(
            "poseest/confidence",
            data=np.zeros((frame_count, 1, 12), dtype=np.float32),
        )
        if dynamic_objects:
            for obj_name, obj_data in dynamic_objects.items():
                grp = f.create_group(f"dynamic_objects/{obj_name}")
                grp.create_dataset("sample_indices", data=obj_data["sample_indices"])
                grp.create_dataset("counts", data=obj_data["counts"])
                pts_ds = grp.create_dataset("points", data=obj_data["points"])
                if "axis_order" in obj_data:
                    pts_ds.attrs["axis_order"] = obj_data["axis_order"]


def _run(
    root: Path,
    out_dir: Path,
    frame_count: int,
    out_frame_count: int,
    start_frame: int | None = None,
    dynamic_objects: dict | None = None,
) -> Path | None:
    """Write a pose file, call _sample_one, and return the output HDF5 path if created."""
    pose_path = root / "video_pose_est_v6.h5"
    _write_pose_h5(pose_path, frame_count, dynamic_objects)
    out_dir.mkdir(exist_ok=True)

    _sample_one(
        vid_filename="video.mp4",
        root_dir=root,
        out_dir=out_dir,
        out_frame_count=out_frame_count,
        start_frame=start_frame,
        only_pose=True,
    )

    outputs = list(out_dir.glob("*.h5"))
    return outputs[0] if outputs else None


# ---------------------------------------------------------------------------
# Frame selection and bounds
# ---------------------------------------------------------------------------


def test_clip_exact_length_produces_output(tmp_path: Path) -> None:
    """frame_count == out_frame_count (max_start == 0) should succeed."""
    out = _run(tmp_path, tmp_path / "out", frame_count=50, out_frame_count=50, start_frame=1)
    assert out is not None
    with h5py.File(out, "r") as f:
        assert f["poseest/points"].shape[0] == 50


def test_clip_too_short_skipped(tmp_path: Path) -> None:
    """frame_count < out_frame_count should produce no output."""
    out = _run(tmp_path, tmp_path / "out", frame_count=49, out_frame_count=50)
    assert out is None


def test_start_frame_at_max_valid(tmp_path: Path) -> None:
    """start_frame == max_start + 1 (1-based) is the last valid start."""
    # frame_count=100, out_frame_count=30 → max_start=70 (0-based) → start_frame=71 (1-based)
    out = _run(tmp_path, tmp_path / "out", frame_count=100, out_frame_count=30, start_frame=71)
    assert out is not None
    with h5py.File(out, "r") as f:
        assert f["poseest/points"].shape[0] == 30


def test_start_frame_too_large_skipped(tmp_path: Path) -> None:
    """start_frame beyond max_start should produce no output."""
    # frame_count=100, out_frame_count=30 → max valid start_frame=71, so 72 is too large
    out = _run(tmp_path, tmp_path / "out", frame_count=100, out_frame_count=30, start_frame=72)
    assert out is None


def test_start_frame_slices_correct_frames(tmp_path: Path) -> None:
    """Output confidence values should correspond to the correct source frames."""
    frame_count = 100
    out_frame_count = 10
    start_frame = 41  # 1-based → 0-based start = 40

    pose_path = tmp_path / "video_pose_est_v6.h5"
    with h5py.File(pose_path, "w") as f:
        conf = np.arange(frame_count, dtype=np.float32).reshape(frame_count, 1, 1)
        f.create_dataset(
            "poseest/points", data=np.zeros((frame_count, 1, 12, 2), dtype=np.float32)
        )
        f.create_dataset(
            "poseest/confidence", data=np.broadcast_to(conf, (frame_count, 1, 12)).copy()
        )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _sample_one(
        vid_filename="video.mp4",
        root_dir=tmp_path,
        out_dir=out_dir,
        out_frame_count=out_frame_count,
        start_frame=start_frame,
        only_pose=True,
    )

    outputs = list(out_dir.glob("*.h5"))
    assert len(outputs) == 1
    with h5py.File(outputs[0], "r") as f:
        conf_out = f["poseest/confidence"][:, 0, 0]
    expected = np.arange(40, 50, dtype=np.float32)
    np.testing.assert_array_equal(conf_out, expected)


@pytest.mark.parametrize("pick", [0, 1], ids=["lowest_start", "highest_start"])
def test_random_start_stays_in_valid_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pick: int
) -> None:
    """The random start is drawn from [0, max_start], so even the extremes give a full clip.

    random.randint is replaced by a stub that records its bounds and returns one of
    them, which tests the range deterministically; an off-by-one upper bound would
    otherwise be hit by only a small fraction of random draws.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest fixture used to replace random.randint.
        pick: Which bound the stub returns: 0 for the lower, 1 for the upper.
    """
    bounds: list[tuple[int, int]] = []

    def fake_randint(low: int, high: int) -> int:
        bounds.append((low, high))
        return (low, high)[pick]

    monkeypatch.setattr("jabs.scripts.cli.sample_pose_intervals.random.randint", fake_randint)

    # frame_count=100, out_frame_count=30 -> max_start=70 (0-based)
    out = _run(tmp_path, tmp_path / "out", frame_count=100, out_frame_count=30)

    assert bounds == [(0, 70)]
    assert out is not None
    with h5py.File(out, "r") as f:
        assert f["poseest/points"].shape[0] == 30


# ---------------------------------------------------------------------------
# Dynamic object sample_indices rebasing
# ---------------------------------------------------------------------------


def _make_dyn(sample_indices, counts=None, n_keypoints: int = 1) -> dict:
    """Build a minimal dynamic object dict for _write_pose_h5.

    Each sample is made identifiable so a test can tell which source samples were
    carried into the clip, and that ``counts`` and ``points`` still line up with
    ``sample_indices``: ``counts`` defaults to 1, 2, 3, ... (the sample's position in
    the source) and every coordinate of a sample's ``points`` row is its own source
    frame index.
    """
    n = len(sample_indices)
    if counts is None:
        counts = np.arange(1, n + 1, dtype=np.int64)
    points = np.broadcast_to(
        np.asarray(sample_indices, dtype=np.float64).reshape(n, 1, 1, 1),
        (n, 1, n_keypoints, 2),
    ).copy()
    return {
        "sample_indices": np.array(sample_indices, dtype=np.int64),
        "counts": np.array(counts, dtype=np.int64),
        "points": points,
        "axis_order": "xy",
    }


def test_dynamic_objects_all_within_rebased(tmp_path: Path) -> None:
    """Samples entirely within [start, stop) are rebased by subtracting start."""
    # clip: start=10 (0-based), out_frame_count=20 → stop=30
    dyn = {"boli": _make_dyn([10, 15, 25, 29])}
    out = _run(
        tmp_path,
        tmp_path / "out",
        frame_count=50,
        out_frame_count=20,
        start_frame=11,
        dynamic_objects=dyn,
    )
    assert out is not None
    with h5py.File(out, "r") as f:
        si = f["dynamic_objects/boli/sample_indices"][:]
        counts = f["dynamic_objects/boli/counts"][:]
        points = f["dynamic_objects/boli/points"][:]
    np.testing.assert_array_equal(si, [0, 5, 15, 19])
    # counts and points stay attached to their samples (source positions 1-4, frames 10-29)
    np.testing.assert_array_equal(counts, [1, 2, 3, 4])
    np.testing.assert_array_equal(points[:, 0, 0, 0], [10, 15, 25, 29])


def test_dynamic_objects_boundary_before_start_clamped_to_zero(tmp_path: Path) -> None:
    """Last sample before start is included and clamped to clip-relative frame 0."""
    # clip: start=10, stop=30; sample at 3 is before start → clamped to 0
    dyn = {"boli": _make_dyn([3, 15, 25])}
    out = _run(
        tmp_path,
        tmp_path / "out",
        frame_count=50,
        out_frame_count=20,
        start_frame=11,
        dynamic_objects=dyn,
    )
    assert out is not None
    with h5py.File(out, "r") as f:
        si = f["dynamic_objects/boli/sample_indices"][:]
        counts = f["dynamic_objects/boli/counts"][:]
        points = f["dynamic_objects/boli/points"][:]
    np.testing.assert_array_equal(si, [0, 5, 15])
    # the carried-in boundary sample keeps its own count and points (source frame 3)
    np.testing.assert_array_equal(counts, [1, 2, 3])
    np.testing.assert_array_equal(points[:, 0, 0, 0], [3, 15, 25])


def test_dynamic_objects_only_last_before_included(tmp_path: Path) -> None:
    """Only the last sample before start is included, not earlier ones."""
    # samples at 2, 5, 8 are before start=10; only 8 should appear (clamped to 0)
    dyn = {"boli": _make_dyn([2, 5, 8, 15])}
    out = _run(
        tmp_path,
        tmp_path / "out",
        frame_count=50,
        out_frame_count=20,
        start_frame=11,
        dynamic_objects=dyn,
    )
    assert out is not None
    with h5py.File(out, "r") as f:
        si = f["dynamic_objects/boli/sample_indices"][:]
        counts = f["dynamic_objects/boli/counts"][:]
        points = f["dynamic_objects/boli/points"][:]
    # counts[2] corresponds to sample at index 8, then counts[3] to sample at 15
    np.testing.assert_array_equal(si, [0, 5])
    np.testing.assert_array_equal(counts, [3, 4])
    # the carried sample is the one at frame 8 (the most recent state), not 2 or 5
    np.testing.assert_array_equal(points[:, 0, 0, 0], [8, 15])


def test_dynamic_objects_no_samples_before_clip(tmp_path: Path) -> None:
    """When no samples precede the clip, no boundary entry is added."""
    dyn = {"boli": _make_dyn([15, 25])}
    out = _run(
        tmp_path,
        tmp_path / "out",
        frame_count=50,
        out_frame_count=20,
        start_frame=11,
        dynamic_objects=dyn,
    )
    assert out is not None
    with h5py.File(out, "r") as f:
        si = f["dynamic_objects/boli/sample_indices"][:]
        counts = f["dynamic_objects/boli/counts"][:]
        points = f["dynamic_objects/boli/points"][:]
    np.testing.assert_array_equal(si, [5, 15])
    np.testing.assert_array_equal(counts, [1, 2])
    np.testing.assert_array_equal(points[:, 0, 0, 0], [15, 25])


def test_dynamic_objects_all_after_stop_produces_empty(tmp_path: Path) -> None:
    """Samples entirely after stop produce empty arrays."""
    # clip: start=10, stop=30; all samples at 35+ are excluded
    dyn = {"boli": _make_dyn([35, 45])}
    out = _run(
        tmp_path,
        tmp_path / "out",
        frame_count=50,
        out_frame_count=20,
        start_frame=11,
        dynamic_objects=dyn,
    )
    assert out is not None
    with h5py.File(out, "r") as f:
        si = f["dynamic_objects/boli/sample_indices"][:]
        counts = f["dynamic_objects/boli/counts"][:]
        points = f["dynamic_objects/boli/points"][:]
    assert len(si) == 0
    assert len(counts) == 0
    assert points.shape[0] == 0


def test_dynamic_objects_samples_at_stop_excluded(tmp_path: Path) -> None:
    """Sample at exactly stop (exclusive boundary) is not included."""
    # clip: start=10, stop=30 (out_frame_count=20, start_frame=11)
    # sample at 30 is at stop — should be excluded
    dyn = {"boli": _make_dyn([20, 30])}
    out = _run(
        tmp_path,
        tmp_path / "out",
        frame_count=50,
        out_frame_count=20,
        start_frame=11,
        dynamic_objects=dyn,
    )
    assert out is not None
    with h5py.File(out, "r") as f:
        si = f["dynamic_objects/boli/sample_indices"][:]
        counts = f["dynamic_objects/boli/counts"][:]
        points = f["dynamic_objects/boli/points"][:]
    np.testing.assert_array_equal(si, [10])
    np.testing.assert_array_equal(counts, [1])
    np.testing.assert_array_equal(points[:, 0, 0, 0], [20])


def test_external_identity_mapping_copied_as_is(tmp_path: Path) -> None:
    """poseest/external_identity_mapping is identity-level and must be copied without slicing."""
    pose_path = tmp_path / "video_pose_est_v6.h5"
    mapping = np.array([b"mouse_a", b"mouse_b"], dtype=h5py.string_dtype())
    with h5py.File(pose_path, "w") as f:
        f.create_dataset("poseest/points", data=np.zeros((50, 2, 12, 2), dtype=np.float32))
        f.create_dataset("poseest/confidence", data=np.zeros((50, 2, 12), dtype=np.float32))
        f.create_dataset("poseest/external_identity_mapping", data=mapping)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _sample_one(
        vid_filename="video.mp4",
        root_dir=tmp_path,
        out_dir=out_dir,
        out_frame_count=20,
        start_frame=11,
        only_pose=True,
    )

    outputs = list(out_dir.glob("*.h5"))
    assert len(outputs) == 1
    with h5py.File(outputs[0], "r") as f:
        result = f["poseest/external_identity_mapping"][:]
    assert list(result) == list(mapping)


def test_dynamic_objects_axis_order_attr_preserved(tmp_path: Path) -> None:
    """axis_order attribute on points dataset is preserved in output."""
    dyn = {"boli": _make_dyn([15])}
    dyn["boli"]["axis_order"] = "xy"
    out = _run(
        tmp_path,
        tmp_path / "out",
        frame_count=50,
        out_frame_count=20,
        start_frame=11,
        dynamic_objects=dyn,
    )
    assert out is not None
    with h5py.File(out, "r") as f:
        assert f["dynamic_objects/boli/points"].attrs.get("axis_order") == "xy"
