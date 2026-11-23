"""The converter refuses a file whose declared shapes would exhaust memory, before allocating."""

import dataclasses
import shutil
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.convert import (
    LegacyConversionError,
    convert_legacy,
    inspect_legacy,
)
from jabs.io.internal.pose_file.convert.convert import DEFAULT_MAX_MEMORY, block_frames_for
from jabs.io.internal.pose_file.convert.memory import (
    STORAGE_RATIO,
    STORAGE_RATIO_FLOOR,
    contour_chunk_frames,
    estimate_memory,
)

from .conftest import generator, legacy

# What a refusal may allocate: the shapes are read, never the data.
_SMALL_PEAK = 16 * 2**20


def _declare(
    path: Path,
    *,
    frames: int,
    instances: int = 4,
    edits: Callable[[h5py.Group], None] | None = None,
    write_first_block: bool = False,
) -> None:
    """Write a legacy file that declares ``frames`` frames and stores almost none of them.

    Every dataset is chunked and, unless ``write_first_block``, never written,
    so the file is a few kilobytes whatever it declares.
    """
    shape = (frames, instances)
    with h5py.File(path, "w") as h5:
        pose = h5.create_group("poseest")
        pose.attrs["version"] = np.array((8, 0, 0), dtype=np.uint16)
        declared = {
            "points": ((*shape, 12, 2), np.float32),
            "confidence": ((*shape, 12), np.float32),
            "id_mask": (shape, np.uint8),
            "instance_embed_id": (shape, np.uint32),
        }
        for name, (dataset_shape, dtype) in declared.items():
            chunks = (min(frames, 900), *dataset_shape[1:])
            dataset = pose.create_dataset(
                name, shape=dataset_shape, dtype=dtype, chunks=chunks, compression="gzip"
            )
            if write_first_block:
                dataset[: chunks[0]] = np.ones(chunks, dtype=dtype)
        if edits is not None:
            edits(pose)


def _refusal(source: Path, out: Path, **options: int) -> tuple[str, int]:
    """Convert, expecting a refusal; return its message and the traced peak."""
    tracemalloc.start()
    try:
        with pytest.raises(LegacyConversionError) as refused:
            convert_legacy(source, out, **options)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert not out.exists()
    return str(refused.value), peak


@pytest.mark.parametrize("write_first_block", [False, True], ids=["unwritten", "one-block"])
def test_huge_frames_are_refused_before_allocating(
    tmp_path: Path, write_first_block: bool
) -> None:
    """Billions of declared frames, stored in kilobytes, are refused with the estimate."""
    src = tmp_path / "huge.h5"
    _declare(src, frames=2_000_000_000, write_first_block=write_first_block)
    assert src.stat().st_size < 2**20

    message, peak = _refusal(src, tmp_path / "out.h5")

    assert "estimated" in message
    assert "GiB" in message
    assert f"{DEFAULT_MAX_MEMORY / 2**30:.1f} GiB" in message
    assert peak < _SMALL_PEAK


def test_a_huge_contour_axis_is_refused_by_its_block(tmp_path: Path) -> None:
    """A few frames whose smallest block would be enormous are refused."""

    def contours(pose: h5py.Group) -> None:
        frames, instances = pose["points"].shape[:2]
        pose.create_dataset(
            "seg_data",
            shape=(frames, instances, 100, 1_000_000, 2),
            dtype=np.int32,
            chunks=(1, 1, 1, 1000, 2),
            compression="gzip",
        )

    src = tmp_path / "wide.h5"
    _declare(src, frames=900, edits=contours)

    message, peak = _refusal(src, tmp_path / "out.h5")

    assert "estimated" in message
    assert peak < _SMALL_PEAK


def test_a_huge_embedding_width_is_refused(tmp_path: Path) -> None:
    """The embedding width counts too, though it does not size the block."""

    def embeddings(pose: h5py.Group) -> None:
        frames, instances = pose["points"].shape[:2]
        pose.create_dataset(
            "identity_embeds",
            shape=(frames, instances, 2**30),
            dtype=np.float32,
            chunks=(1, 1, 2**16),
            compression="gzip",
        )

    src = tmp_path / "embeds.h5"
    _declare(src, frames=900, edits=embeddings)

    message, peak = _refusal(src, tmp_path / "out.h5")

    assert "estimated" in message
    assert peak < _SMALL_PEAK


def test_declared_far_beyond_stored_is_refused(tmp_path: Path) -> None:
    """A dataset the conversion never reads, declared in terabytes and stored in nothing.

    The memory estimate does not count it, so the storage check is what
    refuses it, naming the dataset.
    """

    def unread(pose: h5py.Group) -> None:
        frames, instances = pose["points"].shape[:2]
        pose.create_dataset(
            "instance_embedding",
            shape=(frames, instances, 2**28),
            dtype=np.float32,
            chunks=(1, 1, 2**16),
            compression="gzip",
        )

    src = tmp_path / "unread.h5"
    _declare(src, frames=900, edits=unread)

    message, peak = _refusal(src, tmp_path / "out.h5")

    assert "poseest/instance_embedding" in message
    assert "stores" in message
    assert peak < _SMALL_PEAK


def test_the_storage_ratio_spares_small_datasets(tmp_path: Path) -> None:
    """Below the size floor, a well-compressed dataset is not suspicious."""
    src = tmp_path / "small.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        frames, instances = h5["poseest/points"].shape[:2]
        width = STORAGE_RATIO_FLOOR // (frames * instances * 4) - 1
        h5["poseest"].create_dataset(
            "instance_embedding",
            shape=(frames, instances, width),
            dtype=np.float32,
            chunks=(1, 1, min(width, 2**16)),
            compression="gzip",
        )

    convert_legacy(src, tmp_path / "out.h5")


def test_the_storage_ratio_is_above_real_compression() -> None:
    """gzip of constant data stops near 1000:1, and real files reach under 600:1.

    The ratio is there for storage that was never written, not for data that
    compresses well, so it must sit above what deflate can reach.
    """
    assert STORAGE_RATIO > 1032


def test_corpus_files_convert_under_the_default_cap(tmp_path: Path) -> None:
    """Every convertible corpus file's estimate is far below the default cap."""
    for name in ("v8-edge-cases.h5", "v8-with-centers.h5", "v7-dynamic-objects.h5"):
        layout = inspect_legacy(legacy(name))
        with h5py.File(legacy(name), "r") as h5:
            estimate = estimate_memory(
                h5, identity=layout.identity, block_frames=block_frames_for(layout, 64 * 2**20)
            )
        assert 0 < estimate < DEFAULT_MAX_MEMORY / 100
        convert_legacy(legacy(name), tmp_path / f"{name}.out.h5")


def test_a_cap_below_the_estimate_refuses(tmp_path: Path) -> None:
    """The cap is the caller's: just below a file's estimate refuses it, at it converts."""
    source = legacy("v8-edge-cases.h5")
    layout = inspect_legacy(source)
    block_bytes = 64 * 2**20
    with h5py.File(source, "r") as h5:
        estimate = estimate_memory(
            h5, identity=layout.identity, block_frames=block_frames_for(layout, block_bytes)
        )
    out = tmp_path / "out.h5"

    with pytest.raises(LegacyConversionError, match=f"{estimate} bytes"):
        convert_legacy(source, out, max_memory=estimate - 1, block_bytes=block_bytes)
    assert not out.exists()

    convert_legacy(source, out, max_memory=estimate, block_bytes=block_bytes)
    assert out.exists()


def test_the_estimate_covers_the_measured_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At v6 widths, the estimate is above what conversion really allocates, within 4x.

    Too low and the cap does not protect; too high and real multi-day
    recordings are refused under the default.
    """
    module = generator()
    monkeypatch.setattr(module, "CONTOURS", 4)
    monkeypatch.setattr(module, "CONTOUR_POINTS", 8)
    src = tmp_path / "v6.h5"
    module.write_legacy(src, frames=36_000, instances=5, identities=4, version=(6, 0), seg=True)
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


def test_a_four_day_v6_recording_fits_the_default_cap(tmp_path: Path) -> None:
    """The public v6 sample's widths over four days at 30 fps are under 16 GiB.

    The plan keeps O(frames) arrays, so a long recording needs memory in
    proportion; the default cap must still admit a real multi-day one.
    """
    frames = 4 * 24 * 3600 * 30

    def v6(pose: h5py.Group) -> None:
        instances = pose["points"].shape[1]
        for name, shape, dtype in (
            ("seg_data", (frames, instances, 4, 319, 2), np.int16),
            ("seg_external_flag", (frames, instances, 4), np.bool_),
            ("instance_track_id", (frames, instances), np.uint32),
            ("longterm_seg_id", (frames, instances), np.uint32),
            ("identity_embeds", (frames, instances, 16), np.float32),
        ):
            pose.create_dataset(name, shape=shape, dtype=dtype, chunks=True)
        pose.create_dataset("instance_id_center", data=np.zeros((4, 16)))

    src = tmp_path / "v6-four-days.h5"
    _declare(src, frames=frames, instances=5, edits=v6)
    with h5py.File(src, "r") as h5:
        estimate = estimate_memory(h5, identity=4, block_frames=900)

    assert estimate < DEFAULT_MAX_MEMORY


def test_a_real_wide_segmentation_fits_the_default_cap(tmp_path: Path) -> None:
    """The widest real file seen, 79 contours of 5,923 points, estimates under 16 GiB.

    Its shapes and legacy chunks are a real 1800-frame v8 file's, the largest
    of 7,729. With 900-frame blocks the estimate was 98 GiB at inspection and
    133 GiB in full, and the file was refused.
    """
    frames, instances, identities = 1800, 5, 5

    def wide(pose: h5py.Group) -> None:
        pose.create_dataset(
            "seg_data",
            shape=(frames, instances, 79, 5923, 2),
            dtype=np.int16,
            chunks=(113, 1, 5, 371, 1),
            compression="gzip",
        )
        pose.create_dataset(
            "seg_external_flag", shape=(frames, instances, 79), dtype=np.bool_, chunks=True
        )
        pose.create_dataset("instance_id_center", data=np.zeros((identities, 16)))

    src = tmp_path / "wide.h5"
    _declare(src, frames=frames, instances=instances, edits=wide)
    seen = inspect_legacy(legacy("v8-edge-cases.h5"))
    layout = dataclasses.replace(
        seen,
        frames=frames,
        instances=instances,
        identity=identities,
        contour_axis=79,
        point_axis=5923,
    )
    with h5py.File(src, "r") as h5:
        floor = estimate_memory(h5, identity=identities, block_frames=contour_chunk_frames(layout))
        full = estimate_memory(
            h5, identity=identities, block_frames=block_frames_for(layout, 64 * 2**20)
        )

    assert floor == full < DEFAULT_MAX_MEMORY


# _declare's largest chunk: 900 frames of 4 instances' float32 points.
_POINTS_CHUNK = 900 * 4 * 12 * 2 * 4
# A chunk longer than the 900 frames declared, which a resizable dataset may have.
_BIG_CHUNK_FRAMES = 2**16
_BIG_CHUNK = _BIG_CHUNK_FRAMES * 4 * 12 * 2 * 4


def _one_big_chunk(pose: h5py.Group) -> None:
    """Rewrite the points as one chunk past the extent, written, so it is decompressed whole."""
    frames, instances = pose["points"].shape[:2]
    del pose["points"]
    points = pose.create_dataset(
        "points",
        shape=(frames, instances, 12, 2),
        maxshape=(None, instances, 12, 2),
        dtype=np.float32,
        chunks=(_BIG_CHUNK_FRAMES, instances, 12, 2),
        compression="gzip",
    )
    points[:] = 1


def _estimate(path: Path) -> int:
    """Estimate a file's conversion with a fixed identity count and block."""
    with h5py.File(path, "r") as h5:
        return estimate_memory(h5, identity=4, block_frames=900)


def test_the_estimate_counts_the_largest_chunk(tmp_path: Path) -> None:
    """Reading any part of a chunk decompresses all of it, so the largest chunk is counted."""
    small, big = tmp_path / "small.h5", tmp_path / "big.h5"
    _declare(small, frames=900)
    _declare(big, frames=900, edits=_one_big_chunk)

    assert _estimate(big) - _estimate(small) == _BIG_CHUNK - _POINTS_CHUNK


def test_the_estimate_counts_an_object_dataset_chunk(tmp_path: Path) -> None:
    """The static and dynamic objects are read too, and their chunks count the same way."""
    contiguous, chunked = tmp_path / "contiguous.h5", tmp_path / "chunked.h5"
    for path, chunks in ((contiguous, None), (chunked, (2**20, 2))):
        _declare(path, frames=900)
        with h5py.File(path, "r+") as h5:
            h5.create_group("static_objects").create_dataset(
                "corners",
                data=np.zeros((4, 2)),
                maxshape=(None, 2) if chunks else None,
                chunks=chunks,
            )

    assert _estimate(chunked) - _estimate(contiguous) == 2**20 * 2 * 8 - _POINTS_CHUNK


def test_one_huge_compressible_chunk_is_refused(tmp_path: Path) -> None:
    """A chunk far larger than the data it holds, stored in kilobytes, counts against the cap.

    It passes the storage check, being small, yet every block read
    decompresses all of it.
    """
    small, src = tmp_path / "small.h5", tmp_path / "big.h5"
    _declare(small, frames=900)
    _declare(src, frames=900, edits=_one_big_chunk)
    assert src.stat().st_size < 2**20
    cap = _estimate(small) + _BIG_CHUNK // 2

    message, peak = _refusal(src, tmp_path / "out.h5", max_memory=cap)

    assert "estimated" in message
    assert f"({cap} bytes)" in message
    assert peak < _SMALL_PEAK
