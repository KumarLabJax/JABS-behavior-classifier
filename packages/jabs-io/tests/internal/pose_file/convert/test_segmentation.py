"""Legacy segmentation becomes the dense baseline plus its ``ragged`` alternate."""

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file.convert import convert_legacy, inspect_legacy, plan_placement
from jabs.io.internal.pose_file.convert.convert import block_frames_for
from jabs.io.internal.pose_file.convert.inspect import BLOCK_FRAMES
from jabs.io.internal.pose_file.convert.memory import contour_chunk_frames
from jabs.io.internal.pose_file.convert.segmentation import segmentation_components
from jabs.io.internal.pose_file.convert.sources import pose_components
from jabs.io.internal.pose_file.reader import read_pose_file
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12, jabs_mouse12
from jabs.io.internal.pose_file.types import PoseFile, VideoInfo
from jabs.io.internal.pose_file.validate import validate
from jabs.io.internal.pose_file.writer import COMPRESSION_LEVEL, write_pose_file

from .conftest import SAMPLES, generator, legacy, requires_samples, widen_segmentation

_SEG = "/jabs/segmentation"


def _convert(path: Path, out: Path) -> None:
    """Write one legacy file's pose and segmentation components to ``out``, deep-checked."""
    layout = inspect_legacy(path)
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout)
    # The smallest blocks the converter takes: one chunk row of the contours.
    block_frames = block_frames_for(layout, 1)
    components = (
        *pose_components(path, layout, plan, block_frames=block_frames),
        *segmentation_components(path, layout, plan, block_frames=block_frames),
    )
    pose_file = PoseFile(
        dimensions={"frame": layout.frames, "slot": plan.slots, "identity": layout.identity},
        video=VideoInfo(frame_count=layout.frames),
        skeletons={JABS_MOUSE12: jabs_mouse12()},
        components=components,
    )
    write_pose_file(pose_file, out, deep=True)


def _assert_deep_clean(out: Path) -> None:
    """Assert the deep check finds no error and decoded every alternate.

    An undecodable alternate is only a warning, so a clean write alone does
    not prove the alternates were compared. Other warnings, such as the
    provenance the caller of these components assigns, are not this task's.
    """
    findings = validate(out, deep=True)
    assert [f for f in findings if f.severity == "error"] == []
    assert [f for f in findings if f.check == "alternate_deep_unchecked"] == []


def _long_seg_file(tmp_path: Path) -> Path:
    """Write a segmented legacy file several chunk rows, so several blocks, long."""
    gen = generator()
    path = tmp_path / "long-seg.h5"
    with pytest.MonkeyPatch.context() as patch:
        # Wide enough that a chunk row is a few hundred frames.
        patch.setattr(gen, "CONTOUR_POINTS", 400)
        gen.write_legacy(path, frames=2000, instances=3, identities=3, version=gen.V8, seg=True)
    assert contour_chunk_frames(inspect_legacy(path)) < 1000
    return path


def test_components_in_order() -> None:
    """The segmentation components come in the brief's order, with their alternates."""
    path = legacy("v8-edge-cases.h5")
    layout = inspect_legacy(path)
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout)
    components = segmentation_components(path, layout, plan, block_frames=BLOCK_FRAMES)
    assert [c.id for c in components] == [
        "jabs.segmentation.contours",
        "jabs.segmentation.contour_count",
        "jabs.segmentation.contour_length",
        "jabs.segmentation.external_flag",
    ]
    contours, external = components[0], components[3]
    assert [a.name for a in contours.alternates] == ["ragged"]
    assert [a.name for a in external.alternates] == ["ragged"]
    assert contours.alternates[0].index is external.alternates[0].index


def test_without_segmentation_returns_nothing() -> None:
    """A file without seg_data has no segmentation components."""
    path = legacy("v8-with-centers.h5")
    layout = inspect_legacy(path)
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout)
    assert segmentation_components(path, layout, plan, block_frames=BLOCK_FRAMES) == []


def test_refuses_blocks_off_the_chunk_row_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block that would split the dense contours' chunks is refused, short of the whole file."""
    path = _wide_seg_file(tmp_path, monkeypatch)
    layout = inspect_legacy(path)
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout)
    rows = contour_chunk_frames(layout)
    assert 1 < rows < layout.frames
    with pytest.raises(ValueError, match="block_frames"):
        segmentation_components(path, layout, plan, block_frames=rows + 1)
    assert segmentation_components(path, layout, plan, block_frames=2 * rows)
    assert segmentation_components(path, layout, plan, block_frames=layout.frames + 1)


def _wide_seg_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a segmented legacy file whose contours are wide: a few frames a chunk."""
    gen = generator()
    monkeypatch.setattr(gen, "CONTOUR_POINTS", 20_000)
    path = tmp_path / "wide-seg.h5"
    gen.write_legacy(path, frames=30, instances=3, identities=3, version=gen.V8, seg=True)
    return path


def test_wide_contours_stream_in_blocks_under_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every dense block is whole chunk rows and within the budget, or one chunk row."""
    path = _wide_seg_file(tmp_path, monkeypatch)
    layout = inspect_legacy(path)
    budget = 16 * 2**20
    block_frames = block_frames_for(layout, budget)
    rows = contour_chunk_frames(layout)
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout, block_frames=block_frames)
    contours = segmentation_components(path, layout, plan, block_frames=block_frames)[0]
    blocks = [block for _, block in contours.data.blocks()]
    assert sum(block.shape[0] for block in blocks) == layout.frames
    assert len(blocks) > 1
    for block in blocks[:-1]:
        assert block.shape[0] % rows == 0
        assert block.nbytes <= max(budget, rows * block.nbytes // block.shape[0])

    out = tmp_path / "out.h5"
    _convert(path, out)
    with h5py.File(out, "r") as h5:
        dense = h5[f"{_SEG}/contours"]
        assert dense.chunks == (rows, 1, 1, layout.point_axis, 1)
        assert dense.compression == "gzip"
        assert dense.compression_opts == COMPRESSION_LEVEL
        assert dense.shuffle


def test_round_trip_passes_the_deep_check(tmp_path: Path) -> None:
    """The edge-case file writes, and the deep check decodes both alternates cleanly."""
    out = tmp_path / "out.h5"
    _convert(legacy("v8-edge-cases.h5"), out)
    _assert_deep_clean(out)
    with h5py.File(out, "r") as h5:
        assert h5[f"{_SEG}/contours"].dtype == np.int32
        assert h5[f"{_SEG}/ragged/contours"].dtype == np.int32
        assert h5[f"{_SEG}/contour_count"].dtype == np.uint32
        assert h5[f"{_SEG}/contour_length"].dtype == np.uint32
        assert h5[f"{_SEG}/external_flag"].dtype == np.bool_
        assert h5[f"{_SEG}/ragged/external_flag"].dtype == np.bool_
        assert h5[f"{_SEG}/ragged/contour_offsets"].dtype == np.uint64
        assert h5[f"{_SEG}/ragged/instance_offsets"].dtype == np.uint64


def test_dense_contours_are_packed_and_unflipped(tmp_path: Path) -> None:
    """The contour-gap instance's legacy contour 1 is packed to contour 0, unflipped."""
    path = legacy("v8-edge-cases.h5")
    case = generator().EDGE["contour_gap"]
    frame, instance = case.frames[0], case.instances[0]
    layout = inspect_legacy(path)
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout)
        source = h5["poseest/seg_data"][frame, instance, 1]
    slot = int(plan.slot_of[frame, instance])
    assert slot >= 0
    length = int(np.argmax(source[:, 0] == -1)) if (source[:, 0] == -1).any() else len(source)

    out = tmp_path / "out.h5"
    _convert(path, out)
    with h5py.File(out, "r") as h5:
        contours = h5[f"{_SEG}/contours"][frame, slot]
        assert h5[f"{_SEG}/contour_count"][frame, slot] == 1
        lengths = h5[f"{_SEG}/contour_length"][frame, slot]
        flags = h5[f"{_SEG}/external_flag"][frame, slot]
    assert contours.dtype == np.int32
    assert lengths.tolist() == [length] + [0] * (len(lengths) - 1)
    # (x, y) as stored: not flipped.
    np.testing.assert_array_equal(contours[0, :length], source[:length].astype(np.int32))
    assert (contours[0, length:] == -1).all()
    assert (contours[1:] == -1).all()
    assert flags.tolist() == [True] + [False] * (len(flags) - 1)


def _decode_ragged(
    payload: npt.NDArray[np.int32],
    flags: npt.NDArray[np.bool_],
    group_offsets: npt.NDArray[np.uint64],
    instance_offsets: npt.NDArray[np.uint64],
    frame: int,
    slot: int,
    slots: int,
) -> tuple[list[npt.NDArray[np.int32]], list[bool]]:
    """Decode one instance's contours and flags by ADR 0002:621-652, one group at a time."""
    position = frame * slots + slot
    first, last = int(instance_offsets[position]), int(instance_offsets[position + 1])
    contours = [
        payload[int(group_offsets[g]) : int(group_offsets[g + 1])] for g in range(first, last)
    ]
    return contours, flags[first:last].tolist()


@pytest.mark.parametrize("source", ["edge", "long"])
def test_ragged_payload_matches_dense(tmp_path: Path, source: str) -> None:
    """Decoding the ragged alternate instance by instance gives the dense valid points.

    The long file spans several blocks, so block start rows past 0
    are covered.
    """
    path = legacy("v8-edge-cases.h5") if source == "edge" else _long_seg_file(tmp_path)
    out = tmp_path / "out.h5"
    _convert(path, out)
    with h5py.File(out, "r") as h5:
        dense = h5[f"{_SEG}/contours"][...]
        count = h5[f"{_SEG}/contour_count"][...]
        length = h5[f"{_SEG}/contour_length"][...]
        dense_flags = h5[f"{_SEG}/external_flag"][...]
        payload = h5[f"{_SEG}/ragged/contours"][...]
        flags = h5[f"{_SEG}/ragged/external_flag"][...]
        group_offsets = h5[f"{_SEG}/ragged/contour_offsets"][...]
        instance_offsets = h5[f"{_SEG}/ragged/instance_offsets"][...]
    frames, slots = count.shape
    assert len(instance_offsets) == frames * slots + 1
    assert int(instance_offsets[-1]) == len(group_offsets) - 1 == len(flags)
    assert int(group_offsets[-1]) == len(payload)
    assert payload.shape[1:] == (2,)
    seen = 0
    for f in range(frames):
        for s in range(slots):
            contours, contour_flags = _decode_ragged(
                payload, flags, group_offsets, instance_offsets, f, s, slots
            )
            assert len(contours) == count[f, s], (f, s)
            for c, points in enumerate(contours):
                np.testing.assert_array_equal(
                    points, dense[f, s, c, : length[f, s, c]], err_msg=f"{(f, s, c)}"
                )
                seen += len(points)
            assert contour_flags == dense_flags[f, s, : count[f, s]].tolist(), (f, s)
    assert seen == len(payload) > 0


def test_no_contour_points(tmp_path: Path) -> None:
    """A file with no contour points anywhere converts with an empty, contiguous payload."""
    out = tmp_path / "out.h5"
    _convert(legacy("v8-no-contours.h5"), out)
    _assert_deep_clean(out)
    with h5py.File(out, "r") as h5:
        payload = h5[f"{_SEG}/ragged/contours"]
        assert payload.shape == (0, 2)
        assert payload.chunks is None
        assert h5[f"{_SEG}/ragged/external_flag"].shape == (0,)
        assert (h5[f"{_SEG}/contour_count"][...] == 0).all()
        assert (h5[f"{_SEG}/ragged/instance_offsets"][...] == 0).all()


def test_without_external_flag(tmp_path: Path) -> None:
    """Without seg_external_flag, the flag component and its alternate are left out."""
    gen = generator()
    path = tmp_path / "no-flag.h5"
    gen.write_legacy(path, **gen.SMALL, version=gen.V8, seg=True, omit={"seg_external_flag"})
    out = tmp_path / "out.h5"
    _convert(path, out)
    _assert_deep_clean(out)
    with h5py.File(out, "r") as h5:
        assert "external_flag" not in h5[_SEG]
        assert "external_flag" not in h5[f"{_SEG}/ragged"]
        assert h5[f"{_SEG}/ragged/contours"].shape[0] > 0
    carriers = {
        c.id: [a.name for a in c.alternates]
        for c in read_pose_file(out).components
        if c.alternates
    }
    assert carriers == {"jabs.segmentation.contours": ["ragged"]}


@requires_samples
def test_sample_v6_segmentation(tmp_path: Path) -> None:
    """The public v6 sample's segmentation writes, and the deep check finds nothing."""
    out = tmp_path / "out.h5"
    _convert(SAMPLES / "sample_pose_est_v6.h5", out)
    _assert_deep_clean(out)


def test_contours_follow_their_instance(tmp_path: Path) -> None:
    """Each placed instance's legacy contours land in its slot, across several blocks."""
    path = _long_seg_file(tmp_path)
    layout = inspect_legacy(path)
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout)
        seg = h5["poseest/seg_data"][...]
        flag = h5["poseest/seg_external_flag"][...]
    out = tmp_path / "out.h5"
    _convert(path, out)
    with h5py.File(out, "r") as h5:
        dense = h5[f"{_SEG}/contours"][...]
        dense_flags = h5[f"{_SEG}/external_flag"][...]
    frame, instance = np.nonzero(plan.slot_of >= 0)
    slot = plan.slot_of[frame, instance]
    # The clean file has no contour gaps, so legacy order is packed order and
    # -1 padding is already where the converter writes it.
    np.testing.assert_array_equal(dense[frame, slot], seg[frame, instance].astype(np.int32))
    np.testing.assert_array_equal(dense_flags[frame, slot], flag[frame, instance])


@pytest.mark.parametrize("contour_frame", [None, 2], ids=["empty-spare-row", "waived-spare-row"])
def test_wider_segmentation_keeps_each_contour_with_its_instance(
    tmp_path: Path, contour_frame: int | None
) -> None:
    """Spare segmentation rows are ignored; every pose instance keeps its own contours."""
    gen = generator()
    path = tmp_path / "wide.h5"
    gen.write_legacy(
        path,
        **{**gen.SMALL, "version": gen.V8, "seg": True},
        edits=widen_segmentation(1, contour_frame=contour_frame),
    )
    out = tmp_path / "out.h5"
    convert_legacy(path, out, strict=False)
    _assert_deep_clean(out)
    layout = inspect_legacy(path, strict=False)
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout)
        seg = h5["poseest/seg_data"][...]
        flag = h5["poseest/seg_external_flag"][...]
    with h5py.File(out, "r") as h5:
        dense = h5[f"{_SEG}/contours"][...]
        dense_flags = h5[f"{_SEG}/external_flag"][...]
    frame, instance = np.nonzero(plan.slot_of >= 0)
    slot = plan.slot_of[frame, instance]
    np.testing.assert_array_equal(dense[frame, slot], seg[frame, instance].astype(np.int32))
    np.testing.assert_array_equal(dense_flags[frame, slot], flag[frame, instance])
    # The spare row's contour went nowhere: every contour written is an instance's.
    assert (dense >= 0).sum() == (seg[:, : layout.instances] >= 0).sum()
