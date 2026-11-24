"""Pass 2: legacy segmentation as the dense baseline and its ``ragged`` alternate.

The dense components follow the plan: slot ``s`` of frame ``f`` holds the
contours of the legacy instance ``plan.slot_of`` placed there, packed so its
existing contours come first in legacy order (``plan.contour_source``), each
cut to its leading ``plan.contour_length`` points. ``seg_data`` is already
``(x, y)`` and is not flipped (ADR 0002:1305-1310). Padding is -1, as in the
legacy file, though only ``contour_count`` and ``contour_length`` say what is
valid (ADR 0002:813-815).

The ``ragged`` alternate is the dense valid points and flags in
``(frame, slot, contour, point)`` order, which is the order the plan's offsets
were built in. Each payload block is cut from the matching dense block, so
the two cannot disagree; the deep check confirms it.

So ``seg_data`` and ``seg_external_flag`` are each read twice: once for the
ragged payload and once for the dense component. The writer drains each
component's source fully before it starts the next, and the ragged byte
range must be contiguous (spec 0029-A §2.6), so one read cannot feed both.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from jabs.io.internal.pose_file.alternates import Alternate, RaggedIndex
from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.convert.inspect import LegacyLayout
from jabs.io.internal.pose_file.convert.plan import PlacementPlan
from jabs.io.internal.pose_file.convert.sources import (
    _memory_source,
    _placed_source,
    check_block_frames,
)
from jabs.io.internal.pose_file.types import Component

logger = logging.getLogger(__name__)

_GROUP = "/jabs/segmentation"
_RAGGED = f"{_GROUP}/ragged"
_NONE = {"policy": "none"}


def segmentation_components(
    path: Path, layout: LegacyLayout, plan: PlacementPlan, *, block_frames: int
) -> list[Component]:
    """Return the legacy file's segmentation as components with their ``ragged`` alternate.

    In order: ``jabs.segmentation.contours``, ``contour_count``,
    ``contour_length`` and, when the file has ``seg_external_flag``,
    ``external_flag``. ``contours`` and ``external_flag`` each carry a
    ``ragged`` alternate; both share one :class:`RaggedIndex` over the plan's
    offsets. A file without ``seg_external_flag`` gets no flag component:
    JABS's readers treat the flag as optional, and all-false flags would be
    invented data.

    Args:
        path: The legacy file. Each block source opens it when read.
        layout: What ``inspect_legacy`` found in it.
        plan: Where ``plan_placement`` put each legacy instance and contour.
        block_frames: Frames per block; a multiple of 900.

    Returns:
        The components; empty when the file has no segmentation.

    Raises:
        ValueError: If ``block_frames`` is not a positive multiple of 900.
    """
    if not layout.has_segmentation:
        return []
    check_block_frames(block_frames)
    count, length = plan.contour_count, plan.contour_length
    group_offsets, instance_offsets = plan.group_offsets, plan.instance_offsets
    if count is None or length is None or group_offsets is None or instance_offsets is None:
        raise ValueError(f"{path}: has segmentation, but the plan has no contours")
    with h5py.File(path, "r") as h5:
        has_flag = "seg_external_flag" in h5["poseest"]

    frames, slots = layout.frames, plan.slots
    contour_axis, point_axis = layout.contour_axis, layout.point_axis
    index = RaggedIndex(
        group_offsets_path=f"{_RAGGED}/contour_offsets",
        instance_offsets_path=f"{_RAGGED}/instance_offsets",
        group_offsets=group_offsets,
        instance_offsets=instance_offsets,
    )

    contours = _placed_source(
        path,
        plan,
        (frames, slots, contour_axis, point_axis, 2),
        np.int32,
        -1,
        lambda pose, start, stop: _read_contours(pose, plan, start, stop),
        block_frames,
    )
    points = _ragged_source(
        contours,
        (int(group_offsets[-1]), 2),
        # Each contour's leading points, beyond the last of which is padding.
        lambda start, stop: np.arange(point_axis) < length[start:stop, ..., None],
        lambda start: int(group_offsets[instance_offsets[start * slots]]),
    )
    components = [
        Component(
            id="jabs.segmentation.contours",
            axes=("frame", "slot", "contour", "point", "coord"),
            data=contours,
            missing={"policy": "length", "length": "jabs.segmentation.contour_length"},
            units="pixel",
            coord_order="xy",
            alternates=(
                Alternate(
                    name="ragged",
                    path=f"{_RAGGED}/contours",
                    axes=("point", "coord"),
                    data=points,
                    missing=_NONE,
                    index=index,
                    level="leaf",
                    units="pixel",
                    coord_order="xy",
                ),
            ),
        ),
        Component(
            id="jabs.segmentation.contour_count",
            axes=("frame", "slot"),
            data=_memory_source(count, block_frames),
            missing=_NONE,
        ),
        Component(
            id="jabs.segmentation.contour_length",
            axes=("frame", "slot", "contour"),
            data=_memory_source(length, block_frames),
            missing=_NONE,
        ),
    ]
    if has_flag:
        flags = _placed_source(
            path,
            plan,
            (frames, slots, contour_axis),
            np.bool_,
            False,
            lambda pose, start, stop: _read_flags(pose, plan, start, stop),
            block_frames,
        )
        components.append(
            Component(
                id="jabs.segmentation.external_flag",
                axes=("frame", "slot", "contour"),
                data=flags,
                missing=_NONE,
                alternates=(
                    Alternate(
                        name="ragged",
                        path=f"{_RAGGED}/external_flag",
                        axes=("contour",),
                        data=_ragged_source(
                            flags,
                            (int(instance_offsets[-1]),),
                            # Packed, so the first contour_count contours are the slot's.
                            lambda start, stop: np.arange(contour_axis)
                            < count[start:stop, ..., None],
                            lambda start: int(instance_offsets[start * slots]),
                        ),
                        missing=_NONE,
                        index=index,
                        level="group",
                    ),
                ),
            )
        )
    return components


def _legacy_packing(
    plan: PlacementPlan, start: int, stop: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return each legacy instance's packed contour sources and lengths, ``(b, N, C)``.

    The plan records them per slot; this maps them back to the legacy
    instance placed there. An unplaced instance gets source -1 and length 0.
    """
    assert plan.contour_source is not None and plan.contour_length is not None
    slot_of = plan.slot_of[start:stop]
    placed = (slot_of >= 0)[..., None]
    frame = np.arange(stop - start)[:, None]
    # Unplaced instances index slot 0, then are masked: -1 would pick the last slot.
    slot = np.where(slot_of >= 0, slot_of, 0)
    source = np.where(placed, plan.contour_source[start:stop][frame, slot], -1).astype(np.int64)
    length = np.where(placed, plan.contour_length[start:stop][frame, slot], 0).astype(np.int64)
    return source, length


def _read_contours(
    pose: h5py.Group, plan: PlacementPlan, start: int, stop: int
) -> npt.NDArray[np.int32]:
    """Read each legacy instance's contours, packed and cut to length, -1 elsewhere.

    Not flipped: ``seg_data`` is already ``(x, y)``. Points past a contour's
    length are set to -1 rather than trusted, since a legacy contour ends at
    its first -1 x whatever follows.
    """
    source, length = _legacy_packing(plan, start, stop)
    # Rows past the legacy instances are spare (see ``inspect_legacy``).
    seg = pose["seg_data"][start:stop, : source.shape[1]]
    packed = np.take_along_axis(seg, np.maximum(source, 0)[..., None, None], axis=2)
    keep = np.arange(seg.shape[3]) < length[..., None]
    return np.where(keep[..., None], packed, -1).astype(np.int32)


def _read_flags(
    pose: h5py.Group, plan: PlacementPlan, start: int, stop: int
) -> npt.NDArray[np.bool_]:
    """Read each legacy instance's external flags, packed like its contours, false where unused."""
    source, _ = _legacy_packing(plan, start, stop)
    flags = pose["seg_external_flag"][start:stop, : source.shape[1]].astype(np.bool_)
    return np.take_along_axis(flags, np.maximum(source, 0), axis=2) & (source >= 0)


def _ragged_source(
    dense: BlockSource,
    shape: tuple[int, ...],
    valid: Callable[[int, int], npt.NDArray[np.bool_]],
    first_row: Callable[[int], int],
) -> BlockSource:
    """Return a ragged payload cut from ``dense``'s blocks.

    Each dense block's ``valid`` entries, taken row-major, are the payload rows
    of its frames, in ``(frame, slot, contour, point)`` order; ``first_row``
    says where they start. A block with no rows is skipped.
    """

    def blocks() -> Iterator[tuple[int, npt.NDArray[np.generic]]]:
        """Yield each frame block's payload rows."""
        for start, block in dense.blocks():
            rows = block[valid(start, start + block.shape[0])]
            if rows.shape[0]:
                yield first_row(start), rows

    return BlockSource(shape, dense.dtype, blocks)
