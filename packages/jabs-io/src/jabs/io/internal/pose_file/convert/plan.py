"""Pass 1: place every legacy instance in a slot before anything is written.

Placement decides the output's slot count, so nothing can be written until
every frame has been seen. This pass reads the legacy file once, in frame
blocks, and records where each legacy instance goes, how its contours pack,
and how its tracklets split. It refuses a duplicate identity or a tracklet in
two slots of one frame here, so a refusal never leaves a partial output.

The rules are ADR 0002:1294-1315 and spec 0029-A §2.3-§2.5. Everything kept is
``O(F*S*C)`` integers; ``seg_data`` is read one block at a time.
"""

from dataclasses import dataclass

import h5py
import numpy as np
import numpy.typing as npt

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import BLOCK_FRAMES, LegacyLayout


@dataclass(frozen=True, eq=False)
class PlacementPlan:
    """Where every legacy instance, contour and tracklet goes in the output.

    Attributes:
        slots: The output slot count: the identity count plus the most tail
            instances in any frame.
        slot_of: ``(F, N)``, the slot each legacy instance is placed in; -1
            where it is not placed.
        tail_slots: One record per tail-slot instance, in frame then slot
            order: ``{"frame", "slot", "origin", "legacy_index"}``, with
            origin ``"embed_out_of_range"`` or ``"masked"``.
        longterm_mismatches: One record per valid instance whose
            ``longterm_seg_id`` differs from its embed id:
            ``{"frame", "legacy_index", "embed_id", "longterm_seg_id"}``.
        contour_count: ``(F, S)``, contours per slot; ``None`` without
            segmentation.
        contour_length: ``(F, S, C)``, points per contour, packed: the slot's
            contours first, in legacy order, then 0.
        contour_source: ``(F, S, C)``, the legacy contour index at each packed
            position; -1 where unused.
        group_offsets: The ragged ``contour_offsets``: where each contour's
            points start, in ``(frame, slot, contour)`` order, then the total.
        instance_offsets: Where each ``(frame, slot)``'s contours start, in
            row-major order, then the total; ``F * S + 1`` long.
        tracklet_id: ``(F, S)`` tracklet ids after splitting, 0 for none;
            ``None`` when the file has no ``instance_track_id``.
        tracklet_splits: One record per run that took a new id:
            ``{"id", "from", "first_frame", "last_frame"}``, in the order the
            new ids were given.
    """

    slots: int
    slot_of: npt.NDArray[np.int16]
    tail_slots: tuple[dict, ...]
    longterm_mismatches: tuple[dict, ...]
    contour_count: npt.NDArray[np.uint32] | None
    contour_length: npt.NDArray[np.uint32] | None
    contour_source: npt.NDArray[np.int8] | None
    group_offsets: npt.NDArray[np.uint64] | None
    instance_offsets: npt.NDArray[np.uint64] | None
    tracklet_id: npt.NDArray[np.uint32] | None
    tracklet_splits: tuple[dict, ...]


def plan_placement(
    h5: h5py.File, layout: LegacyLayout, *, block_frames: int = BLOCK_FRAMES
) -> PlacementPlan:
    """Place every legacy instance, pack its contours and split its tracklets.

    A valid instance (``id_mask == 0``) whose embed id ``e`` is in
    ``1..identity`` goes to slot ``e - 1``. Every other instance with anything
    to place, a valid one with an embed id out of that range or a masked one
    with a confidence above 0 or a contour, goes to a tail slot, in legacy
    index order. Contours follow the instance, never ``longterm_seg_id``, which
    is only checked against the embed id.

    Args:
        h5: The open legacy file.
        layout: What ``inspect_legacy`` found in it.
        block_frames: Frames read at a time.

    Returns:
        The plan.

    Raises:
        LegacyConversionError: If two valid instances of a frame hold one
            identity, if one tracklet id is in two slots of a frame, or if the
            file's extents do not fit the plan's index types. The message
            names the file, and the frame and id where there is one.
    """
    pose = h5["poseest"]
    frames, instances, identity = layout.frames, layout.instances, layout.identity
    contours = layout.contour_axis
    _check_extents(layout)

    seg = pose["seg_data"] if layout.has_segmentation else None
    track = pose["instance_track_id"] if layout.has_tracklets else None
    longterm = pose.get("longterm_seg_id")

    slot_of = np.full((frames, instances), -1, dtype=np.int16)
    # Per legacy instance until the slot count is known, then scattered.
    legacy_track = np.zeros((frames, instances), dtype=np.uint32) if track is not None else None
    legacy_length = np.zeros((frames, instances, contours), dtype=np.uint32)
    legacy_source = np.full((frames, instances, contours), -1, dtype=np.int8)
    tails: list[dict] = []
    mismatches: list[dict] = []
    most_tails = 0

    for start in range(0, frames, block_frames):
        stop = min(start + block_frames, frames)
        valid = pose["id_mask"][start:stop] == 0
        # Signed, so e - 1 cannot wrap at e == 0.
        embed = pose["instance_embed_id"][start:stop].astype(np.int64)
        anything = (pose["confidence"][start:stop] > 0).any(axis=-1)
        if seg is not None:
            anything |= _pack_contours(
                seg[start:stop, :, :, :, 0], legacy_length[start:stop], legacy_source[start:stop]
            )

        in_range = valid & (embed >= 1) & (embed <= identity)
        _refuse_duplicates(layout, start, embed, in_range)
        tail = (valid & ~in_range) | (~valid & anything)
        slots = np.full(valid.shape, -1, dtype=np.int64)
        slots[in_range] = embed[in_range] - 1
        slots[tail] = identity + (np.cumsum(tail, axis=1) - 1)[tail]
        slot_of[start:stop] = slots
        most_tails = max(most_tails, int(tail.sum(axis=1).max(initial=0)))

        for f, i in zip(*np.nonzero(tail), strict=True):
            tails.append(
                {
                    "frame": start + int(f),
                    "slot": int(slots[f, i]),
                    "origin": "embed_out_of_range" if valid[f, i] else "masked",
                    "legacy_index": int(i),
                }
            )
        if longterm is not None:
            seg_id = longterm[start:stop].astype(np.int64)
            for f, i in zip(*np.nonzero(valid & (seg_id != embed)), strict=True):
                mismatches.append(
                    {
                        "frame": start + int(f),
                        "legacy_index": int(i),
                        "embed_id": int(embed[f, i]),
                        "longterm_seg_id": int(seg_id[f, i]),
                    }
                )
        if track is not None and legacy_track is not None:
            placed = np.where(slots >= 0, track[start:stop], 0).astype(np.uint32)
            _refuse_shared_tracklets(layout, start, placed, slots)
            legacy_track[start:stop] = placed

    slot_count = identity + most_tails
    placed_frame, placed_instance = np.nonzero(slot_of >= 0)
    placed_slot = slot_of[placed_frame, placed_instance]

    tracklet_id = None
    splits: tuple[dict, ...] = ()
    if legacy_track is not None:
        tracklet_id = np.zeros((frames, slot_count), dtype=np.uint32)
        tracklet_id[placed_frame, placed_slot] = legacy_track[placed_frame, placed_instance]
        splits = _split_tracklets(tracklet_id)

    count = length = source = group_offsets = instance_offsets = None
    if seg is not None:
        length = np.zeros((frames, slot_count, contours), dtype=np.uint32)
        source = np.full((frames, slot_count, contours), -1, dtype=np.int8)
        length[placed_frame, placed_slot] = legacy_length[placed_frame, placed_instance]
        source[placed_frame, placed_slot] = legacy_source[placed_frame, placed_instance]
        count = (source >= 0).sum(axis=-1, dtype=np.uint32)
        instance_offsets = _offsets(count.ravel())
        # Packing puts used positions first, so this is (frame, slot, contour) order.
        group_offsets = _offsets(length[source >= 0])

    return PlacementPlan(
        slots=slot_count,
        slot_of=slot_of,
        tail_slots=tuple(tails),
        longterm_mismatches=tuple(mismatches),
        contour_count=count,
        contour_length=length,
        contour_source=source,
        group_offsets=group_offsets,
        instance_offsets=instance_offsets,
        tracklet_id=tracklet_id,
        tracklet_splits=splits,
    )


def _check_extents(layout: LegacyLayout) -> None:
    """Refuse a file whose slots or contour indexes would overflow the plan's types."""
    if layout.identity + layout.instances > np.iinfo(np.int16).max:
        raise LegacyConversionError(
            f"{layout.path}: {layout.identity} identities and {layout.instances} instances "
            "exceed the slots a plan can index"
        )
    if layout.contour_axis > np.iinfo(np.int8).max + 1:
        raise LegacyConversionError(
            f"{layout.path}: poseest/seg_data has {layout.contour_axis} contours per instance, "
            "more than a plan can index"
        )


def _pack_contours(
    x: npt.NDArray[np.integer],
    length: npt.NDArray[np.uint32],
    source: npt.NDArray[np.int8],
) -> npt.NDArray[np.bool_]:
    """Fill ``length`` and ``source`` from a block of ``seg_data`` x values; return any-contour."""
    padding = x == -1
    exists = ~padding[..., 0]
    # A contour's points are the leading ones before the first -1 x.
    points = np.where(padding.any(axis=-1), padding.argmax(axis=-1), x.shape[-1])
    # Stable, so existing contours keep their legacy order at the front.
    order = np.argsort(~exists, axis=-1, kind="stable")
    packed = np.take_along_axis(exists, order, axis=-1)
    source[...] = np.where(packed, order, -1)
    length[...] = np.where(packed, np.take_along_axis(points, order, axis=-1), 0)
    return exists.any(axis=-1)


def _refuse_duplicates(
    layout: LegacyLayout,
    start: int,
    embed: npt.NDArray[np.int64],
    in_range: npt.NDArray[np.bool_],
) -> None:
    """Refuse the first frame where two valid instances hold one identity (spec 0029-A §2.4)."""
    frame, instance = np.nonzero(in_range)
    keys = frame * (layout.identity + 1) + embed[frame, instance]
    unique, counts = np.unique(keys, return_counts=True)
    shared = unique[counts > 1]
    if shared.size:
        key = shared[0]
        holders = instance[keys == key].tolist()
        raise LegacyConversionError(
            f"{layout.path}: frame {start + key // (layout.identity + 1)} has valid legacy "
            f"instances {holders} that all hold identity {key % (layout.identity + 1)}; "
            "refusing to choose which one the identity follows"
        )


def _refuse_shared_tracklets(
    layout: LegacyLayout,
    start: int,
    track: npt.NDArray[np.uint32],
    slots: npt.NDArray[np.int64],
) -> None:
    """Refuse the first frame where one nonzero tracklet id is in two slots."""
    ordered = np.sort(track, axis=1)
    repeated = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] > 0)
    if repeated.any():
        frame = int(np.nonzero(repeated.any(axis=1))[0][0])
        tracklet = int(ordered[frame, 1:][repeated[frame]][0])
        held = slots[frame, track[frame] == tracklet].tolist()
        raise LegacyConversionError(
            f"{layout.path}: frame {start + frame} has tracklet {tracklet} in slots {held}; "
            "a tracklet is one animal, in one slot per frame"
        )


def _split_tracklets(tracklet_id: npt.NDArray[np.uint32]) -> tuple[dict, ...]:
    """Split each id's placed frames into unbroken runs, in place; return the new ids' records.

    The first run of an id keeps it. Each later run, across all ids in order
    of first frame, takes the next id above the largest placed one. A move to
    another slot is not a gap: runs are over frames.
    """
    frame, slot = np.nonzero(tracklet_id)
    if frame.size == 0:
        return ()
    ids = tracklet_id[frame, slot].astype(np.int64)
    order = np.lexsort((frame, ids))
    frame, slot, ids = frame[order], slot[order], ids[order]

    new_id = np.ones(ids.size, dtype=bool)
    new_id[1:] = ids[1:] != ids[:-1]
    # An id is in at most one slot per frame (refused otherwise), so a run
    # breaks exactly where its next frame is not the following one.
    new_run = new_id.copy()
    new_run[1:] |= frame[1:] != frame[:-1] + 1
    starts = np.flatnonzero(new_run)
    ends = np.append(starts[1:], ids.size) - 1
    run_ids = ids[starts]

    later = np.flatnonzero(~new_id[starts])
    later = later[np.lexsort((run_ids[later], frame[starts[later]]))]
    largest = int(ids.max())
    splits = []
    for n, run in enumerate(later.tolist(), start=1):
        splits.append(
            {
                "id": largest + n,
                "from": int(run_ids[run]),
                "first_frame": int(frame[starts[run]]),
                "last_frame": int(frame[ends[run]]),
            }
        )
        run_ids[run] = largest + n
    if splits:
        run_of = np.cumsum(new_run) - 1
        tracklet_id[frame, slot] = run_ids[run_of]
    return tuple(splits)


def _offsets(sizes: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint64]:
    """Return the uint64 offsets of consecutive ``sizes``: 0, then their running total."""
    offsets = np.zeros(sizes.size + 1, dtype=np.uint64)
    np.cumsum(sizes, dtype=np.uint64, out=offsets[1:])
    return offsets
