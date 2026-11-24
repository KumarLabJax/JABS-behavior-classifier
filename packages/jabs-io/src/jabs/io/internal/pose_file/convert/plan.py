"""Pass 1: place every legacy instance in a slot before anything is written.

Placement decides the output's slot count, so nothing can be written until
every frame has been seen. This pass reads the legacy file once, in frame
blocks, and records where each legacy instance goes, how its contours pack,
and how its tracklets split. It refuses a duplicate identity or a tracklet in
two slots of one frame here, so a refusal never leaves a partial output.

The rules are ADR 0002:1294-1315 and spec 0029-A §2.3-§2.5. Everything kept is
``O(F*S*C)`` integers; ``seg_data`` is read one block at a time.
"""

import heapq
import logging
from dataclasses import dataclass

import h5py
import numpy as np
import numpy.typing as npt

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import BLOCK_FRAMES, LegacyLayout

logger = logging.getLogger(__name__)

_TRACKLET_MAX = int(np.iinfo(np.uint32).max)

# A pose_est_v3 track id is stored plus this, since tracklet 0 means none.
TRACKLET_ID_OFFSET = 1


@dataclass(frozen=True, eq=False)
class TailSlots:
    """The instances placed in tail slots, one row each, in frame then slot order.

    Columns, not one record per row: an unusual file can put an instance in a
    tail slot in most of its frames.

    Attributes:
        frame: The frame of each row.
        slot: The tail slot the instance went to.
        legacy_index: The instance's legacy index.
        masked: True where the instance was masked (``id_mask`` set), false
            where it was valid with an embed id outside ``1..identity``.
    """

    frame: npt.NDArray[np.uint32]
    slot: npt.NDArray[np.int16]
    legacy_index: npt.NDArray[np.int16]
    masked: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        """Return the number of rows."""
        return int(self.frame.size)


@dataclass(frozen=True, eq=False)
class LongtermMismatches:
    """The valid instances whose ``longterm_seg_id`` differs from their embed id, one row each.

    Attributes:
        frame: The frame of each row.
        legacy_index: The instance's legacy index.
        embed_id: Its embed id.
        longterm_seg_id: Its ``longterm_seg_id``.
    """

    frame: npt.NDArray[np.uint32]
    legacy_index: npt.NDArray[np.int16]
    embed_id: npt.NDArray[np.int64]
    longterm_seg_id: npt.NDArray[np.int64]

    def __len__(self) -> int:
        """Return the number of rows."""
        return int(self.frame.size)


@dataclass(frozen=True, eq=False)
class TrackletSplits:
    """The tracklet runs that took a new id, one row each, in the order the ids were given.

    Attributes:
        id: The new id.
        from_id: The legacy id the run had (``"from"`` in the history).
        first_frame: The run's first frame.
        last_frame: The run's last frame.
    """

    id: npt.NDArray[np.uint32]
    from_id: npt.NDArray[np.uint32]
    first_frame: npt.NDArray[np.uint32]
    last_frame: npt.NDArray[np.uint32]

    def __len__(self) -> int:
        """Return the number of rows."""
        return int(self.id.size)


_NO_SPLITS = TrackletSplits(
    id=np.empty(0, np.uint32),
    from_id=np.empty(0, np.uint32),
    first_frame=np.empty(0, np.uint32),
    last_frame=np.empty(0, np.uint32),
)
_NO_TAILS = TailSlots(
    frame=np.empty(0, np.uint32),
    slot=np.empty(0, np.int16),
    legacy_index=np.empty(0, np.int16),
    masked=np.empty(0, np.bool_),
)
_NO_MISMATCHES = LongtermMismatches(
    frame=np.empty(0, np.uint32),
    legacy_index=np.empty(0, np.int16),
    embed_id=np.empty(0, np.int64),
    longterm_seg_id=np.empty(0, np.int64),
)


@dataclass(frozen=True, eq=False)
class PlacementPlan:
    """Where every legacy instance, contour and tracklet goes in the output.

    Attributes:
        slots: The output slot count: the identity count plus the most tail
            instances in any frame.
        slot_of: ``(F, N)``, the slot each legacy instance is placed in; -1
            where it is not placed.
        tail_slots: Every tail-slot instance, in frame then slot order.
        longterm_mismatches: Every valid instance whose ``longterm_seg_id``
            differs from its embed id, in frame then legacy index order.
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
        tracklet_id: ``(F, S)`` tracklet ids after splitting, 0 for none and
            at every unoccupied ``(frame, slot)``; ``None`` when the file has
            no ``instance_track_id``.
        tracklet_splits: Every run that took a new id, in the order the new
            ids were given; empty without tracklets.
    """

    slots: int
    slot_of: npt.NDArray[np.int16]
    tail_slots: TailSlots
    longterm_mismatches: LongtermMismatches
    contour_count: npt.NDArray[np.uint32] | None
    contour_length: npt.NDArray[np.uint32] | None
    contour_source: npt.NDArray[np.int8] | None
    group_offsets: npt.NDArray[np.uint64] | None
    instance_offsets: npt.NDArray[np.uint64] | None
    tracklet_id: npt.NDArray[np.uint32] | None
    tracklet_splits: TrackletSplits


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

    A file with no instance axis (``pose_est_v2``) is one animal: its one
    instance goes to slot 0 in every frame, and nothing is read.

    A ``pose_est_v3`` file has tracks and no identity, and is placed by track
    (see ``_track_plan``): each track keeps one slot for its run, and its id
    is stored plus ``TRACKLET_ID_OFFSET``.

    Args:
        h5: The open legacy file.
        layout: What ``inspect_legacy`` found in it.
        block_frames: Frames read at a time.

    Returns:
        The plan.

    Raises:
        LegacyConversionError: If two valid instances of a frame hold one
            identity, if one tracklet id is in two slots of a frame, if a
            tracklet id is negative or above the uint32 range, if splitting
            needs a new id above that range, or if the file's extents do not
            fit the plan's index types; or, in a ``pose_est_v3`` file, if a
            track id is on two counted instances of a frame or is the uint32
            maximum. The message names the file, and the frame and id where
            there is one.
    """
    if not layout.instance_axis:
        return _single_plan(layout)
    if layout.placed_by_track:
        return _track_plan(h5, layout, block_frames)
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
    # Per block, concatenated at the end.
    tails: list[tuple[npt.NDArray[np.generic], ...]] = []
    mismatches: list[tuple[npt.NDArray[np.generic], ...]] = []
    most_tails = 0

    for start in range(0, frames, block_frames):
        stop = min(start + block_frames, frames)
        valid = pose["id_mask"][start:stop] == 0
        # Signed, so e - 1 cannot wrap at e == 0.
        embed = pose["instance_embed_id"][start:stop].astype(np.int64)
        # slot_occupied, as pass 2 writes it: any confidence above 0.
        occupied = (pose["confidence"][start:stop] > 0).any(axis=-1)
        anything = occupied.copy()
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

        # nonzero is row-major: frame order, then legacy index order, which
        # within a frame is tail slot order.
        f, i = np.nonzero(tail)
        tails.append((start + f, slots[f, i], i, ~valid[f, i]))
        if longterm is not None:
            seg_id = longterm[start:stop].astype(np.int64)
            f, i = np.nonzero(valid & (seg_id != embed))
            mismatches.append((start + f, i, embed[f, i], seg_id[f, i]))
        if track is not None and legacy_track is not None:
            placed = np.where(slots >= 0, _read_tracklets(layout, track, start, stop), 0)
            _refuse_shared_tracklets(layout, start, placed, slots)
            # tracklet_id is masked by slot_occupied, so a placed but
            # unoccupied instance gets 0 (no tracklet): a reader never sees a
            # tracklet with a hole in it, and the split below runs over
            # occupied frames (spec 0029-B §5.3).
            legacy_track[start:stop] = np.where(occupied, placed, 0)

    slot_count = identity + most_tails
    placed_frame, placed_instance = np.nonzero(slot_of >= 0)
    placed_slot = slot_of[placed_frame, placed_instance]

    tracklet_id = None
    splits = _NO_SPLITS
    if legacy_track is not None:
        tracklet_id = np.zeros((frames, slot_count), dtype=np.uint32)
        tracklet_id[placed_frame, placed_slot] = legacy_track[placed_frame, placed_instance]
        splits = _split_tracklets(layout, tracklet_id)

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
        tail_slots=TailSlots(*_columns(tails, (np.uint32, np.int16, np.int16, np.bool_))),
        longterm_mismatches=LongtermMismatches(
            *_columns(mismatches, (np.uint32, np.int16, np.int64, np.int64))
        ),
        contour_count=count,
        contour_length=length,
        contour_source=source,
        group_offsets=group_offsets,
        instance_offsets=instance_offsets,
        tracklet_id=tracklet_id,
        tracklet_splits=splits,
    )


def _single_plan(layout: LegacyLayout) -> PlacementPlan:
    """Return the plan for a file with no instance axis: its one instance in slot 0."""
    return PlacementPlan(
        slots=1,
        slot_of=np.zeros((layout.frames, 1), dtype=np.int16),
        tail_slots=_NO_TAILS,
        longterm_mismatches=_NO_MISMATCHES,
        contour_count=None,
        contour_length=None,
        contour_source=None,
        group_offsets=None,
        instance_offsets=None,
        tracklet_id=None,
        tracklet_splits=_NO_SPLITS,
    )


def _track_plan(h5: h5py.File, layout: LegacyLayout, block_frames: int) -> PlacementPlan:
    """Return the plan for a ``pose_est_v3`` file: each track in one slot for its run.

    Only the first ``instance_count`` legacy instances of a frame hold tracks,
    compacted to the front, so a track's legacy index shifts when an earlier
    one ends; its slot does not. Slots come from the free list the legacy v3
    reader used to build its identity map: at each frame, the slots of
    tracks gone since the last frame are freed, then each new track, in legacy
    index order, takes the lowest free slot. A slot is therefore reused the
    frame after its track's last, and slot ``i`` is the legacy reader's
    identity ``i``. Unlike that reader, a track id that ends and comes back
    starts a new run in a fresh slot, and the tracklet split gives it a new id.

    The stored tracklet id is the legacy id plus ``TRACKLET_ID_OFFSET``,
    masked by ``slot_occupied`` and split into unbroken runs as for v4 on.
    """
    pose = h5["poseest"]
    frames, instances, slots = layout.frames, layout.instances, layout.track_slots
    _check_extents(layout)
    track_ids = pose["instance_track_id"]

    slot_of = np.full((frames, instances), -1, dtype=np.int16)
    tracklet_id = np.zeros((frames, slots), dtype=np.uint32)
    free = list(range(slots))
    slot_of_track: dict[int, int] = {}
    previous: list[int] = []
    for start in range(0, frames, block_frames):
        stop = min(start + block_frames, frames)
        count = pose["instance_count"][start:stop].astype(np.int64)
        track = _read_tracklets(layout, track_ids, start, stop)
        counted = np.arange(instances) < count[:, np.newaxis]
        _refuse_repeated_tracks(layout, start, np.where(counted, track.astype(np.int64), -1))
        _refuse_unstorable_track(layout, start, track, counted)
        block = slot_of[start:stop]
        for f in range(stop - start):
            current = track[f, : count[f]].tolist()
            if current != previous:
                ended = slot_of_track.keys() - set(current)
                for gone in ended:
                    heapq.heappush(free, slot_of_track.pop(gone))
                for track_id in current:
                    if track_id not in slot_of_track:
                        # Never empty: a frame's tracks are distinct and at
                        # most track_slots of them.
                        slot_of_track[track_id] = heapq.heappop(free)
            block[f, : count[f]] = [slot_of_track[t] for t in current]
            previous = current
        occupied = (pose["confidence"][start:stop] > 0).any(axis=-1) & counted
        frame, instance = np.nonzero(occupied)
        tracklet_id[start + frame, block[frame, instance]] = (
            track[frame, instance] + TRACKLET_ID_OFFSET
        )

    return PlacementPlan(
        slots=slots,
        slot_of=slot_of,
        tail_slots=_NO_TAILS,
        longterm_mismatches=_NO_MISMATCHES,
        contour_count=None,
        contour_length=None,
        contour_source=None,
        group_offsets=None,
        instance_offsets=None,
        tracklet_id=tracklet_id,
        tracklet_splits=_split_tracklets(layout, tracklet_id),
    )


def _refuse_repeated_tracks(
    layout: LegacyLayout, start: int, track: npt.NDArray[np.int64]
) -> None:
    """Refuse the first frame where one track id is on two counted instances; -1 is uncounted."""
    ordered = np.sort(track, axis=1)
    repeated = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
    if repeated.any():
        frame = int(np.nonzero(repeated.any(axis=1))[0][0])
        track_id = int(ordered[frame, 1:][repeated[frame]][0])
        holders = np.flatnonzero(track[frame] == track_id).tolist()
        raise LegacyConversionError(
            f"{layout.path}: frame {start + frame} has track {track_id} on legacy instances "
            f"{holders}; a track is one animal, in one instance per frame"
        )


def _refuse_unstorable_track(
    layout: LegacyLayout,
    start: int,
    track: npt.NDArray[np.uint32],
    counted: npt.NDArray[np.bool_],
) -> None:
    """Refuse a counted track id with no ``id + TRACKLET_ID_OFFSET`` in uint32."""
    unstorable = counted & (track.astype(np.int64) > _TRACKLET_MAX - TRACKLET_ID_OFFSET)
    if unstorable.any():
        frame, instance = (int(i[0]) for i in np.nonzero(unstorable))
        raise LegacyConversionError(
            f"{layout.path}: poseest/instance_track_id holds track id {track[frame, instance]} "
            f"at frame {start + frame}, which plus {TRACKLET_ID_OFFSET} (tracklet 0 means none) "
            f"would pass {_TRACKLET_MAX}, the uint32 maximum"
        )


def _columns(
    blocks: list[tuple[npt.NDArray[np.generic], ...]], dtypes: tuple[type[np.generic], ...]
) -> list[npt.NDArray[np.generic]]:
    """Concatenate each column of per-block rows into one array of its dtype."""
    return [
        np.concatenate([np.empty(0, dtype), *(block[n] for block in blocks)]).astype(dtype)
        for n, dtype in enumerate(dtypes)
    ]


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


def _read_tracklets(
    layout: LegacyLayout, track: h5py.Dataset, start: int, stop: int
) -> npt.NDArray[np.uint32]:
    """Read a block of ``instance_track_id`` as uint32, refusing any id the cast would wrap."""
    block = track[start:stop]
    name = f"{layout.path}: poseest/instance_track_id"
    if block.dtype.kind not in "iu":
        raise LegacyConversionError(f"{name} has dtype {block.dtype}; expected integers")
    if block.size and block.dtype.kind == "i" and block.min() < 0:
        frame = start + int(np.nonzero((block < 0).any(axis=1))[0][0])
        raise LegacyConversionError(f"{name} holds a negative tracklet id at frame {frame}")
    if block.size and block.max() > _TRACKLET_MAX:
        frame = start + int(np.nonzero((block > _TRACKLET_MAX).any(axis=1))[0][0])
        raise LegacyConversionError(
            f"{name} holds a tracklet id above {_TRACKLET_MAX}, the uint32 maximum, "
            f"at frame {frame}"
        )
    return block.astype(np.uint32)


def _split_tracklets(layout: LegacyLayout, tracklet_id: npt.NDArray[np.uint32]) -> TrackletSplits:
    """Split each id's occupied frames into unbroken runs, in place; return the new ids' rows.

    The first run of an id keeps it. Each later run, across all ids in order
    of first frame, takes the next id above the largest written one. A move to
    another slot is not a gap: runs are over frames.
    """
    frame, slot = np.nonzero(tracklet_id)
    if frame.size == 0:
        return _NO_SPLITS
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
    if largest + later.size > _TRACKLET_MAX:
        raise LegacyConversionError(
            f"{layout.path}: splitting tracklets needs {later.size} new ids above the largest, "
            f"{largest}, which would pass {_TRACKLET_MAX}, the uint32 maximum"
        )
    # Checked above, so the casts cannot wrap.
    given = largest + 1 + np.arange(later.size, dtype=np.int64)
    splits = TrackletSplits(
        id=given.astype(np.uint32),
        from_id=run_ids[later].astype(np.uint32),
        first_frame=frame[starts[later]].astype(np.uint32),
        last_frame=frame[ends[later]].astype(np.uint32),
    )
    if later.size:
        run_ids[later] = given
        run_of = np.cumsum(new_run) - 1
        tracklet_id[frame, slot] = run_ids[run_of].astype(np.uint32)
    return splits


def _offsets(sizes: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint64]:
    """Return the uint64 offsets of consecutive ``sizes``: 0, then their running total."""
    offsets = np.zeros(sizes.size + 1, dtype=np.uint64)
    np.cumsum(sizes, dtype=np.uint64, out=offsets[1:])
    return offsets
