"""Pass 1 places every legacy instance in a slot, packs its contours and splits its tracklets."""

import re
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file.convert import (
    LegacyConversionError,
    PlacementPlan,
    inspect_legacy,
    plan_placement,
)
from jabs.io.internal.pose_file.convert.plan import (
    LongtermMismatches,
    TailSlots,
    TrackletSplits,
)

from .conftest import SAMPLES, generator, legacy, requires_samples


def _plan(path: Path, **options: int) -> PlacementPlan:
    """Inspect and plan one legacy file."""
    layout = inspect_legacy(path)
    with h5py.File(path, "r") as h5:
        return plan_placement(h5, layout, **options)


def _write(path: Path, edits: Callable[[h5py.File], None], **overrides: object) -> Path:
    """Write a small clean v8 file with ``edits`` applied, and return its path."""
    gen = generator()
    options: dict[str, object] = {**gen.SMALL, "version": gen.V8, "seg": False, **overrides}
    gen.write_legacy(path, edits=edits, **options)
    return path


def _rows(records: TailSlots | LongtermMismatches | TrackletSplits) -> list[dict[str, object]]:
    """Return a plan's columnar records as one dict per row, for comparison."""
    columns = {
        name: getattr(records, name).tolist() for name in type(records).__dataclass_fields__
    }
    count = len(records)
    assert all(len(values) == count for values in columns.values())
    return [{name: values[n] for name, values in columns.items()} for n in range(count)]


def _identity_slot(instance: int) -> int:
    """Return the identity slot a clean v8-edge-cases instance is placed in."""
    gen = generator()
    return gen.clean_embed_id(instance, gen.EDGE_IDENTITIES) - 1


@pytest.fixture(scope="module")
def edge() -> PlacementPlan:
    """The plan of ``v8-edge-cases.h5``."""
    return _plan(legacy("v8-edge-cases.h5"))


def test_identity_slots_and_tails(edge: PlacementPlan) -> None:
    """Valid in-range instances take their identity slot; the rest go to tails in order."""
    gen = generator()
    tail = gen.EDGE_IDENTITIES
    # One case per frame, so no frame holds more than one tail instance.
    assert edge.slots == gen.EDGE_IDENTITIES + 1

    expected = []
    for name, masked in (
        ("embed_zero", False),
        ("embed_above_identities", False),
        ("masked_confident", True),
        ("masked_contour", True),
    ):
        (frame,), (instance,) = gen.EDGE[name].frames, gen.EDGE[name].instances
        assert edge.slot_of[frame, instance] == tail
        expected.append({"frame": frame, "slot": tail, "legacy_index": instance, "masked": masked})
    assert _rows(edge.tail_slots) == expected
    # Arrays, not records: an unusual file can have a tail in most frames.
    tails = edge.tail_slots
    assert len(tails) == 4
    assert (tails.frame.dtype, tails.slot.dtype) == (np.uint32, np.int16)
    assert (tails.legacy_index.dtype, tails.masked.dtype) == (np.int16, np.bool_)

    # Every other instance holding an identity is in its identity slot.
    frame = 0
    assert edge.slot_of[frame].tolist() == [_identity_slot(i) for i in range(3)] + [-1]
    # An empty instance is not placed.
    for frame in gen.EDGE["tracklet_gap"].frames:
        assert edge.slot_of[frame, gen.EDGE["tracklet_gap"].instances[0]] == -1
    assert edge.slot_of.dtype == np.int16
    assert edge.slot_of.shape == (gen.EDGE_FRAMES, gen.EDGE_INSTANCES)


def test_tails_follow_legacy_index_order(tmp_path: Path) -> None:
    """Out-of-range and masked tails share one legacy index order (ADR 0002:1300-1303)."""
    frame = 7

    def edits(h5: h5py.File) -> None:
        pose = h5["poseest"]
        # Instance 0 is masked with confidence; instance 2 is valid with embed id 0.
        pose["id_mask"][frame, 0] = True
        pose["instance_embed_id"][frame, 2] = 0

    plan = _plan(_write(tmp_path / "order.h5", edits))

    identity = generator().SMALL["identities"]
    assert plan.slots == identity + 2
    assert plan.slot_of[frame, 0] == identity
    assert plan.slot_of[frame, 2] == identity + 1
    assert plan.tail_slots.legacy_index.tolist() == [0, 2]
    assert plan.tail_slots.masked.tolist() == [True, False]


@pytest.mark.parametrize("placed", [True, False])
def test_identity_zero_places_only_tails(tmp_path: Path, placed: bool) -> None:
    """With no identity every placed instance is a tail, and an empty file has no slots."""
    frame = 3

    def edits(h5: h5py.File) -> None:
        pose = h5["poseest"]
        if placed:
            pose["id_mask"][frame, 1] = False
            pose["confidence"][frame, 1] = 0.9

    path = _write(
        tmp_path / "zero.h5",
        edits,
        identities=0,
        omit={"instance_id_center"},
    )
    layout = inspect_legacy(path)
    assert layout.identity == 0
    with h5py.File(path, "r") as h5:
        plan = plan_placement(h5, layout)

    if placed:
        assert plan.slots == 1
        assert plan.slot_of[frame, 1] == 0
        assert np.count_nonzero(plan.slot_of >= 0) == 1
        assert _rows(plan.tail_slots) == [
            {"frame": frame, "slot": 0, "legacy_index": 1, "masked": False}
        ]
    else:
        assert plan.slots == 0
        assert (plan.slot_of == -1).all()
        assert len(plan.tail_slots) == 0
    assert plan.tracklet_id is not None
    assert plan.tracklet_id.shape == (layout.frames, plan.slots)


def test_duplicates_are_refused() -> None:
    """Two valid instances with one identity in a frame are refused, naming both."""
    gen = generator()
    with pytest.raises(LegacyConversionError) as caught:
        _plan(legacy("v8-duplicate-embed.h5"))
    message = str(caught.value)
    assert "v8-duplicate-embed.h5" in message
    assert re.search(rf"\bframe {gen.DUPLICATE_EMBED.frames[0]}\b", message)
    assert re.search(rf"\bidentity {gen.DUPLICATE_EMBED_ID}\b", message)


def test_contours_are_packed(edge: PlacementPlan) -> None:
    """A contour gap keeps legacy order, packed to the front so the count is a prefix."""
    gen = generator()
    (frame,), (instance,) = gen.EDGE["contour_gap"].frames, gen.EDGE["contour_gap"].instances
    slot = _identity_slot(instance)
    assert edge.contour_count is not None
    assert edge.contour_length is not None
    assert edge.contour_source is not None

    assert edge.contour_count[frame, slot] == 1
    assert edge.contour_source[frame, slot, 0] == 1
    assert edge.contour_length[frame, slot, 0] == 5
    assert (edge.contour_source[frame, slot, 1:] == -1).all()
    assert (edge.contour_length[frame, slot, 1:] == 0).all()

    # Packing holds everywhere: used positions first, sources increasing.
    used = edge.contour_source >= 0
    assert (used == (np.arange(used.shape[-1]) < edge.contour_count[..., None])).all()
    assert ((edge.contour_length > 0) == used).all()
    # Unused positions get increasing fillers above any source, so only an
    # out-of-order source can fail.
    filler = 128 + np.arange(used.shape[-1], dtype=np.int16)
    sources = np.where(used, edge.contour_source.astype(np.int16), filler)
    assert (np.diff(sources, axis=-1) > 0).all()
    assert edge.contour_count.dtype == np.uint32
    assert edge.contour_length.dtype == np.uint32
    assert edge.contour_source.dtype == np.int8
    assert edge.contour_length.shape == (gen.EDGE_FRAMES, edge.slots, gen.CONTOURS)


def test_contours_follow_the_instance(edge: PlacementPlan) -> None:
    """Each placed instance's contour lengths come from its own legacy index."""
    with h5py.File(legacy("v8-edge-cases.h5"), "r") as h5:
        x = h5["poseest/seg_data"][..., 0]
    assert edge.contour_length is not None
    assert edge.contour_source is not None
    frames, instances = np.nonzero(edge.slot_of >= 0)
    for f, i in zip(frames.tolist(), instances.tolist(), strict=True):
        s = edge.slot_of[f, i]
        for position, c in enumerate(edge.contour_source[f, s].tolist()):
            if c < 0:
                break
            length = int(np.argmax(x[f, i, c] == -1)) if (x[f, i, c] == -1).any() else x.shape[-1]
            assert edge.contour_length[f, s, position] == length


def test_offsets_match_counts(edge: PlacementPlan) -> None:
    """Instance offsets follow contour counts; group offsets follow valid lengths in order."""
    assert edge.instance_offsets is not None
    assert edge.group_offsets is not None
    assert edge.contour_count is not None
    assert edge.contour_length is not None
    assert edge.contour_source is not None

    for offsets in (edge.instance_offsets, edge.group_offsets):
        assert offsets.dtype == np.uint64
        assert offsets[0] == 0
    assert len(edge.instance_offsets) == edge.contour_count.size + 1
    assert (np.diff(edge.instance_offsets) == edge.contour_count.ravel()).all()
    lengths = edge.contour_length[edge.contour_source >= 0]
    assert len(edge.group_offsets) == lengths.size + 1
    assert (np.diff(edge.group_offsets) == lengths).all()
    assert edge.instance_offsets[-1] == lengths.size


def test_zero_confidence_instance_keeps_contours(edge: PlacementPlan) -> None:
    """A valid instance with every confidence 0 stays in its slot, with its contours."""
    gen = generator()
    case = gen.EDGE["zero_confidence_contours"]
    (frame,), (instance,) = case.frames, case.instances
    slot = _identity_slot(instance)
    assert edge.slot_of[frame, instance] == slot
    assert edge.contour_count is not None
    assert edge.contour_count[frame, slot] > 0


def test_masked_contour_instance_keeps_contours(edge: PlacementPlan) -> None:
    """A masked instance placed for its contour carries that contour in its tail slot."""
    gen = generator()
    (frame,), _ = gen.EDGE["masked_contour"].frames, gen.EDGE["masked_contour"].instances
    assert edge.contour_count is not None
    assert edge.contour_count[frame, gen.EDGE_IDENTITIES] == 1


def test_longterm_mismatches_are_recorded(edge: PlacementPlan) -> None:
    """A valid instance whose longterm_seg_id differs from its embed id is recorded."""
    gen = generator()
    case = gen.EDGE["shared_longterm"]
    first, second = case.instances
    identities = gen.EDGE_IDENTITIES
    assert _rows(edge.longterm_mismatches) == [
        {
            "frame": case.frames[0],
            "legacy_index": second,
            "embed_id": gen.clean_embed_id(second, identities),
            "longterm_seg_id": gen.clean_embed_id(first, identities),
        },
    ]
    assert edge.longterm_mismatches.frame.dtype == np.uint32


def _runs(frames: npt.NDArray[np.intp]) -> int:
    """Return how many runs of consecutive frames ``frames`` forms."""
    return 0 if frames.size == 0 else int(np.count_nonzero(np.diff(frames) != 1)) + 1


def test_tracklet_gap_is_split(edge: PlacementPlan) -> None:
    """A tracklet that leaves and returns, or goes unoccupied, takes a new id above the largest.

    Runs are over occupied frames (spec 0029-B §5.3): a placed instance with
    every confidence 0 breaks its tracklet as an absent one does, and its
    ``(frame, slot)`` carries tracklet 0.
    """
    gen = generator()
    gap = gen.EDGE["tracklet_gap"]
    instance = gap.instances[0]
    original = gen.clean_track_id(instance)
    largest = gen.clean_track_id(gen.EDGE_IDENTITIES - 1)
    returns = gap.frames[-1] + 1
    unoccupied = gen.EDGE["unoccupied_in_tracklet"]
    (hole,), (unoccupied_instance,) = unoccupied.frames, unoccupied.instances
    assert edge.tracklet_id is not None

    # Instance 0 hops into a tail slot at frames 10 and 20 and keeps its
    # tracklet: a slot change is not a gap. Its unoccupied frame is one.
    assert _rows(edge.tracklet_splits) == [
        {
            "id": largest + 1,
            "from_id": original,
            "first_frame": returns,
            "last_frame": gen.EDGE_FRAMES - 1,
        },
        {
            "id": largest + 2,
            "from_id": gen.clean_track_id(unoccupied_instance),
            "first_frame": hole + 1,
            "last_frame": gen.EDGE_FRAMES - 1,
        },
    ]
    slot = _identity_slot(instance)
    assert edge.tracklet_id[gap.frames[0] - 1, slot] == original
    assert edge.tracklet_id[returns, slot] == largest + 1

    unoccupied_slot = _identity_slot(unoccupied_instance)
    assert edge.slot_of[hole, unoccupied_instance] == unoccupied_slot
    assert edge.tracklet_id[hole - 1, unoccupied_slot] == gen.clean_track_id(unoccupied_instance)
    assert edge.tracklet_id[hole, unoccupied_slot] == 0
    assert edge.tracklet_id[hole + 1, unoccupied_slot] == largest + 2

    # The zero-confidence instance in the last frame is placed but unoccupied.
    case = gen.EDGE["zero_confidence_contours"]
    (frame,), (zero,) = case.frames, case.instances
    assert edge.slot_of[frame, zero] == _identity_slot(zero)
    assert edge.tracklet_id[frame, _identity_slot(zero)] == 0
    assert edge.tracklet_id.dtype == np.uint32
    assert edge.tracklet_id.shape == (gen.EDGE_FRAMES, edge.slots)

    for tracklet in np.unique(edge.tracklet_id[edge.tracklet_id > 0]).tolist():
        frames, _ = np.nonzero(edge.tracklet_id == tracklet)
        assert _runs(frames) == 1, tracklet


def test_tracklet_in_two_slots_is_refused() -> None:
    """One tracklet id in two slots of a frame is refused, naming the frame and id."""
    gen = generator()
    with pytest.raises(LegacyConversionError) as caught:
        _plan(legacy("v8-tracklet-two-slots.h5"))
    message = str(caught.value)
    assert "v8-tracklet-two-slots.h5" in message
    assert re.search(rf"\bframe {gen.TWO_SLOT_TRACKLET.frames[0]}\b", message)
    assert re.search(rf"\btracklet {gen.clean_track_id(0)}\b", message)


_UINT32_MAX = int(np.iinfo(np.uint32).max)


def _retrack(dtype: type[np.integer], value: int, frame: int = 5) -> Callable[[h5py.File], None]:
    """Return edits storing ``instance_track_id`` as ``dtype``, with ``value`` at one instance."""

    def edits(h5: h5py.File) -> None:
        track = h5["poseest/instance_track_id"][()].astype(dtype)
        track[frame, 1] = value
        del h5["poseest/instance_track_id"]
        h5["poseest"].create_dataset("instance_track_id", data=track)

    return edits


@pytest.mark.parametrize(
    ("dtype", "value", "match"),
    [
        pytest.param(np.int32, -1, r"negative", id="negative"),
        pytest.param(np.int64, -(2**40), r"negative", id="negative-int64"),
        pytest.param(np.uint64, _UINT32_MAX + 1, r"above", id="above-uint32"),
    ],
)
def test_tracklet_ids_that_would_wrap_are_refused(
    tmp_path: Path, dtype: type[np.integer], value: int, match: str
) -> None:
    """A tracklet id that is negative or above the uint32 range is refused, not wrapped."""
    path = _write(tmp_path / "wrap.h5", _retrack(dtype, value))
    with pytest.raises(LegacyConversionError, match=match) as caught:
        _plan(path)
    assert "poseest/instance_track_id" in str(caught.value)
    assert "wrap.h5" in str(caught.value)


def _split_near_the_top(largest: int) -> Callable[[h5py.File], None]:
    """Return edits giving instance 0 tracklet ``largest`` and instance 1 one gap."""

    def edits(h5: h5py.File) -> None:
        h5["poseest/instance_track_id"][:, 0] = largest
        # Unoccupied for one frame, so its tracklet splits once.
        h5["poseest/confidence"][50, 1] = 0

    return edits


def test_new_tracklet_ids_may_reach_the_uint32_maximum(tmp_path: Path) -> None:
    """A split whose new id is exactly the uint32 maximum is kept."""
    plan = _plan(_write(tmp_path / "top.h5", _split_near_the_top(_UINT32_MAX - 1)))
    assert plan.tracklet_splits.id.tolist() == [_UINT32_MAX]
    assert plan.tracklet_id is not None
    assert plan.tracklet_id.max() == _UINT32_MAX


def test_new_tracklet_ids_past_the_uint32_maximum_are_refused(tmp_path: Path) -> None:
    """A split that needs an id above the uint32 maximum is refused rather than wrapped."""
    path = _write(tmp_path / "over.h5", _split_near_the_top(_UINT32_MAX))
    with pytest.raises(LegacyConversionError, match="over.h5.*tracklet") as caught:
        _plan(path)
    assert str(_UINT32_MAX) in str(caught.value)


def test_without_tracklets_or_segmentation() -> None:
    """A file without instance_track_id or seg_data plans no tracklets and no contours."""
    plan = _plan(legacy("v8-without-centers.h5"))
    assert plan.tracklet_id is None
    assert len(plan.tracklet_splits) == 0
    assert plan.contour_count is None
    assert plan.contour_length is None
    assert plan.contour_source is None
    assert plan.group_offsets is None
    assert plan.instance_offsets is None
    assert len(plan.longterm_mismatches) == 0


def test_no_contours_has_empty_offsets() -> None:
    """Segmentation holding no contour anywhere plans zero counts and a lone 0 offset."""
    plan = _plan(legacy("v8-no-contours.h5"))
    assert plan.contour_count is not None
    assert plan.group_offsets is not None
    assert plan.instance_offsets is not None
    assert (plan.contour_count == 0).all()
    assert plan.group_offsets.tolist() == [0]
    assert (plan.instance_offsets == 0).all()


def test_block_size_does_not_change_the_plan(edge: PlacementPlan) -> None:
    """Blocks that cut through the tracklet gap and the cases give the same plan."""
    small = _plan(legacy("v8-edge-cases.h5"), block_frames=50)
    for name in PlacementPlan.__dataclass_fields__:
        ours, theirs = getattr(small, name), getattr(edge, name)
        if isinstance(ours, np.ndarray):
            assert ours.dtype == theirs.dtype, name
            assert np.array_equal(ours, theirs), name
        elif hasattr(ours, "__dataclass_fields__"):
            for column in type(ours).__dataclass_fields__:
                mine, other = getattr(ours, column), getattr(theirs, column)
                assert mine.dtype == other.dtype, (name, column)
                assert np.array_equal(mine, other), (name, column)
        else:
            assert ours == theirs, name


@requires_samples
def test_sample_v6_plans() -> None:
    """The v6 sample places every identity and splits its one gapped tracklet once."""
    layout = inspect_legacy(SAMPLES / "sample_pose_est_v6.h5")
    with h5py.File(layout.path, "r") as h5:
        plan = plan_placement(h5, layout)
    assert plan.slots >= layout.identity
    assert len(plan.tracklet_splits) == 1


def test_masked_instances_sharing_a_tracklet_are_still_refused(tmp_path: Path) -> None:
    """A valid instance's tracklet wins over a masked one's; two masked ones are refused."""

    def two_masked(h5: h5py.File) -> None:
        """Give spare instances 3 and 4 pose data and one tracklet id, both masked."""
        pose = h5["poseest"]
        for instance in (3, 4):
            pose["confidence"][5, instance] = pose["confidence"][5, 0]
            pose["instance_track_id"][5, instance] = 99

    path = _write(tmp_path / "two-masked.h5", two_masked, instances=5, identities=3)
    with pytest.raises(LegacyConversionError, match=r"frame 5 has tracklet 99"):
        _plan(path)
