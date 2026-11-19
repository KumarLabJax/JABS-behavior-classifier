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
    for name, origin in (
        ("embed_zero", "embed_out_of_range"),
        ("embed_above_identities", "embed_out_of_range"),
        ("masked_confident", "masked"),
        ("masked_contour", "masked"),
    ):
        (frame,), (instance,) = gen.EDGE[name].frames, gen.EDGE[name].instances
        assert edge.slot_of[frame, instance] == tail
        expected.append({"frame": frame, "slot": tail, "origin": origin, "legacy_index": instance})
    assert list(edge.tail_slots) == expected

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
    assert [(t["legacy_index"], t["origin"]) for t in plan.tail_slots] == [
        (0, "masked"),
        (2, "embed_out_of_range"),
    ]


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
        assert plan.tail_slots == (
            {"frame": frame, "slot": 0, "origin": "embed_out_of_range", "legacy_index": 1},
        )
    else:
        assert plan.slots == 0
        assert (plan.slot_of == -1).all()
        assert plan.tail_slots == ()
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
    assert edge.longterm_mismatches == (
        {
            "frame": case.frames[0],
            "legacy_index": second,
            "embed_id": gen.clean_embed_id(second, identities),
            "longterm_seg_id": gen.clean_embed_id(first, identities),
        },
    )


def _runs(frames: npt.NDArray[np.intp]) -> int:
    """Return how many runs of consecutive frames ``frames`` forms."""
    return 0 if frames.size == 0 else int(np.count_nonzero(np.diff(frames) != 1)) + 1


def test_tracklet_gap_is_split(edge: PlacementPlan) -> None:
    """A tracklet that leaves and returns becomes two ids, the second above the largest."""
    gen = generator()
    gap = gen.EDGE["tracklet_gap"]
    instance = gap.instances[0]
    original = gen.clean_track_id(instance)
    largest = gen.clean_track_id(gen.EDGE_IDENTITIES - 1)
    returns = gap.frames[-1] + 1
    assert edge.tracklet_id is not None

    # Instance 0 hops into a tail slot at frames 10 and 20 and keeps its
    # tracklet: a slot change is not a gap.
    assert edge.tracklet_splits == (
        {
            "id": largest + 1,
            "from": original,
            "first_frame": returns,
            "last_frame": gen.EDGE_FRAMES - 1,
        },
    )
    slot = _identity_slot(instance)
    assert edge.tracklet_id[gap.frames[0] - 1, slot] == original
    assert edge.tracklet_id[returns, slot] == largest + 1
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


def test_without_tracklets_or_segmentation() -> None:
    """A file without instance_track_id or seg_data plans no tracklets and no contours."""
    plan = _plan(legacy("v8-without-centers.h5"))
    assert plan.tracklet_id is None
    assert plan.tracklet_splits == ()
    assert plan.contour_count is None
    assert plan.contour_length is None
    assert plan.contour_source is None
    assert plan.group_offsets is None
    assert plan.instance_offsets is None
    assert plan.longterm_mismatches == ()


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
