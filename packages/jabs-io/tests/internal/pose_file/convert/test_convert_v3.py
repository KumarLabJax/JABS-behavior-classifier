"""``convert_legacy`` converts a ``pose_est_v3`` file: tracklets, and no identity.

ADR 0002's conversion table and the design spec: ``dimensions.identity`` is 0,
each legacy track keeps one slot for its whole run, assigned by the free list
of the legacy v3 reader's identity map (a freed slot is reused
from the frame after its track's last, lowest first), and a legacy track id is
stored as ``id + 1``, since tracklet 0 means none. A track id that ends and
comes back starts a new run in a fresh slot, and takes a new id.
"""

import json
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
from jabs.io.internal.pose_file.validate import validate

from .conftest import LEGACY, SAMPLES, generator, legacy, requires_samples

CREATED = "2026-11-20T00:00:00+00:00"
FIXTURE = "v3-tracklets.h5"
COMPONENTS = {
    "jabs.pose.points",
    "jabs.pose.confidence",
    "jabs.pose.point_valid",
    "jabs.pose.slot_occupied",
    "jabs.pose.tracklet_id",
}

# The slot each run of generate.V3_RUNS takes, by the legacy free list:
# track 0 and 5 take 0 and 1 at frame 0, track 7 takes 2 at frame 5; track 9
# takes 7's slot 2 at frame 15, the frame after 7 ends; track 11 takes 0's
# slot at frame 22; track 7 comes back at frame 25 to the only free slot, 3.
RUN_SLOTS = (0, 1, 2, 2, 0, 3)
# The stored id of each run: legacy id + 1; the second run of track 7 is a
# split, and so is track 5 after its unoccupied frame (V3_UNOCCUPIED).
SPLITS = {"id": [13, 14], "from": [6, 8], "first_frame": [11, 25], "last_frame": [39, 34]}

# Frames of tracks that end and come back, which the legacy v3 reader mishandled
# (it kept the first run's identity and stacked the runs' frames), excluded
# from the golden comparison. Neither public sample has one; the recording
# script asserted that when it recorded the goldens.
REAPPEARING_FRAMES: dict[str, tuple[int, ...]] = {
    "sample_pose_est_v3": (),
    "identity_with_no_data_pose_est_v3": (),
}
# What the legacy v3 reader (``jabs/pose_estimation/pose_est_v3.py``) read from
# the two public v3 samples. The archive is data: it was recorded by the script
# ``v3_sample_expected.py`` beside it, committed in 13cb266e, which needed that
# reader. The reader and the script no longer exist, so it cannot be re-recorded.
# For each sample, keyed ``<file stem>/<name>``, it holds:
#   identity_to_track (I, F) int32: the legacy track id of each identity, -1 for none
#   point_mask (I, F, 12) bool: confidence above 0.3
#   identity_mask (I, F) bool: at least one body keypoint valid
#   frames: every 10th frame, the frames ``points`` holds
#   points (I, len(frames), 12, 2) float32 (x, y); the samples store uint16, held exactly
EXPECTED = LEGACY / "v3-sample-expected.npz"


def _component(path: Path, component_id: str) -> npt.NDArray[np.generic]:
    """Read one written component's dense data."""
    with h5py.File(path, "r") as h5:
        return h5[component_id.replace(".", "/")][()]


def _convert_entry(path: Path) -> dict:
    """Read the written file's ``convert`` history entry."""
    with h5py.File(path, "r") as h5:
        history = json.loads(h5["provenance"][()])["history"]
    (entry,) = [e for e in history if e["operation"] == "convert"]
    return entry


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic v3 file, converted."""
    out = tmp_path_factory.mktemp("v3") / "out.h5"
    convert_legacy(legacy(FIXTURE), out, created=CREATED)
    return out


def _expected_tracklets() -> npt.NDArray[np.uint32]:
    """Build the expected ``tracklet_id`` from the runs, before occupancy and splits."""
    g = generator()
    expected = np.zeros((g.V3_FRAMES, 4), dtype=np.uint32)
    for (track, first, last), slot in zip(g.V3_RUNS, RUN_SLOTS, strict=True):
        expected[first : last + 1, slot] = track + 1
    return expected


def test_inspect_reads_v3_as_tracks_without_identity() -> None:
    """A v3 file has no identity; its track slots are its largest instance count."""
    layout = inspect_legacy(legacy(FIXTURE))

    assert layout.placed_by_track is True
    assert layout.identity == 0
    assert layout.track_slots == 4
    assert layout.instances == generator().V3_INSTANCES
    assert layout.has_tracklets
    assert layout.version == (3, 0)
    assert layout.dropped == ("poseest/instance_count", "poseest/instance_embedding")


def test_other_versions_are_not_placed_by_track() -> None:
    """Files from v4 on, and v2, place by identity and need no track slots."""
    for name in ("v8-with-centers.h5", "v2-single.h5"):
        layout = inspect_legacy(legacy(name))
        assert layout.placed_by_track is False
        assert layout.track_slots == 0


def test_converted_v3_is_deep_valid(converted: Path) -> None:
    """The output passes the deep check and holds the pose data and tracklets only."""
    assert [f for f in validate(converted, deep=True) if f.severity == "error"] == []
    with h5py.File(converted, "r") as h5:
        components = set()
        h5.visititems(
            lambda name, obj: components.add(name.replace("/", "."))
            if isinstance(obj, h5py.Dataset) and name.startswith("jabs/")
            else None
        )
    assert components == COMPONENTS


def test_report_dimensions(tmp_path: Path) -> None:
    """Identity 0; slots are the most concurrent tracks, not the legacy instance axis."""
    report = convert_legacy(legacy(FIXTURE), tmp_path / "out.h5", created=CREATED)

    assert report.dimensions == {"frame": generator().V3_FRAMES, "slot": 4, "identity": 0}
    assert report.tail_slots == report.longterm_mismatches == 0
    assert report.tracklet_splits == 2
    assert report.dropped == ("poseest/instance_count", "poseest/instance_embedding")


def test_each_run_keeps_its_free_list_slot(converted: Path) -> None:
    """Every run sits in its free-list slot, as legacy id + 1, wherever it is occupied."""
    g = generator()
    expected = _expected_tracklets()
    frame, track = g.V3_UNOCCUPIED
    expected[frame, RUN_SLOTS[1]] = 0
    # Split ids, in the order of the runs' first frames.
    expected[frame + 1 :, RUN_SLOTS[1]] = 13
    expected[25:35, 3] = 14

    np.testing.assert_array_equal(_component(converted, "jabs.pose.tracklet_id"), expected)
    np.testing.assert_array_equal(_component(converted, "jabs.pose.slot_occupied"), expected > 0)


def test_track_zero_is_stored_as_one(converted: Path) -> None:
    """Legacy track 0 is a track, stored as 1, since tracklet 0 means none."""
    tracklet = _component(converted, "jabs.pose.tracklet_id")
    first, last = generator().V3_RUNS[0][1:]

    assert (tracklet[first : last + 1, 0] == 1).all()
    assert not (tracklet[last + 1 :] == 1).any()


def test_each_tracklet_has_one_slot_and_no_gap(converted: Path) -> None:
    """A stored tracklet sits in one slot on consecutive frames.

    A slot's tracklet changes only where the previous one has ended, which can
    be between two occupied frames: the free list reuses a slot the frame
    after its track ends (track 9 after track 7 here; the public sample does
    the same at frame 141, slot 1, and frame 1622, slot 0).
    """
    tracklet = _component(converted, "jabs.pose.tracklet_id")
    for tracklet_id in np.unique(tracklet[tracklet > 0]):
        frames, slots = np.nonzero(tracklet == tracklet_id)
        assert np.unique(slots).size == 1, tracklet_id
        assert (np.diff(frames) == 1).all(), tracklet_id
    frame, slot = np.nonzero((tracklet[1:] != tracklet[:-1]) & (tracklet[:-1] > 0))
    for f, s in zip(frame, slot, strict=True):
        assert tracklet[f, s] not in tracklet[f + 1], (f, s)
    # The handover from track 7 to track 9 is between two occupied frames.
    assert (tracklet[14, 2], tracklet[15, 2]) == (8, 10)


def test_points_follow_the_track_through_compaction(converted: Path) -> None:
    """A track's points come from whichever instance holds it in each frame, flipped."""
    g = generator()
    with h5py.File(legacy(FIXTURE), "r") as h5:
        points = h5["poseest/points"][()]
        confidence = h5["poseest/confidence"][()]
        track = h5["poseest/instance_track_id"][()]
        count = h5["poseest/instance_count"][()]
    written = _component(converted, "jabs.pose.points")
    written_confidence = _component(converted, "jabs.pose.confidence")

    for (track_id, first, last), slot in zip(g.V3_RUNS, RUN_SLOTS, strict=True):
        for frame in range(first, last + 1):
            (instance,) = np.flatnonzero(track[frame, : count[frame]] == track_id)
            has = confidence[frame, instance] > 0
            np.testing.assert_array_equal(
                written[frame, slot][has], points[frame, instance][has][:, ::-1]
            )
            assert np.isnan(written[frame, slot][~has]).all()
            np.testing.assert_array_equal(
                written_confidence[frame, slot], confidence[frame, instance]
            )
    # Track 5 moves from instance 1 to 0 when track 0 ends.
    assert track[19, 1] == track[20, 0] == 5


def test_instances_past_the_count_are_not_read(converted: Path) -> None:
    """The stale instance past ``instance_count`` is left out, though it has confidence."""
    stale = generator().V3_STALE_FRAME
    occupied = _component(converted, "jabs.pose.slot_occupied")

    with h5py.File(legacy(FIXTURE), "r") as h5:
        count = int(h5["poseest/instance_count"][stale])
        assert (h5["poseest/confidence"][stale, count] > 0).any()
    assert occupied[stale].sum() == count


def test_convert_entry_records_the_offset(converted: Path) -> None:
    """The entry records the +1 offset, tracklet 0 as none, the splits and the dropped data."""
    entry = _convert_entry(converted)

    assert entry["source"]["format"] == "pose_est_v3"
    assert entry["extra"]["tracklet_id_offset"] == 1
    assert entry["extra"]["tracklet_id_none"] == 0
    assert entry["extra"]["tracklet_splits"] == SPLITS
    assert entry["dropped"] == ["poseest/instance_count", "poseest/instance_embedding"]


def test_v4_files_record_no_offset(tmp_path: Path) -> None:
    """Files from v4 on store their tracklet ids as they are."""
    out = tmp_path / "out.h5"
    convert_legacy(legacy("v8-with-centers.h5"), out, created=CREATED)

    assert "tracklet_id_offset" not in _convert_entry(out)["extra"]


def _edit_fixture(path: Path, name: str, frame: int, value: object) -> Path:
    """Copy the fixture to ``path`` with one value of one ``poseest`` dataset changed."""
    path.write_bytes(legacy(FIXTURE).read_bytes())
    with h5py.File(path, "a") as h5:
        h5["poseest"][name][frame] = value
    return path


def test_repeated_track_in_a_frame_is_refused(tmp_path: Path) -> None:
    """One track id on two counted instances of a frame is refused, naming both."""
    src = _edit_fixture(tmp_path / "x.h5", "instance_track_id", 3, [5, 5, 0, 0, 0])

    with pytest.raises(
        LegacyConversionError, match=r"frame 3 has track 5 on legacy instances \[0, 1\]"
    ):
        convert_legacy(src, tmp_path / "out.h5")


def test_count_above_the_instance_axis_is_refused(tmp_path: Path) -> None:
    """An ``instance_count`` above the instances ``points`` has is refused."""
    src = _edit_fixture(tmp_path / "x.h5", "instance_count", 7, 6)

    with pytest.raises(LegacyConversionError, match=r"instance_count holds 6 at frame 7"):
        inspect_legacy(src)


def test_largest_track_id_is_refused(tmp_path: Path) -> None:
    """A track id of 2**32 - 1 has no ``id + 1`` in uint32, so it is refused."""
    src = _edit_fixture(tmp_path / "x.h5", "instance_track_id", 3, [0, 2**32 - 1, 0, 0, 0])

    with pytest.raises(LegacyConversionError, match=r"track id 4294967295 at frame 3"):
        convert_legacy(src, tmp_path / "out.h5")


def test_v3_with_another_dataset_is_refused(tmp_path: Path) -> None:
    """A v3 file holding a dataset from v4 on is refused, naming it."""
    src = tmp_path / "x.h5"
    src.write_bytes(legacy(FIXTURE).read_bytes())
    with h5py.File(src, "a") as h5:
        h5["poseest"].create_dataset("bbox", data=np.zeros((40, 5, 2, 2), dtype=np.float32))

    with pytest.raises(LegacyConversionError, match="poseest/bbox is not in a pose_est_v3 file"):
        inspect_legacy(src)


def test_v3_without_tracks_is_refused(tmp_path: Path) -> None:
    """A file with neither v4 dataset nor ``instance_track_id`` is refused, naming what is missing."""
    src = tmp_path / "x.h5"
    src.write_bytes(legacy(FIXTURE).read_bytes())
    with h5py.File(src, "a") as h5:
        del h5["poseest/instance_track_id"]

    with pytest.raises(LegacyConversionError, match="poseest/instance_track_id"):
        inspect_legacy(src)


@requires_samples
@pytest.mark.parametrize("stem", sorted(REAPPEARING_FRAMES))
def test_public_v3_samples_match_todays_reader(tmp_path: Path, stem: str) -> None:
    """Slot i is the legacy v3 reader's identity i: same tracks, masks and points.

    The masks agree exactly because neither sample has a confidence of
    exactly float32 0.3, which the legacy reader compared in float32 and the
    converter in float64. ``slot_occupied`` matches the track mask because
    neither sample has a counted instance with every confidence 0.
    """
    out = tmp_path / "out.h5"
    report = convert_legacy(SAMPLES / f"{stem}.h5", out, created=CREATED)
    expected = np.load(EXPECTED)
    to_track = expected[f"{stem}/identity_to_track"]
    keep = np.setdiff1d(np.arange(to_track.shape[1]), REAPPEARING_FRAMES[stem])

    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []
    assert report.dimensions["identity"] == 0
    assert report.dimensions["slot"] == to_track.shape[0]
    tracklet = _component(out, "jabs.pose.tracklet_id").T.astype(np.int64)
    occupied = _component(out, "jabs.pose.slot_occupied").T
    point_valid = _component(out, "jabs.pose.point_valid").transpose(1, 0, 2)
    np.testing.assert_array_equal(tracklet[:, keep] - 1, to_track[:, keep])
    np.testing.assert_array_equal(occupied[:, keep], to_track[:, keep] >= 0)
    np.testing.assert_array_equal(point_valid[:, keep], expected[f"{stem}/point_mask"][:, keep])
    np.testing.assert_array_equal(
        point_valid[:, keep, :10].any(axis=-1), expected[f"{stem}/identity_mask"][:, keep]
    )

    frames = expected[f"{stem}/frames"]
    frames = frames[np.isin(frames, keep)]
    sampled = np.isin(expected[f"{stem}/frames"], frames)
    points = _component(out, "jabs.pose.points").transpose(1, 0, 2, 3)[:, frames]
    golden = expected[f"{stem}/points"][:, sampled]
    mask = expected[f"{stem}/point_mask"][:, frames]
    np.testing.assert_array_equal(points[mask], golden[mask])
