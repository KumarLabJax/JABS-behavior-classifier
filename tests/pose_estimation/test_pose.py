"""Tests for :class:`jabs.pose_estimation.pose.Pose` over tiny unified pose files."""

import gc
import json
import logging
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pytest
from shapely.geometry import Point

import jabs.io.internal.pose_file as pose_file_package
from jabs.core.enums import PoseCapability
from jabs.core.pose_protocol import PoseLike
from jabs.core.pose_skeleton import (
    FULL_CONNECTED_SEGMENTS,
    NVSN_CONNECTED_SEGMENTS,
    KeypointIndex,
)
from jabs.io.internal.pose_file import reader as pose_file_reader
from jabs.pose_estimation.pose import Pose

from .conftest import KEYPOINTS, make_points, make_segmentation

Factory = Callable[..., Path]

_TAIL = [KeypointIndex.MID_TAIL.value, KeypointIndex.TIP_TAIL.value]


def _valid(frames: int, slots: int) -> np.ndarray:
    """All-valid ``(F, S, K)`` point validity."""
    return np.ones((frames, slots, KEYPOINTS), dtype=bool)


def test_satisfies_the_protocol(pose_file_factory: Factory) -> None:
    """Pose is structurally a PoseLike."""
    with Pose(pose_file_factory()) as pose:
        assert isinstance(pose, PoseLike)


def test_invalid_points_are_nan_and_masked(pose_file_factory: Factory) -> None:
    """An invalid point is NaN in every accessor and False in the mask."""
    valid = _valid(6, 2)
    valid[2, 1, KeypointIndex.NOSE.value] = False
    with Pose(pose_file_factory(point_valid=valid)) as pose:
        points, mask = pose.get_identity_poses(1)
        assert points.dtype == np.float32
        assert mask.dtype == np.bool_
        assert np.isnan(points[2, KeypointIndex.NOSE.value]).all()
        assert not mask[2, KeypointIndex.NOSE.value]
        assert mask.sum() == 6 * KEYPOINTS - 1
        frame_points, frame_mask = pose.get_points(2, 1)
        assert np.isnan(frame_points[KeypointIndex.NOSE.value]).all()
        np.testing.assert_array_equal(frame_mask, mask[2])
        np.testing.assert_array_equal(pose.get_identity_point_mask(1), mask)
        # valid coordinates come through, and scale applies
        expected = make_points(6, 2)[3, 1]
        np.testing.assert_array_equal(pose.get_points(3, 1)[0], expected)
        np.testing.assert_allclose(pose.get_points(3, 1, scale=0.5)[0], expected * 0.5)
        np.testing.assert_allclose(pose.get_identity_poses(1, scale=2.0)[0][3], expected * 2)


def test_slot_usable_is_used_when_present(pose_file_factory: Factory) -> None:
    """The file's slot_usable decides presence, even against the point counts."""
    usable = np.ones((6, 2), dtype=bool)
    usable[1, 0] = False
    with Pose(pose_file_factory(slot_usable=usable)) as pose:
        mask = pose.identity_mask(0)
        assert mask.dtype == np.bool_
        np.testing.assert_array_equal(mask, usable[:, 0])
        assert pose.get_points(1, 0) == (None, None)


def test_derived_presence_excludes_tail_keypoints(pose_file_factory: Factory) -> None:
    """Without slot_usable, presence needs 3 valid keypoints other than the tail ones."""
    valid = np.zeros((6, 1, KEYPOINTS), dtype=bool)
    valid[0, 0, :3] = True  # three body keypoints: present
    valid[1, 0, :2] = True  # two body keypoints and both tail keypoints: absent
    valid[1, 0, _TAIL] = True
    valid[2, 0, :2] = True  # two body keypoints: absent
    with Pose(pose_file_factory(slots=1, identity=1, point_valid=valid)) as pose:
        np.testing.assert_array_equal(
            pose.identity_mask(0), [True, False, False, False, False, False]
        )


def test_tail_slots_are_not_identities(pose_file_factory: Factory) -> None:
    """Only slots below dimensions.identity become identities."""
    with Pose(pose_file_factory(slots=3, identity=2)) as pose:
        assert pose.identities == [0, 1]
        assert pose.num_identities == 2


def test_identity_zero_with_tracklets_makes_every_slot_an_identity(
    pose_file_factory: Factory,
) -> None:
    """A v3 conversion: each slot is an identity, tracks are tracklet_id - 1."""
    tracklets = np.array([[1, 3], [1, 3], [2, 0]] * 2, dtype=np.uint32)
    valid = _valid(6, 2)
    valid[2, 1] = False  # unoccupied: no track
    path = pose_file_factory(
        identity=0,
        tracklet_id=tracklets,
        point_valid=valid,
        convert_extra={"tracklet_id_offset": 1},
        convert_format="pose_est_v3",
    )
    with Pose(path) as pose:
        assert pose.identities == [0, 1]
        track = pose.identity_to_track
        assert track is not None
        assert track.dtype == np.int32
        assert track.shape == (2, 6)
        np.testing.assert_array_equal(track[0], [0, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(track[1], [2, 2, -1, 2, 2, -1])


def test_identity_to_track_undoes_tracklet_splits(pose_file_factory: Factory) -> None:
    """A run the converter gave a new tracklet id reports the track id it split from."""
    # track 4 (stored 5) in slot 0; its second run got tracklet 9
    tracklets = np.array([[5], [5], [0], [9], [9], [9]], dtype=np.uint32)
    valid = _valid(6, 1)
    valid[2] = False
    path = pose_file_factory(
        slots=1,
        identity=0,
        tracklet_id=tracklets,
        point_valid=valid,
        convert_extra={
            "tracklet_id_offset": 1,
            "tracklet_splits": {"id": [9], "from": [5], "first_frame": [3], "last_frame": [5]},
        },
        convert_format="pose_est_v3",
    )
    with Pose(path) as pose:
        np.testing.assert_array_equal(pose.identity_to_track[0], [4, 4, -1, 4, 4, 4])


def test_identity_to_track_is_none_when_identities_are_not_tracks(
    pose_file_factory: Factory,
) -> None:
    """A file with long-term identities has no track mapping, even with tracklets."""
    tracklets = np.ones((6, 2), dtype=np.uint32)
    with Pose(pose_file_factory(tracklet_id=tracklets)) as pose:
        assert pose.identity_to_track is None
    with Pose(pose_file_factory(name="b_pose.h5")) as pose:
        assert pose.identity_to_track is None


@pytest.mark.parametrize(
    "convert",
    [None, "pose_est_v4", "pose_est_v8", "pose_est_v3"],
    ids=["native", "converted-v4", "converted-v8", "v3-version-without-offset"],
)
def test_identity_zero_with_tracklets_not_from_v3_has_no_identities(
    pose_file_factory: Factory, convert: str | None
) -> None:
    """Only a v3 conversion makes slots identities; other identity = 0 files have none.

    A v4+ file whose instances are all masked converts to identity = 0 and keeps
    its tracklets; the legacy readers gave it no identities. It may still record
    version 3.0 (as the public v5 sample does), so only the converter's
    ``tracklet_id_offset`` marks a v3 conversion.
    """
    tracklets = np.ones((6, 2), dtype=np.uint32)
    if convert is None:
        path = pose_file_factory(identity=0, tracklet_id=tracklets)
    else:
        path = pose_file_factory(
            identity=0, tracklet_id=tracklets, convert_extra={}, convert_format=convert
        )
    with Pose(path) as pose:
        assert pose.identities == []
        assert pose.identity_to_track is None


def test_identity_zero_without_tracklets_has_no_identities(pose_file_factory: Factory) -> None:
    """identity = 0 without tracklets means no identities at all."""
    with Pose(pose_file_factory(identity=0)) as pose:
        assert pose.identities == []
        assert pose.num_identities == 0
        assert pose.get_reduced_point_mask().shape == (KEYPOINTS,)
        assert not pose.get_reduced_point_mask().any()


def test_fps_from_the_manifest(pose_file_factory: Factory) -> None:
    """fps is the manifest's video.fps."""
    with Pose(pose_file_factory(fps=25.0)) as pose:
        assert pose.fps == 25


def test_fps_defaults_to_30_with_one_info_log(
    pose_file_factory: Factory, caplog: pytest.LogCaptureFixture
) -> None:
    """Without video.fps, fps is 30, logged once at INFO."""
    with (
        caplog.at_level(logging.INFO, logger="jabs.pose_estimation.pose"),
        Pose(pose_file_factory(fps=None)) as pose,
    ):
        assert pose.fps == 30
        assert pose.fps == 30
    assert sum("fps" in r.getMessage() for r in caplog.records) == 1


def test_fps_default_log_names_the_source_file(
    pose_file_factory: Factory, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """The default-fps log names the user's file, not a converted copy."""
    converted = pose_file_factory(name="vid.abc-1.0.h5", fps=None)
    source = tmp_path / "user_pose_est_v6.h5"
    with (
        caplog.at_level(logging.INFO, logger="jabs.pose_estimation.pose"),
        Pose(converted, source_path=source, source_hash="abc"),
    ):
        pass
    messages = [r.getMessage() for r in caplog.records if "fps" in r.getMessage()]
    assert len(messages) == 1
    assert source.name in messages[0]
    assert converted.name not in messages[0]


def test_hash_failure_opens_no_file(
    pose_file_factory: Factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the source cannot be hashed, no reader is left open."""
    opened: list[object] = []
    real_reader = pose_file_package.ComponentReader

    def tracking_reader(*args: object, **kwargs: object) -> object:
        reader = real_reader(*args, **kwargs)
        opened.append(reader)
        return reader

    monkeypatch.setattr("jabs.pose_estimation.pose.ComponentReader", tracking_reader)
    path = pose_file_factory()
    with pytest.raises(OSError):
        Pose(path, source_path=tmp_path / "missing_pose_est_v6.h5")
    assert opened == []


def test_video_scale(pose_file_factory: Factory) -> None:
    """cm_per_pixel is the manifest's value, None when absent."""
    with Pose(pose_file_factory(cm_per_pixel=0.125)) as pose:
        assert pose.cm_per_pixel == 0.125
        assert isinstance(pose.cm_per_pixel, float)
    with Pose(pose_file_factory(name="b_pose.h5")) as pose:
        assert pose.cm_per_pixel is None


def test_static_objects_squeeze_single_point_objects(pose_file_factory: Factory) -> None:
    """P == 1 objects lose their point axis; num_lixit_keypoints is P of lixit."""
    corners = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
    lixit = np.arange(4, dtype=np.float32).reshape(2, 1, 2)
    path = pose_file_factory(static_objects={"corners": corners, "lixit": lixit})
    with Pose(path) as pose:
        assert pose.static_objects["corners"].shape == (4, 2)
        assert pose.static_objects["corners"].dtype == np.float32
        np.testing.assert_array_equal(pose.static_objects["lixit"], lixit[:, 0])
        assert pose.num_lixit_keypoints == 1
        assert pose.has_static_object("corners")
        assert not pose.has_static_object("food_hopper")


def test_three_point_lixit_is_unchanged(pose_file_factory: Factory) -> None:
    """A (1, 3, 2) lixit keeps its shape and has 3 keypoints."""
    lixit = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    with Pose(pose_file_factory(static_objects={"lixit": lixit})) as pose:
        np.testing.assert_array_equal(pose.static_objects["lixit"], lixit)
        assert pose.num_lixit_keypoints == 3
    with Pose(pose_file_factory(name="b_pose.h5")) as pose:
        assert pose.num_lixit_keypoints == 0
        assert pose.static_objects == {}


def test_one_animal_by_format_is_not_social(pose_file_factory: Factory) -> None:
    """A file with room for one animal (one slot, one identity) has no capabilities."""
    with Pose(pose_file_factory(slots=1, identity=1)) as pose:
        assert pose.capabilities == frozenset()
        assert not pose.has_social
        assert not pose.has_segmentation
        assert not pose.has_bounding_boxes
        assert pose.get_bounding_boxes(0) is None


def test_one_identity_with_tail_slots_is_social(pose_file_factory: Factory) -> None:
    """A file that can hold several animals is social even with one identity, as on main."""
    with Pose(pose_file_factory(slots=3, identity=1)) as pose:
        assert pose.num_identities == 1
        assert pose.capabilities == frozenset({PoseCapability.SOCIAL})
        assert pose.has_social


@pytest.mark.parametrize(
    ("source_name", "social"),
    [
        ("vid_pose_est_v2.h5", False),
        ("vid_pose_est_v3.h5", True),
        ("vid_pose_est_v4.h5", True),
        ("vid_pose_est_v8.h5", True),
    ],
)
def test_legacy_source_is_social_from_v3(
    pose_file_factory: Factory, source_name: str, social: bool
) -> None:
    """A conversion is social as its legacy file was on main: by the name's version, 3 on."""
    path = pose_file_factory(slots=1, identity=1)
    with Pose(path, source_path=path.parent / source_name, source_hash="h") as pose:
        assert pose.has_social is social
        assert (PoseCapability.SOCIAL in pose.capabilities) is social


@pytest.mark.parametrize(
    ("convert_format", "social"),
    [("pose_est_v2", False), ("pose_est_v4", True), ("pose_est", False)],
)
def test_renamed_conversion_is_social_by_its_source_format(
    pose_file_factory: Factory, convert_format: str, social: bool
) -> None:
    """A conversion under a unified name follows its recorded source format, when it has one."""
    path = pose_file_factory(slots=1, identity=1, convert_extra={}, convert_format=convert_format)
    with Pose(path) as pose:
        assert pose.has_social is social


def test_capabilities_of_a_full_file(pose_file_factory: Factory) -> None:
    """Two identities, segmentation and boxes give every capability."""
    with Pose(pose_file_factory(segmentation=True, bbox=True)) as pose:
        assert pose.capabilities == frozenset(PoseCapability)
        assert pose.has_social and pose.has_segmentation and pose.has_bounding_boxes


def test_bounding_boxes(pose_file_factory: Factory) -> None:
    """Boxes come back per identity, (F, 2, 2) float32."""
    with Pose(pose_file_factory(bbox=True)) as pose:
        boxes = pose.get_bounding_boxes(1)
        assert boxes.shape == (6, 2, 2)
        assert boxes.dtype == np.float32
        np.testing.assert_array_equal(boxes[4], [[4, 1], [5, 2]])
        with pytest.raises(ValueError, match="identity"):
            pose.get_bounding_boxes(2)


def test_connected_segments_full_and_nvsn(pose_file_factory: Factory) -> None:
    """Without any keypoint NVSN pose lacks, the NVSN segments are used."""
    with Pose(pose_file_factory()) as pose:
        assert pose.get_connected_segments() == FULL_CONNECTED_SEGMENTS
    valid = _valid(6, 2)
    for missing in (
        KeypointIndex.BASE_NECK,
        KeypointIndex.LEFT_FRONT_PAW,
        KeypointIndex.RIGHT_FRONT_PAW,
        KeypointIndex.CENTER_SPINE,
        KeypointIndex.LEFT_REAR_PAW,
        KeypointIndex.RIGHT_REAR_PAW,
        KeypointIndex.MID_TAIL,
    ):
        valid[..., missing.value] = False
    with Pose(pose_file_factory(name="nvsn_pose.h5", point_valid=valid)) as pose:
        assert pose.get_connected_segments() == NVSN_CONNECTED_SEGMENTS
        assert not pose.get_reduced_point_mask()[KeypointIndex.BASE_NECK.value]


def test_reduced_point_mask_ignores_tail_slots(pose_file_factory: Factory) -> None:
    """A keypoint valid only in a tail slot is not in the reduced mask."""
    valid = _valid(6, 3)
    valid[:, :2, KeypointIndex.BASE_NECK.value] = False
    with Pose(pose_file_factory(slots=3, identity=2, point_valid=valid)) as pose:
        assert not pose.get_reduced_point_mask()[KeypointIndex.BASE_NECK.value]


def test_bearings(pose_file_factory: Factory) -> None:
    """Bearings follow compute_bearing, NaN where the identity is absent."""
    usable = np.ones((6, 2), dtype=bool)
    usable[0, 0] = False
    with Pose(pose_file_factory(slot_usable=usable)) as pose:
        bearings = pose.compute_all_bearings(0)
        assert bearings.dtype == np.float32
        assert np.isnan(bearings[0])
        points, _ = pose.get_identity_poses(0)
        assert bearings[3] == pytest.approx(pose.compute_bearing(points[3]))


def test_dynamic_objects(pose_file_factory: Factory) -> None:
    """Dynamic objects come back with the legacy dtypes."""
    frame_index = np.array([0, 3], dtype=np.uint32)
    points = np.arange(2 * 3 * 1 * 2, dtype=np.float32).reshape(2, 3, 1, 2)
    counts = np.array([3, 1], dtype=np.uint32)
    path = pose_file_factory(dynamic_objects={"fecal_boli": (frame_index, points, counts)})
    with Pose(path) as pose:
        boli = pose.get_dynamic_object("fecal_boli")
        assert pose.dynamic_objects == {"fecal_boli": boli}
        assert boli.sample_indices.dtype == np.int64
        assert boli.points.dtype == np.float64
        assert boli.counts.dtype == np.int64
        np.testing.assert_array_equal(boli.sample_indices, frame_index)
        np.testing.assert_array_equal(boli.points, points)
        np.testing.assert_array_equal(boli.counts, counts)
        assert pose.get_dynamic_object("nothing") is None


def test_external_identities(pose_file_factory: Factory) -> None:
    """External ids name identities for display."""
    with Pose(pose_file_factory(external_ids=["mouse_a", "mouse_b"])) as pose:
        assert pose.external_identities == ["mouse_a", "mouse_b"]
        assert pose.identity_index_to_display(1) == "mouse_b"
        assert pose.identity_index_to_display(5) == "5"
    with Pose(pose_file_factory(name="b_pose.h5")) as pose:
        assert pose.external_identities is None
        assert pose.identity_index_to_display(0) == "0"


def test_never_reads_the_whole_file(pose_file_factory: Factory, monkeypatch) -> None:
    """Pose reads components, never read_pose_file."""
    path = pose_file_factory(segmentation=True, bbox=True)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("read_pose_file was called")

    monkeypatch.setattr(pose_file_reader, "read_pose_file", refuse)
    monkeypatch.setattr(pose_file_package, "read_pose_file", refuse)
    with Pose(path) as pose:
        pose.segmentation_window(0, 0, 2)
        assert pose.model_metadata is None


def test_segmentation_window_matches_the_file(pose_file_factory: Factory) -> None:
    """A window holds the identity's contours, across a 300-frame boundary too."""
    path = pose_file_factory(frames=605, segmentation=True)
    seg = make_segmentation(605, 2)
    with Pose(path) as pose:
        window = pose.segmentation_window(1, 295, 305)
        assert window.start == 295
        assert window.contours.dtype == np.int32
        assert window.contour_count.dtype == np.uint32
        assert window.contour_length.dtype == np.uint32
        assert window.external_flag.dtype == np.bool_
        np.testing.assert_array_equal(window.contours, seg["contours"][295:305, 1])
        np.testing.assert_array_equal(window.contour_count, seg["count"][295:305, 1])
        np.testing.assert_array_equal(window.contour_length, seg["length"][295:305, 1])
        np.testing.assert_array_equal(window.external_flag, seg["flags"][295:305, 1])
        empty = pose.segmentation_window(0, 10, 10)
        assert empty.contours.shape[0] == 0
        assert empty.contour_count.shape == (0,)


def test_iter_segmentation_tiles_the_file(pose_file_factory: Factory) -> None:
    """iter_segmentation covers every frame in blocks."""
    path = pose_file_factory(frames=605, segmentation=True)
    seg = make_segmentation(605, 2)
    with Pose(path) as pose:
        windows = list(pose.iter_segmentation(0))
        assert [w.start for w in windows] == [0, 300, 600]
        counts = np.concatenate([w.contour_count for w in windows])
        np.testing.assert_array_equal(counts, seg["count"][:, 0])
        assert [w.start for w in pose.iter_segmentation(0, block=250)] == [0, 250, 500]


def test_segmentation_validates_eagerly(pose_file_factory: Factory) -> None:
    """Bad arguments raise on the call, not on the first next()."""
    with Pose(pose_file_factory(segmentation=True)) as pose:
        with pytest.raises(ValueError, match="identity"):
            pose.iter_segmentation(2)
        with pytest.raises(ValueError, match="block"):
            pose.iter_segmentation(0, block=0)
        with pytest.raises(ValueError, match="identity"):
            pose.segmentation_window(-1, 0, 1)
        with pytest.raises(ValueError, match="frame range"):
            pose.segmentation_window(0, 3, 7)


def test_segmentation_absent_raises(pose_file_factory: Factory) -> None:
    """Without segmentation both members raise NotImplementedError."""
    with Pose(pose_file_factory()) as pose:
        with pytest.raises(NotImplementedError):
            pose.segmentation_window(0, 0, 1)
        with pytest.raises(NotImplementedError):
            pose.iter_segmentation(0)


def test_segmentation_windows_share_one_read(pose_file_factory: Factory, monkeypatch) -> None:
    """Ten single-frame windows for two identities read each dataset once."""
    reads: list[str] = []
    original = pose_file_reader.ComponentReader.read

    def counting(self, component_id: str, frames: slice | None = None) -> np.ndarray:
        reads.append(component_id)
        return original(self, component_id, frames)

    monkeypatch.setattr(pose_file_reader.ComponentReader, "read", counting)
    with Pose(pose_file_factory(frames=20, segmentation=True)) as pose:
        reads.clear()
        for frame in range(10):
            for identity in (0, 1):
                pose.segmentation_window(identity, frame, frame + 1)
    assert sorted(reads) == sorted(
        [
            "jabs.segmentation.contours",
            "jabs.segmentation.contour_count",
            "jabs.segmentation.contour_length",
            "jabs.segmentation.external_flag",
        ]
    )


def test_hash_is_the_passed_source_hash(pose_file_factory: Factory) -> None:
    """A given source hash is used as is."""
    with Pose(pose_file_factory(), source_hash="abc123") as pose:
        assert pose.hash == "abc123"


def test_source_path_names_the_users_file(pose_file_factory: Factory, tmp_path: Path) -> None:
    """pose_file and source_format follow the file the user has."""
    legacy = tmp_path / "vid_pose_est_v6.h5"
    legacy.write_bytes(b"legacy")
    with Pose(pose_file_factory(name="converted.h5"), source_path=legacy) as pose:
        assert pose.pose_file == legacy
        assert pose.source_format == "pose_est_v6"
        assert pose.hash  # hashed from the legacy file
    with Pose(pose_file_factory()) as pose:
        assert pose.source_format == "jabs.pose-file"


def test_model_metadata_from_the_convert_entry(pose_file_factory: Factory) -> None:
    """model_metadata parses the convert entry's preserved model_metadata_json."""
    raw = json.dumps({"model": "hrnet", "epoch": 3})
    path = pose_file_factory(convert_extra={"model_metadata_json": raw})
    with Pose(path) as pose:
        assert pose.model_metadata == {"model": "hrnet", "epoch": 3}
    path = pose_file_factory(name="b_pose.h5", convert_extra={"model_metadata_json": "nope"})
    with Pose(path) as pose:
        assert pose.model_metadata is None
    with Pose(pose_file_factory(name="c_pose.h5", convert_extra={})) as pose:
        assert pose.model_metadata is None


def test_convex_hulls_are_cached_by_source_hash(pose_file_factory: Factory, tmp_path) -> None:
    """Hulls are cached under the source hash, so a new hash recomputes them."""
    cache = tmp_path / "cache"
    usable = np.ones((6, 2), dtype=bool)
    usable[0, 0] = False
    valid = _valid(6, 2)
    valid[0, 0] = False
    path = pose_file_factory(point_valid=valid, slot_usable=usable)
    with Pose(path, source_hash="h1", cache_dir=cache) as pose:
        hulls = pose.get_identity_convex_hulls(0)
    assert len(hulls) == 6
    assert hulls[0] is None
    assert hulls[1] is not None
    cached = cache / "convex_hulls" / "vid" / "h1" / "convex_hulls_0.pickle"
    assert cached.exists()

    # a readable cache is used as is, under its own hash only
    joblib.dump(["sentinel"] * 6, cached)
    with Pose(path, source_hash="h1", cache_dir=cache) as pose:
        assert pose.get_identity_convex_hulls(0) == ["sentinel"] * 6
    with Pose(path, source_hash="h2", cache_dir=cache) as pose:
        assert pose.get_identity_convex_hulls(0)[1] is not None
    assert (cache / "convex_hulls" / "vid" / "h2" / "convex_hulls_0.pickle").exists()

    # an unreadable cache is recomputed
    cached.write_bytes(b"not a pickle")
    with Pose(path, source_hash="h1", cache_dir=cache) as pose:
        assert pose.get_identity_convex_hulls(0)[1] is not None


def test_convex_hull_algorithm_omits_the_tail(pose_file_factory: Factory) -> None:
    """A hull uses only valid non-tail keypoints and needs three of them."""
    valid = np.zeros((6, 1, KEYPOINTS), dtype=bool)
    valid[:, 0, :4] = True
    valid[1, 0, 2:4] = False  # two body points left
    valid[1, 0, _TAIL] = True
    usable = np.ones((6, 1), dtype=bool)
    path = pose_file_factory(slots=1, identity=1, point_valid=valid, slot_usable=usable)
    with Pose(path) as pose:
        hulls = pose.get_identity_convex_hulls(0)
        assert hulls[1] is None
        points, _ = pose.get_identity_poses(0)
        assert hulls[0].equals(hulls[0].convex_hull)
        for x, y in points[0, :4]:
            assert hulls[0].buffer(1e-3).contains(Point(x, y))


def test_close_is_idempotent_and_finalized(pose_file_factory: Factory) -> None:
    """close releases the file; a dropped Pose closes it too."""
    path = pose_file_factory(segmentation=True)
    pose = Pose(path)
    pose.close()
    pose.close()
    # eagerly loaded data stays available
    assert pose.identity_mask(0).all()

    pose = Pose(path)
    finalizer = pose._finalizer
    del pose
    gc.collect()
    assert not finalizer.alive


def test_empty_lixit_is_not_a_static_object(pose_file_factory: Factory) -> None:
    """A lixit with no objects is dropped, as the legacy readers and the converter drop it.

    Kept, it enabled lixit features that index its first object and crash.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    lixit = np.zeros((0, 3, 2), dtype=np.float32)
    with Pose(pose_file_factory(static_objects={"corners": corners, "lixit": lixit})) as pose:
        assert set(pose.static_objects) == {"corners"}
        assert pose.num_lixit_keypoints == 0
