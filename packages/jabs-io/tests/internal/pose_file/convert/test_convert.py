"""``convert_legacy`` writes a valid pose file and records where it came from."""

import hashlib
import importlib
import json
import shutil
import tracemalloc
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.convert import (
    ConversionReport,
    LegacyConversionError,
    VideoOverrides,
    convert_legacy,
    inspect_legacy,
    plan_placement,
)
from jabs.io.internal.pose_file.convert.convert import block_frames_for
from jabs.io.internal.pose_file.convert.inspect import BLOCK_FRAMES
from jabs.io.internal.pose_file.convert.provenance import legacy_hash
from jabs.io.internal.pose_file.reader import read_pose_file
from jabs.io.internal.pose_file.validate import validate

from .conftest import SAMPLES, generator, legacy, requires_samples

REFUSALS = (
    "v8-duplicate-embed.h5",
    "v8-tracklet-two-slots.h5",
    "v8-unknown-dataset.h5",
    "v7-bad-dynamic-object.h5",
)
CONVERTIBLE = (
    "v7-dynamic-objects.h5",
    "v8-with-centers.h5",
    "v8-without-centers.h5",
    "v8-edge-cases.h5",
    "v8-no-contours.h5",
)
SAMPLE_NAMES = ("sample_pose_est_v4.h5", "sample_pose_est_v5.h5", "sample_pose_est_v6.h5")
CREATED = "2026-11-18T00:00:00+00:00"


def _not_json(token: str) -> None:
    """Refuse the NaN and infinity tokens strict JSON parsers (Go, JavaScript) reject."""
    raise ValueError(f"{token} is not strict JSON")


def _provenance(path: Path) -> dict:
    """Read a pose file's provenance document, as a strict JSON parser would."""
    with h5py.File(path, "r") as h5:
        return json.loads(h5["provenance"][()], parse_constant=_not_json)


def _history(path: Path) -> list[dict]:
    """Read a pose file's history entries."""
    return _provenance(path)["history"]


def _sources() -> list:
    """Every legacy file that converts: the synthetic corpus, then the samples."""
    return [pytest.param(legacy(name), id=name) for name in CONVERTIBLE] + [
        pytest.param(SAMPLES / name, id=name, marks=requires_samples) for name in SAMPLE_NAMES
    ]


@pytest.mark.parametrize("source", _sources())
def test_converted_corpus_is_valid(tmp_path: Path, source: Path) -> None:
    """Every convertible file converts to one with no errors under the deep check."""
    out = tmp_path / "out.h5"
    convert_legacy(source, out, created=CREATED)

    findings = validate(out, deep=True)
    assert [f for f in findings if f.severity == "error"] == []
    if inspect_legacy(source).has_segmentation:
        # The ragged alternate was decoded and compared, not skipped.
        assert [f for f in findings if f.check == "alternate_deep_unchecked"] == []
    read_pose_file(out)


def _assert_tracklets_are_unbroken(path: Path) -> None:
    """Assert each written tracklet id holds one unbroken run of occupied frames.

    An unoccupied ``(frame, slot)`` carries tracklet 0, so a reader that masks
    ``tracklet_id`` by ``slot_occupied`` never sees a hole inside a tracklet.
    """
    with h5py.File(path, "r") as h5:
        if "jabs/pose/tracklet_id" not in h5:
            return
        tracklet = h5["jabs/pose/tracklet_id"][()]
        occupied = h5["jabs/pose/slot_occupied"][()]
    assert (tracklet[~occupied] == 0).all()
    for tracklet_id in np.unique(tracklet[tracklet > 0]).tolist():
        frames, _ = np.nonzero(tracklet == tracklet_id)
        assert occupied[tracklet == tracklet_id].all(), tracklet_id
        assert (np.diff(frames) == 1).all(), tracklet_id


@pytest.mark.parametrize("source", _sources())
def test_converted_tracklets_are_unbroken(tmp_path: Path, source: Path) -> None:
    """Every written tracklet is one run of occupied frames; an unoccupied slot has none."""
    out = tmp_path / "out.h5"
    convert_legacy(source, out, created=CREATED)
    _assert_tracklets_are_unbroken(out)


@requires_samples
def test_unoccupied_frame_splits_a_sample_tracklet(tmp_path: Path) -> None:
    """Zeroing one instance's confidences mid-tracklet splits it there, with no hole (C1).

    The reviewer's reproduction: frame 100, instance 0 of the v6 sample.
    """
    frame, instance = 100, 0
    src = tmp_path / "sample_pose_est_v6.h5"
    shutil.copyfile(SAMPLES / "sample_pose_est_v6.h5", src)
    with h5py.File(src, "r+") as h5:
        tracklet = int(h5["poseest/instance_track_id"][frame, instance])
        assert tracklet > 0
        assert not h5["poseest/id_mask"][frame, instance]
        h5["poseest/confidence"][frame, instance] = 0
    layout = inspect_legacy(src)
    with h5py.File(src, "r") as h5:
        slot = int(plan_placement(h5, layout).slot_of[frame, instance])
    assert slot >= 0

    out = tmp_path / "out.h5"
    report = convert_legacy(src, out, created=CREATED)
    with h5py.File(out, "r") as h5:
        assert not h5["jabs/pose/slot_occupied"][frame, slot]
        assert h5["jabs/pose/tracklet_id"][frame, slot] == 0
    _assert_tracklets_are_unbroken(out)
    (entry, _) = _history(out)
    assert any(
        split["from"] == tracklet and split["first_frame"] == frame + 1
        for split in entry["extra"]["tracklet_splits"]
    )
    assert report.tracklet_splits == len(entry["extra"]["tracklet_splits"])
    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []


def test_provenance_records_the_conversion(tmp_path: Path) -> None:
    """The history says what the source was, what was invented and what was left behind."""
    src = tmp_path / "v8-with-centers.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        attrs = h5["poseest/instance_embed_id"].attrs
        attrs["tracklet_gen"] = "greedy"
        attrs["version"] = np.int64(6)
        attrs["window"] = np.array([1, 2], dtype=np.uint16)
        attrs["raw"] = np.bytes_(b"ascii")
    out = tmp_path / "out.h5"
    convert_legacy(src, out, created=CREATED)

    provenance = _provenance(out)
    assert provenance["records"] == {}
    (entry,) = provenance["history"]
    assert entry["operation"] == "convert"
    assert entry["tool"] == "jabs-io"
    assert entry["time"] == CREATED

    digest = hashlib.blake2b(src.read_bytes(), digest_size=32).hexdigest()
    assert legacy_hash(src) == f"blake2b:{digest}"
    assert entry["source"] == {
        "format": "pose_est_v8",
        "content_hash": legacy_hash(src),
        "filename": "v8-with-centers.h5",
    }

    with h5py.File(src, "r") as h5:
        stored = h5["poseest"].attrs["model_metadata_json"]
    assert entry["extra"]["model_metadata_json"] == stored
    assert json.loads(entry["extra"]["model_metadata_json"]) == generator().MODEL_METADATA

    legacy_attributes = entry["extra"]["legacy_attributes"]
    assert legacy_attributes["poseest/instance_embed_id"] == {
        "tracklet_gen": "greedy",
        "version": 6,
        "window": [1, 2],
        "raw": "ascii",
    }
    assert legacy_attributes["poseest"]["version"] == [8, 0, 0]
    assert legacy_attributes["poseest/bbox"] == {"bboxes_generated": True}
    # Only objects with attributes are listed.
    assert "poseest/points" not in legacy_attributes
    assert "/" not in legacy_attributes

    assert entry["dropped"] == ["poseest/id_mask", "poseest/instance_embed_id"]
    assert entry["synthesized"] == [
        "skeletons.jabs.mouse12",
        "jabs.pose.point_valid",
        "jabs.pose.slot_occupied",
        "dimensions.identity",
    ]
    assert entry["extra"]["point_valid"] == {
        "confidence_threshold": 0.3,
        "requires_finite_coordinates": True,
    }
    assert entry["extra"]["tail_slots"] == []
    assert entry["extra"]["tracklet_splits"] == []
    assert entry["extra"]["longterm_mismatches"] == []
    assert all(c.provenance is None for c in read_pose_file(out).components)


def test_segmentation_adds_an_encode_entry(tmp_path: Path) -> None:
    """A file with segmentation also records the ragged alternates the converter added."""
    out = tmp_path / "out.h5"
    convert_legacy(legacy("v8-edge-cases.h5"), out, created=CREATED)

    history = _history(out)
    assert [entry["operation"] for entry in history] == ["convert", "encode"]
    encode = history[1]
    assert encode["tool"] == "jabs-io"
    assert encode["time"] == CREATED
    assert encode["synthesized"] == [
        "jabs.segmentation.contours:alternate:ragged",
        "jabs.segmentation.external_flag:alternate:ragged",
    ]


def test_root_attributes_are_kept(tmp_path: Path) -> None:
    """Attributes on the root group are listed under ``/``."""
    src = tmp_path / "rooted.h5"
    shutil.copyfile(legacy("v8-without-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        h5.attrs["origin"] = b"rig 7"
        h5.attrs["scale"] = np.float32(0.5)
    out = tmp_path / "out.h5"
    convert_legacy(src, out, created=CREATED)

    (entry,) = _history(out)
    assert entry["extra"]["legacy_attributes"]["/"] == {"origin": "rig 7", "scale": 0.5}
    assert "model_metadata_json" not in entry["extra"]


def test_video_overrides(tmp_path: Path) -> None:
    """Given video fields are written and listed as synthesized; absent ones are null and noted."""
    given = tmp_path / "given.h5"
    convert_legacy(legacy("v8-with-centers.h5"), given, video=VideoOverrides(800, 800, 30.0))
    video = read_pose_file(given).video
    assert (video.width, video.height, video.fps) == (800, 800, 30.0)
    assert video.cm_per_pixel == pytest.approx(0.0794)
    (entry,) = _history(given)
    assert entry["synthesized"][-3:] == ["video.width", "video.height", "video.fps"]
    assert "notes" not in entry

    absent = tmp_path / "absent.h5"
    convert_legacy(legacy("v8-with-centers.h5"), absent)
    video = read_pose_file(absent).video
    assert (video.width, video.height, video.fps) == (None, None, None)
    assert video.filename is None
    (entry,) = _history(absent)
    assert not any(item.startswith("video.") for item in entry["synthesized"])
    for name in ("video.width", "video.height", "video.fps"):
        assert name in entry["notes"]
    assert "not recorded in legacy pose files" in entry["notes"]

    partial = tmp_path / "partial.h5"
    convert_legacy(legacy("v8-with-centers.h5"), partial, video=VideoOverrides(fps=29.97))
    (entry,) = _history(partial)
    assert entry["synthesized"][-1] == "video.fps"
    assert "video.width" in entry["notes"] and "video.fps" not in entry["notes"]


def test_tracklet_splits_are_recorded(tmp_path: Path) -> None:
    """A gapped tracklet's split is listed as synthesized and recorded."""
    out = tmp_path / "out.h5"
    convert_legacy(legacy("v8-edge-cases.h5"), out)
    (entry, _) = _history(out)
    assert "jabs.pose.tracklet_id" in entry["synthesized"]
    assert entry["extra"]["tracklet_splits"]
    assert {"id", "from", "first_frame", "last_frame"} == set(entry["extra"]["tracklet_splits"][0])
    # Legacy tracklet 0 and every unoccupied slot carry 0: no tracklet.
    assert entry["extra"]["tracklet_id_none"] == 0


def test_no_tracklets_no_tracklet_component(tmp_path: Path) -> None:
    """A file without tracklets or embeddings gets neither component and synthesizes no tracklets."""
    out = tmp_path / "out.h5"
    convert_legacy(legacy("v8-without-centers.h5"), out)
    ids = {c.id for c in read_pose_file(out).components}
    assert "jabs.pose.tracklet_id" not in ids
    assert "jabs.identity.embeddings" not in ids
    (entry,) = _history(out)
    assert "jabs.pose.tracklet_id" not in entry["synthesized"]
    assert "tracklet_id_none" not in entry["extra"]


def test_zero_identities_convert(tmp_path: Path) -> None:
    """A file with no identified instance converts, with ``dimensions.identity`` 0."""
    src = tmp_path / "anonymous.h5"

    def anonymous(h5: h5py.File) -> None:
        """Mask every instance."""
        h5["poseest/id_mask"][...] = True
        h5["poseest/instance_embed_id"][...] = 0

    generator().write_legacy(
        src,
        frames=60,
        instances=3,
        identities=3,
        version=(8, 0, 0),
        seg=False,
        edits=anonymous,
        omit={"instance_id_center"},
    )
    out = tmp_path / "out.h5"
    report = convert_legacy(src, out)
    assert report.dimensions["identity"] == 0
    assert report.dimensions["slot"] == 3
    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []


@pytest.mark.parametrize("name", REFUSALS)
def test_refusals_leave_nothing(tmp_path: Path, name: str) -> None:
    """A refused file leaves an existing destination's bytes and no temporary file."""
    src = tmp_path / name
    shutil.copyfile(legacy(name), src)
    before = src.read_bytes()
    out = tmp_path / "out.h5"
    sentinel = b"an earlier conversion"
    out.write_bytes(sentinel)

    with pytest.raises(LegacyConversionError):
        convert_legacy(src, out)

    assert out.read_bytes() == sentinel
    # The writer's temporary file is a dotfile, which a glob would skip.
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
    assert src.read_bytes() == before


@pytest.mark.parametrize(
    ("scale", "source", "written", "nulled"),
    [
        pytest.param(
            np.float32(-1),
            "corner_detection",
            (None, None),
            ("video.cm_per_pixel is null", "video.cm_per_pixel_source is null"),
            id="negative-scale",
        ),
        pytest.param(
            np.float64(np.nan),
            None,
            (None, None),
            ("video.cm_per_pixel is null",),
            id="nan-scale",
        ),
        pytest.param(
            np.float32(0.0794),
            b"eyeballed",
            (pytest.approx(0.0794), None),
            ("video.cm_per_pixel_source is null",),
            id="unknown-source",
        ),
    ],
)
def test_unusable_scale_is_unknown(
    tmp_path: Path,
    scale: np.floating,
    source: str | bytes | None,
    written: tuple[object, object],
    nulled: tuple[str, ...],
) -> None:
    """A scale or source the manifest cannot hold is written as null, noted, and kept raw."""
    src = tmp_path / "odd-scale.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        h5["poseest"].attrs["cm_per_pixel"] = scale
        if source is not None:
            h5["poseest"].attrs["cm_per_pixel_source"] = source
    out = tmp_path / "out.h5"
    convert_legacy(src, out, video=VideoOverrides(800, 800, 30.0))

    video = read_pose_file(out).video
    assert (video.cm_per_pixel, video.cm_per_pixel_source) == written
    (entry,) = _history(out)
    for note in nulled:
        assert note in entry["notes"]
    if "video.cm_per_pixel is null" in nulled:
        assert "not a positive number" in entry["notes"]
    if source == b"eyeballed":
        assert "not a recognized source" in entry["notes"]
    raw = entry["extra"]["legacy_attributes"]["poseest"]
    if np.isnan(scale):
        # Strict JSON has no NaN; the string keeps the raw value.
        assert raw["cm_per_pixel"] == "NaN"
    else:
        assert raw["cm_per_pixel"] == scale.item()
    if source is not None:
        assert raw["cm_per_pixel_source"] == (
            source.decode() if isinstance(source, bytes) else source
        )


def test_every_attribute_kind_is_recorded_as_json(tmp_path: Path) -> None:
    """Non-finite floats, empty attributes and references all get a strict JSON form."""
    src = tmp_path / "kinds.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        attrs = h5["poseest/points"].attrs
        attrs["nan"] = np.float32(np.nan)
        attrs["limits"] = np.array([np.inf, -np.inf, 1.5])
        attrs["empty"] = h5py.Empty("f4")
        attrs["target"] = h5["poseest/confidence"].ref
        attrs["nowhere"] = h5py.Reference()
        attrs["targets"] = np.array(
            [h5["poseest/bbox"].ref, h5py.Reference()], dtype=h5py.ref_dtype
        )
    out = tmp_path / "out.h5"
    convert_legacy(src, out)

    (entry,) = _history(out)
    assert entry["extra"]["legacy_attributes"]["poseest/points"] == {
        "nan": "NaN",
        "limits": ["Infinity", "-Infinity", 1.5],
        "empty": None,
        "target": "<reference:/poseest/confidence>",
        "nowhere": "<reference:null>",
        "targets": ["<reference:/poseest/bbox>", "<reference:null>"],
    }


def test_a_compound_attribute_is_refused_by_name(tmp_path: Path) -> None:
    """An attribute with no JSON form is refused, naming where it is, and nothing is written."""
    src = tmp_path / "compound.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        h5["poseest/points"].attrs["calibration"] = np.array(
            (1, 2.0), dtype=[("a", "i4"), ("b", "f4")]
        )
    out = tmp_path / "out.h5"
    with pytest.raises(LegacyConversionError, match="'calibration' of poseest/points"):
        convert_legacy(src, out)
    assert not out.exists()


def test_a_failed_deep_check_leaves_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the writer's own validation fails, the destination keeps its bytes and no temp is left."""
    deep_check = importlib.import_module("jabs.io.internal.pose_file.validate")
    finding = deep_check.Finding(deep_check.ERROR, "alternate_decodes", "planted for the test")
    monkeypatch.setattr(deep_check, "validate", lambda path, deep=False: [finding])
    out = tmp_path / "out.h5"
    sentinel = b"an earlier conversion"
    out.write_bytes(sentinel)

    with pytest.raises(LegacyConversionError, match="planted for the test"):
        convert_legacy(legacy("v8-edge-cases.h5"), out)
    assert out.read_bytes() == sentinel
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_created_is_shared_by_manifest_and_history(tmp_path: Path) -> None:
    """Without ``created``, the manifest and the history carry one timestamp."""
    out = tmp_path / "out.h5"
    convert_legacy(legacy("v8-with-centers.h5"), out)
    with h5py.File(out, "r") as h5:
        created = json.loads(h5["manifest"][()])["created"]
    (entry,) = _history(out)
    assert entry["time"] == created


def test_write_failure_is_a_conversion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file the writer rejects is refused as a conversion error naming the source."""
    module = importlib.import_module("jabs.io.internal.pose_file.convert.convert")

    def reject(*args: object, **kwargs: object) -> None:
        """Fail as the writer does on an invalid file."""
        raise ValueError("refusing to write an invalid pose file: test")

    monkeypatch.setattr(module, "write_pose_file", reject)
    with pytest.raises(LegacyConversionError, match="v8-with-centers.h5: refusing"):
        convert_legacy(legacy("v8-with-centers.h5"), tmp_path / "out.h5")


def test_report(tmp_path: Path) -> None:
    """The report describes the file that was written."""
    src = legacy("v8-edge-cases.h5")
    out = tmp_path / "out.h5"
    report = convert_legacy(src, out)

    assert isinstance(report, ConversionReport)
    assert report.source == src
    assert report.destination == out
    assert report.format == "pose_est_v8"
    written = read_pose_file(out)
    assert report.dimensions == dict(written.dimensions)
    (entry, _) = _history(out)
    assert report.tail_slots == len(entry["extra"]["tail_slots"]) > 0
    assert report.tracklet_splits == len(entry["extra"]["tracklet_splits"]) > 0
    assert report.longterm_mismatches == len(entry["extra"]["longterm_mismatches"]) > 0
    assert report.dropped == tuple(entry["dropped"])
    assert report.seconds > 0


def test_block_frames_for() -> None:
    """Blocks are the largest multiple of 900 frames that fit the budget, never fewer than 900."""
    seg = inspect_legacy(legacy("v8-edge-cases.h5"))
    assert block_frames_for(seg, 1) == BLOCK_FRAMES

    # The slot count is bounded by identities plus legacy instances.
    slots = seg.identity + seg.instances
    frame_bytes = slots * seg.contour_axis * seg.point_axis * 2 * 4
    budget = 64 * 2**20
    frames = block_frames_for(seg, budget)
    assert frames % BLOCK_FRAMES == 0
    assert frames * frame_bytes <= budget < (frames + BLOCK_FRAMES) * frame_bytes

    points = inspect_legacy(legacy("v8-with-centers.h5"))
    frame_bytes = (points.identity + points.instances) * points.keypoints * 2 * 4
    frames = block_frames_for(points, budget)
    assert frames * frame_bytes <= budget < (frames + BLOCK_FRAMES) * frame_bytes


def test_memory_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory grows with the plan's per-frame arrays, not with the payload."""
    module = generator()
    monkeypatch.setattr(module, "CONTOURS", 4)
    monkeypatch.setattr(module, "CONTOUR_POINTS", 64)
    sources = {}
    for frames in (1800, 7200):
        sources[frames] = tmp_path / f"seg-{frames}.h5"
        module.write_legacy(
            sources[frames], frames=frames, instances=4, identities=4, version=(8, 0, 0), seg=True
        )

    # Measured from the plan of the larger file: what it keeps per frame.
    layout = inspect_legacy(sources[7200])
    with h5py.File(sources[7200], "r") as h5:
        plan = plan_placement(h5, layout)
    kept = (
        plan.slot_of,
        plan.tracklet_id,
        plan.contour_count,
        plan.contour_length,
        plan.contour_source,
        plan.group_offsets,
        plan.instance_offsets,
    )
    per_frame_allowance = sum(a.nbytes for a in kept if a is not None) / layout.frames
    del plan

    # A budget that gives both conversions 900-frame blocks.
    assert block_frames_for(layout, 1) == BLOCK_FRAMES
    # The deep check reads in its own blocks, capped at the constant
    # DEEP_CHECK_BLOCK_BYTES (64 MiB). That cap is a fixed ceiling, so total
    # memory stays bounded, but below it the deep check holds the whole of a
    # small file's contours. Shrunk to about 900 frames' worth, it is bounded
    # the same way as the converter, and the comparison measures the converter.
    deep_check = importlib.import_module("jabs.io.internal.pose_file.validate")
    frame_bytes = layout.instances * layout.contour_axis * layout.point_axis * 2 * 4
    monkeypatch.setattr(deep_check, "DEEP_CHECK_BLOCK_BYTES", BLOCK_FRAMES * frame_bytes)
    peaks = {}
    for frames, src in sources.items():
        tracemalloc.start()
        try:
            convert_legacy(src, tmp_path / f"out-{frames}.h5", block_bytes=1)
            peaks[frames] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    assert peaks[7200] <= 1.5 * peaks[1800] + 7200 * per_frame_allowance
