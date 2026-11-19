"""The synthetic legacy corpus carries the cases the converter is tested on.

Each corpus file exists for one case the public samples lack. These tests pin
that each file still carries its case, and only its case where the case is a
refusal, so a later converter test that expects a refusal fails for the reason
it names.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from .conftest import LEGACY, SAMPLES, generator, legacy

CORPUS = {
    "v7-dynamic-objects.h5",
    "v8-with-centers.h5",
    "v8-without-centers.h5",
    "v8-edge-cases.h5",
    "v8-no-contours.h5",
    "v8-duplicate-embed.h5",
    "v8-tracklet-two-slots.h5",
    "v8-unknown-dataset.h5",
    "v7-bad-dynamic-object.h5",
}

# The datasets a real v8 file from the `pose_v4` model has, and nothing else
# (docs/probes/adr-0029-pose-census-findings.md in the Hub).
POSE_V4_DATASETS = {"bbox", "confidence", "id_mask", "instance_embed_id", "points"}


def _poseest(h5: h5py.File, name: str) -> npt.NDArray[np.generic]:
    """Read one ``poseest`` dataset whole."""
    return h5["poseest"][name][()]


def _valid(h5: h5py.File) -> npt.NDArray[np.bool_]:
    """Return which instances are valid: ``id_mask`` 0 marks an identified instance."""
    return ~_poseest(h5, "id_mask")


def _placed(h5: h5py.File) -> npt.NDArray[np.bool_]:
    """Return which valid instances have any keypoint confidence above 0."""
    return _valid(h5) & (_poseest(h5, "confidence") > 0).any(axis=-1)


def _has_contour(h5: h5py.File) -> npt.NDArray[np.bool_]:
    """Return, per ``(frame, instance, contour)``, whether the contour holds any point."""
    return (_poseest(h5, "seg_data") >= 0).all(axis=-1).any(axis=-1)


def _duplicate_embed_frames(h5: h5py.File) -> list[int]:
    """List frames where two valid instances share an embed id in ``1..identities``."""
    embed = _poseest(h5, "instance_embed_id")
    valid = _valid(h5)
    frames = []
    for f in range(embed.shape[0]):
        ids = embed[f][valid[f] & (embed[f] > 0)]
        if len(ids) != len(set(ids.tolist())):
            frames.append(f)
    return frames


def _two_slot_tracklet_frames(h5: h5py.File) -> list[int]:
    """List frames where one tracklet id sits on two placed instances."""
    track = _poseest(h5, "instance_track_id")
    placed = _placed(h5)
    frames = []
    for f in range(track.shape[0]):
        ids = track[f][placed[f]]
        if len(ids) != len(set(ids.tolist())):
            frames.append(f)
    return frames


def _tracklet_gaps(h5: h5py.File) -> dict[int, int]:
    """Count, per tracklet id, the gaps between the runs of frames where it is placed."""
    track = _poseest(h5, "instance_track_id")
    placed = _placed(h5)
    gaps = {}
    for tracklet in np.unique(track[placed]).tolist():
        frames = np.flatnonzero((placed & (track == tracklet)).any(axis=1))
        gaps[tracklet] = int((np.diff(frames) > 1).sum())
    return gaps


def _assert_clean(h5: h5py.File) -> None:
    """Assert the file trips none of the converter's refusals."""
    assert _duplicate_embed_frames(h5) == []
    if "instance_track_id" in h5["poseest"]:
        assert _two_slot_tracklet_frames(h5) == []
    assert "mystery" not in h5["poseest"]


def _nan_with_confidence(h5: h5py.File) -> npt.NDArray[np.bool_]:
    """Return, per keypoint, whether its coordinate is NaN although its confidence is above 0."""
    nan = np.isnan(_poseest(h5, "points")).any(axis=-1)
    return nan & (_poseest(h5, "confidence") > 0)


def _check_points(h5: h5py.File) -> None:
    """``points`` is uint16 through v7; float32 in v8, NaN at every zero-confidence keypoint."""
    points = _poseest(h5, "points")
    if h5["poseest"].attrs["version"][0] < 8:
        assert points.dtype == np.uint16
        return
    assert points.dtype == np.float32
    nan = np.isnan(points)
    assert (nan[..., 0] == nan[..., 1]).all()
    assert nan[_poseest(h5, "confidence") == 0].all()


def _check_bbox(h5: h5py.File) -> None:
    """Every instance with confidence has the bbox of its finite keypoints, ``(x, y)``."""
    points = _poseest(h5, "points").astype(np.float64)[..., ::-1]
    bbox = _poseest(h5, "bbox")
    for f, i in _where((_poseest(h5, "confidence") > 0).any(axis=-1)):
        xy = points[f, i][np.isfinite(points[f, i]).all(axis=-1)]
        assert (bbox[f, i] == np.stack([xy.min(axis=0), xy.max(axis=0)])).all(), (f, i)


def _check_v7_dynamic_objects(h5: h5py.File) -> None:
    """A 3-D fecal_boli object with no ``axis_order``, and every static object."""
    assert h5["poseest"].attrs["version"][0] == 7
    boli = h5["dynamic_objects/fecal_boli"]
    assert set(boli) == {"points", "counts", "sample_indices"}
    points = boli["points"]
    assert points.ndim == 3 and points.shape[-1] == 2
    assert "axis_order" not in points.attrs
    counts = boli["counts"][()]
    indices = boli["sample_indices"][()]
    assert len(counts) == len(indices) == points.shape[0]
    assert (np.diff(indices.astype(np.int64)) > 0).all()
    assert indices.max() < h5["poseest/points"].shape[0]
    assert counts.max() <= points.shape[1] and 0 in counts
    assert set(h5["static_objects"]) == {"corners", "lixit", "food_hopper"}
    assert h5["static_objects/corners"].shape == (4, 2)
    assert h5["static_objects/lixit"].shape == (2, 3, 2)
    assert h5["static_objects/food_hopper"].shape == (4, 2)
    _assert_clean(h5)


def _check_v8_with_centers(h5: h5py.File) -> None:
    """Identity centers, a trusted bbox and the model's metadata."""
    pose = h5["poseest"]
    assert pose.attrs["version"][0] == 8
    assert pose["instance_id_center"].shape == (3, 16)
    assert pose["bbox"].attrs["bboxes_generated"]
    metadata = json.loads(pose.attrs["model_metadata_json"])
    assert {"experiment_name", "spal_model_version"} <= set(metadata)
    _assert_clean(h5)


def _check_v8_without_centers(h5: h5py.File) -> None:
    """Exactly a real ``pose_v4`` v8 file's datasets, and an untrusted bbox."""
    pose = h5["poseest"]
    assert pose.attrs["version"][0] == 8
    assert set(pose) == POSE_V4_DATASETS
    assert "bboxes_generated" not in pose["bbox"].attrs
    _assert_clean(h5)


class _Case(Protocol):
    """The shape of ``generate.Case``, which is loaded by path."""

    frames: tuple[int, ...]
    instances: tuple[int, ...]
    keypoint: int | None


def _where(mask: npt.NDArray[np.bool_]) -> set[tuple[int, ...]]:
    """Return the indexes where ``mask`` holds."""
    return {tuple(int(v) for v in index) for index in np.argwhere(mask)}


def _at(case: _Case) -> set[tuple[int, ...]]:
    """Return a ``generate.Case``'s locations, as ``_where`` reports them."""
    return {
        (f, i) if case.keypoint is None else (f, i, case.keypoint)
        for f in case.frames
        for i in case.instances
    }


def _shared_longterm(h5: h5py.File) -> npt.NDArray[np.bool_]:
    """Return which valid instances share their frame's ``longterm_seg_id`` with another."""
    longterm = _poseest(h5, "longterm_seg_id")
    valid = _valid(h5)
    shared = np.zeros_like(valid)
    for f in range(longterm.shape[0]):
        values, counts = np.unique(longterm[f][valid[f]], return_counts=True)
        shared[f] = valid[f] & np.isin(longterm[f], values[counts > 1])
    return shared


def _check_v8_edge_cases(h5: h5py.File) -> None:
    """Each case at exactly its ``EDGE`` location, and no refusal."""
    g = generator()
    edge = g.EDGE
    embed = _poseest(h5, "instance_embed_id")
    valid = _valid(h5)
    masked = ~valid
    confidence = _poseest(h5, "confidence")
    unconfident = (confidence == 0).all(axis=-1)
    contour = _has_contour(h5)
    assert _poseest(h5, "points").shape[:2] == (g.EDGE_FRAMES, g.EDGE_INSTANCES)
    assert h5["poseest/instance_id_center"].shape[0] == g.EDGE_IDENTITIES
    assert h5["poseest/seg_data"].dtype == np.int16
    assert (_poseest(h5, "seg_data") >= -1).all()
    assert h5["poseest/seg_external_flag"].dtype == np.bool_

    assert _where(valid & (embed == 0)) == _at(edge["embed_zero"])
    assert _where(valid & (embed > g.EDGE_IDENTITIES)) == _at(edge["embed_above_identities"])
    assert {int(embed[loc]) for loc in _at(edge["embed_above_identities"])} == {g.EDGE_EMBED_ABOVE}
    assert _where(masked & ~unconfident) == _at(edge["masked_confident"])
    assert _where(masked & contour.any(axis=-1) & unconfident) == _at(edge["masked_contour"])
    assert _where(valid & contour.any(axis=-1) & unconfident) == _at(
        edge["zero_confidence_contours"]
    )
    exact = confidence == np.float32(0.3)
    low = (confidence > 0) & (confidence.astype(np.float64) <= 0.3) & ~exact
    assert _where(exact & valid[..., None]) == _at(edge["threshold_exact"])
    assert _where(low & valid[..., None]) == _at(edge["low_confidence"])
    assert {float(confidence[loc]) for loc in _at(edge["low_confidence"])} == {
        float(g.EDGE_LOW_CONFIDENCE)
    }
    assert _where(valid & ~contour[..., 0] & contour[..., 1]) == _at(edge["contour_gap"])
    assert _where(_nan_with_confidence(h5)) == _at(edge["nan_with_confidence"])

    # Two valid instances share a longterm_seg_id, keeping their clean,
    # distinct embed ids; the shared value is the first one's embed id.
    shared = edge["shared_longterm"]
    assert _where(_shared_longterm(h5)) == _at(shared)
    (frame,) = shared.frames
    ids = [g.clean_embed_id(i, g.EDGE_IDENTITIES) for i in shared.instances]
    assert [int(embed[frame, i]) for i in shared.instances] == ids
    assert len(set(ids)) == len(ids)
    longterm = _poseest(h5, "longterm_seg_id")
    assert {int(longterm[frame, i]) for i in shared.instances} == {ids[0]}

    # Every other valid instance holds its clean embed id, so a placement test
    # can read where it lands from clean_embed_id.
    odd = _at(edge["embed_zero"]) | _at(edge["embed_above_identities"])
    for f, i in _where(valid) - odd:
        assert embed[f, i] == g.clean_embed_id(i, g.EDGE_IDENTITIES), (f, i)

    # Exactly one tracklet leaves its instance for the listed frames and
    # comes back to the same instance.
    gap = edge["tracklet_gap"]
    (instance,) = gap.instances
    tracklet = g.clean_track_id(instance)
    placed = _placed(h5)
    track = _poseest(h5, "instance_track_id")
    assert _where(~placed[:, instance]) == {(f,) for f in gap.frames}
    assert (track[placed[:, instance], instance] == tracklet).all()
    assert {t: n for t, n in _tracklet_gaps(h5).items() if n} == {tracklet: 1}

    _assert_clean(h5)


def _check_v8_no_contours(h5: h5py.File) -> None:
    """Segmentation with padded extents and no contour point anywhere."""
    seg = h5["poseest/seg_data"]
    assert seg.dtype == np.int16 and seg.size > 0
    assert (seg[()] == -1).all()
    assert not _poseest(h5, "seg_external_flag").any()
    assert not _poseest(h5, "instance_seg_id").any()
    assert not _poseest(h5, "longterm_seg_id").any()
    _assert_clean(h5)


def _check_v8_duplicate_embed(h5: h5py.File) -> None:
    """``DUPLICATE_EMBED_ID`` on exactly the ``DUPLICATE_EMBED`` instances, and no other refusal."""
    g = generator()
    case = g.DUPLICATE_EMBED
    assert _duplicate_embed_frames(h5) == list(case.frames)
    embed = _poseest(h5, "instance_embed_id")
    frames = set(case.frames)
    holders = _where(_valid(h5) & (embed == g.DUPLICATE_EMBED_ID))
    assert {loc for loc in holders if loc[0] in frames} == _at(case)
    assert len(case.instances) == 2
    assert _two_slot_tracklet_frames(h5) == []


def _check_v8_tracklet_two_slots(h5: h5py.File) -> None:
    """The first instance's tracklet id on both ``TWO_SLOT_TRACKLET`` instances, and no other refusal."""
    g = generator()
    case = g.TWO_SLOT_TRACKLET
    assert _two_slot_tracklet_frames(h5) == list(case.frames)
    track = _poseest(h5, "instance_track_id")
    frames = set(case.frames)
    holders = _where(_placed(h5) & (track == g.clean_track_id(case.instances[0])))
    assert {loc for loc in holders if loc[0] in frames} == _at(case)
    assert _duplicate_embed_frames(h5) == []


def _check_v8_unknown_dataset(h5: h5py.File) -> None:
    """A ``poseest`` dataset no catalog names, and no other refusal."""
    assert isinstance(h5["poseest/mystery"], h5py.Dataset)
    assert _duplicate_embed_frames(h5) == []
    assert _two_slot_tracklet_frames(h5) == []


def _check_v7_bad_dynamic_object(h5: h5py.File) -> None:
    """A fecal_boli object without ``counts``, and no other refusal."""
    assert h5["poseest"].attrs["version"][0] == 7
    assert set(h5["dynamic_objects/fecal_boli"]) == {"points", "sample_indices"}
    _assert_clean(h5)


CASES: dict[str, Callable[[h5py.File], None]] = {
    "v7-dynamic-objects.h5": _check_v7_dynamic_objects,
    "v8-with-centers.h5": _check_v8_with_centers,
    "v8-without-centers.h5": _check_v8_without_centers,
    "v8-edge-cases.h5": _check_v8_edge_cases,
    "v8-no-contours.h5": _check_v8_no_contours,
    "v8-duplicate-embed.h5": _check_v8_duplicate_embed,
    "v8-tracklet-two-slots.h5": _check_v8_tracklet_two_slots,
    "v8-unknown-dataset.h5": _check_v8_unknown_dataset,
    "v7-bad-dynamic-object.h5": _check_v7_bad_dynamic_object,
}


def test_corpus_is_complete() -> None:
    """The corpus holds exactly the files the converter's tests name."""
    assert {p.name for p in LEGACY.glob("*.h5")} == CORPUS


def test_samples_point_at_the_repository_root() -> None:
    """``SAMPLES`` is the repository root's ``tests/data``.

    A wrong depth would not fail anything: it would silently skip every test
    that uses the public samples.
    """
    assert (SAMPLES.parent.parent / "packages" / "jabs-io").is_dir()


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_file_carries_its_case(name: str) -> None:
    """Each file has the v4+ datasets the samples have, and its own case.

    Only ``v8-edge-cases.h5`` has a NaN coordinate on a keypoint with
    confidence; everywhere else NaN means confidence 0.
    """
    with h5py.File(legacy(name), "r") as h5:
        pose = h5["poseest"]
        assert {"version", "cm_per_pixel"} <= set(pose.attrs)
        assert pose["points"].shape[2:] == (12, 2)
        _check_points(h5)
        if "bbox" in pose:
            _check_bbox(h5)
        if name != "v8-edge-cases.h5":
            assert not _nan_with_confidence(h5).any()
        assert pose["confidence"].dtype == np.float32
        assert pose["id_mask"].dtype == np.bool_
        CASES[name](h5)


@pytest.mark.parametrize("name", sorted(CASES))
def test_coordinates_are_asymmetric(name: str) -> None:
    """No stored keypoint has ``y == x``, so a missing flip cannot pass a test."""
    with h5py.File(legacy(name), "r") as h5:
        points = _poseest(h5, "points")
        placed = _poseest(h5, "confidence") > 0
        assert (points[placed][:, 0] != points[placed][:, 1]).all()


def test_write_legacy_refuses_an_unknown_omission(tmp_path: Path) -> None:
    """A misspelled ``omit`` name raises rather than omitting nothing."""
    with pytest.raises(ValueError, match="no_such_dataset"):
        generator().write_legacy(
            tmp_path / "x.h5",
            frames=4,
            instances=2,
            identities=2,
            version=(8, 0, 0),
            seg=False,
            omit={"no_such_dataset"},
        )


def test_corpus_regenerates_byte_identically(tmp_path: Path) -> None:
    """Running the generator again reproduces every committed file byte for byte."""
    generator().main(tmp_path)
    assert {p.name for p in tmp_path.glob("*.h5")} == CORPUS
    for name in sorted(CORPUS):
        assert (tmp_path / name).read_bytes() == legacy(name).read_bytes(), name
