"""Generate the synthetic legacy pose corpus.

The ``.h5`` files beside this script are committed. Regenerate them by running
it from the repository root::

    uv run python packages/jabs-io/tests/data/legacy-pose/generate.py [OUT_DIR]

These are ``pose_est_v7`` and ``v8`` files, the input the legacy converter
reads, built for the cases the public samples in the repository root's
``tests/data`` lack: dynamic objects, v8 with and without
``instance_id_center``, every placement edge case, and files the converter must
refuse. Each file carries one case, and a refusal file carries no other
refusal, so a test that expects one fails for the reason it names.

Dataset names, dtypes and shapes follow the public samples and upstream's
``pose_est_v5.py``..``pose_est_v8.py`` readers: ``points`` is ``uint16``
``(y, x)``, ``seg_data`` is ``int16`` ``(x, y)`` padded with ``-1``, and
``id_mask`` true marks an instance with no identity. Every stored coordinate
pair has ``y != x``, so a missing or extra flip cannot pass unnoticed.

The files reproduce byte for byte: each ``write_legacy`` call seeds its own
generator, datasets are contiguous and uncompressed, and nothing time-dependent
is written.
"""

import json
import sys
from collections.abc import Callable, Collection
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

HERE = Path(__file__).parent

SEED = 29
KEYPOINTS = 12
EMBEDDING = 16
CONTOURS = 3
CONTOUR_POINTS = 8

# Keypoint rows live in y in [100, 400) and columns in x in [420, 720), so a
# stored (y, x) pair never has y == x and a flipped pair never looks unflipped.
Y_RANGE = (100, 400)
X_RANGE = (420, 720)

V7 = (7, 0)
# The version the Hub's production v8 files carry
# (docs/probes/adr-0029-segmentation-read-path-findings.md in the Hub).
V8 = (8, 0, 0)

# What ``write_legacy`` writes under ``poseest``, so ``omit`` can refuse a
# name it would not have written anyway.
_POSEEST = frozenset(
    {
        "points",
        "confidence",
        "id_mask",
        "instance_embed_id",
        "instance_track_id",
        "identity_embeds",
        "instance_id_center",
        "bbox",
        "seg_data",
        "seg_external_flag",
        "instance_seg_id",
        "longterm_seg_id",
    }
)

MODEL_METADATA = {
    "experiment_name": "synthetic_seg_and_pose",
    "spal_model_version": "2026-01-01_00000000_0/best_PCK@0.05",
    "recycle_instance_ids": False,
}


def write_legacy(
    path: Path,
    *,
    frames: int,
    instances: int,
    identities: int,
    version: tuple[int, ...],
    seg: bool,
    edits: Callable[[h5py.File], None] | None = None,
    omit: Collection[str] = (),
) -> None:
    """Write one clean legacy pose file, then let ``edits`` add its case.

    The clean file holds ``identities`` animals, one per instance slot, each a
    single gap-free tracklet with confidence in ``[0.4, 1)``. Slot ``s`` holds
    a fixed embed id from a shuffled ``1..identities``, so instances are not in
    slot ``embed - 1``, as in about half of real instances. Slots past
    ``identities`` are empty: ``id_mask`` true, confidence 0, embed and track
    id 0, and no contours. It trips none of the converter's refusals.

    By version: v5 and later add ``static_objects/corners``; v8 adds
    ``poseest/bbox`` with ``bboxes_generated`` true. ``seg`` adds
    ``seg_data``, ``seg_external_flag``, ``instance_seg_id`` (index + 1) and
    ``longterm_seg_id`` (the embed id), as the Hub's segmentation model writes
    them.

    Args:
        path: The file to write. It is overwritten.
        frames: Number of frames.
        instances: Number of instance slots; at least ``identities``.
        identities: Number of animals, and rows of ``instance_id_center``.
        version: The ``poseest`` ``version`` attribute.
        seg: Whether to write segmentation.
        edits: Called with the open file after the clean content is written,
            to add or change what makes this file's case.
        omit: ``poseest`` dataset names to leave out entirely, rather than
            write and delete, so the file looks as if its producer never had
            them (a real ``pose_v4`` v8 file has no ``instance_track_id``,
            ``identity_embeds`` or ``instance_id_center``).

    Raises:
        ValueError: If ``omit`` names a dataset this function does not write,
            so a misspelled name cannot silently omit nothing.
    """
    unknown = set(omit) - _POSEEST
    if unknown:
        raise ValueError(f"cannot omit datasets write_legacy does not write: {sorted(unknown)}")
    if instances < identities:
        raise ValueError("every identity needs an instance slot")

    rng = np.random.default_rng(SEED)
    shape = (frames, instances)
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, :identities] = True

    points = np.zeros((*shape, KEYPOINTS, 2), dtype=np.uint16)
    points[..., 0] = rng.integers(*Y_RANGE, (*shape, KEYPOINTS))
    points[..., 1] = rng.integers(*X_RANGE, (*shape, KEYPOINTS))
    confidence = rng.uniform(0.4, 1.0, (*shape, KEYPOINTS)).astype(np.float32)
    confidence[~occupied] = 0
    points[~occupied] = 0

    embed_of_slot = np.zeros(instances, dtype=np.uint32)
    embed_of_slot[:identities] = rng.permutation(identities) + 1
    embed = np.where(occupied, embed_of_slot, 0).astype(np.uint32)
    track = np.where(occupied, np.arange(1, instances + 1, dtype=np.uint32), 0).astype(np.uint32)
    embeds = rng.normal(0, 1, (*shape, EMBEDDING)).astype(np.float32)
    embeds[~occupied] = 0
    centers = rng.normal(0, 1, (identities, EMBEDDING))

    datasets: dict[str, npt.NDArray[np.generic]] = {
        "points": points,
        "confidence": confidence,
        "id_mask": ~occupied,
        "instance_embed_id": embed,
        "instance_track_id": track,
        "identity_embeds": embeds,
        "instance_id_center": centers,
    }
    if version[0] >= 8:
        datasets["bbox"] = _bbox(points, occupied)
    if seg:
        seg_data, external = _segmentation(rng, occupied)
        datasets["seg_data"] = seg_data
        datasets["seg_external_flag"] = external
        has_contours = (seg_data[..., 0, 0, 0] >= 0) & occupied
        index = np.arange(1, instances + 1, dtype=np.uint32)
        datasets["instance_seg_id"] = np.where(has_contours, index, 0).astype(np.uint32)
        datasets["longterm_seg_id"] = embed.copy()

    with h5py.File(path, "w") as h5:
        pose = h5.create_group("poseest")
        pose.attrs["version"] = np.array(version, dtype=np.uint16)
        pose.attrs["cm_per_pixel"] = np.float32(0.0794)
        for name, data in datasets.items():
            if name not in omit:
                pose.create_dataset(name, data=data)
        if "bbox" in pose:
            pose["bbox"].attrs["bboxes_generated"] = True
        if version[0] >= 5:
            corners = np.array([[52, 61], [748, 57], [741, 763], [57, 755]], dtype=np.uint16)
            h5.create_dataset("static_objects/corners", data=corners)
        if edits is not None:
            edits(h5)


def _bbox(
    points: npt.NDArray[np.uint16], occupied: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    """Return each instance's keypoint bounding box, ``(x, y)`` min and max corners."""
    xy = points[..., ::-1].astype(np.float32)
    bbox = np.stack([xy.min(axis=2), xy.max(axis=2)], axis=2)
    bbox[~occupied] = 0
    return bbox


def _segmentation(
    rng: np.random.Generator, occupied: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.bool_]]:
    """Return ``seg_data`` and ``seg_external_flag`` for the occupied instances.

    Each occupied instance has an external contour 0 of 4 to 8 points, and on
    about half its frames an internal contour 1 of 3 to 5; contour 2 is
    padding, so the padded extent is wider than any instance's use of it.
    """
    frames, instances = occupied.shape
    seg = np.full((frames, instances, CONTOURS, CONTOUR_POINTS, 2), -1, dtype=np.int16)
    external = np.zeros((frames, instances, CONTOURS), dtype=bool)
    lengths = rng.integers(4, CONTOUR_POINTS + 1, (frames, instances))
    inner = rng.integers(3, 6, (frames, instances)) * (rng.random((frames, instances)) < 0.5)
    for f, s in zip(*np.nonzero(occupied), strict=True):
        _contour(rng, seg[f, s, 0], lengths[f, s])
        external[f, s, 0] = True
        if inner[f, s]:
            _contour(rng, seg[f, s, 1], inner[f, s])
    return seg, external


def _contour(rng: np.random.Generator, out: npt.NDArray[np.int16], length: int) -> None:
    """Fill the first ``length`` rows of ``out`` with ``(x, y)`` points."""
    out[:length, 0] = rng.integers(*X_RANGE, length)
    out[:length, 1] = rng.integers(*Y_RANGE, length)


def _set_contours(h5: h5py.File, frame: int, slot: int, *, contour: int = 0) -> None:
    """Give one instance a single 5-point contour at ``contour``, with the rest empty."""
    seg = np.full((CONTOURS, CONTOUR_POINTS, 2), -1, dtype=np.int16)
    seg[contour, :5] = [[430, 110], [470, 112], [480, 150], [450, 170], [432, 140]]
    flag = np.zeros(CONTOURS, dtype=bool)
    flag[contour] = True
    h5["poseest/seg_data"][frame, slot] = seg
    h5["poseest/seg_external_flag"][frame, slot] = flag


def _empty(h5: h5py.File, frame: int, slot: int) -> None:
    """Make one slot empty in one frame, as an unfilled legacy slot is."""
    pose = h5["poseest"]
    pose["points"][frame, slot] = 0
    pose["confidence"][frame, slot] = 0
    pose["id_mask"][frame, slot] = True
    pose["instance_embed_id"][frame, slot] = 0
    pose["instance_track_id"][frame, slot] = 0
    pose["identity_embeds"][frame, slot] = 0
    pose["bbox"][frame, slot] = 0
    if "seg_data" in pose:
        pose["seg_data"][frame, slot] = -1
        pose["seg_external_flag"][frame, slot] = False
        pose["instance_seg_id"][frame, slot] = 0
        pose["longterm_seg_id"][frame, slot] = 0


def _slot_of_embed(h5: h5py.File, embed: int) -> int:
    """Return the slot that holds ``embed`` in the clean file's first frame."""
    return int(np.flatnonzero(h5["poseest/instance_embed_id"][0] == embed)[0])


def _dynamic_objects(h5: h5py.File) -> None:
    """Add v7 dynamic objects and the v5 static objects beside ``corners``."""
    # fecal_boli is stored 3-D, one keypoint per object, (y, x) by default
    # since it carries no axis_order (pose_est_v7.py:15-19, 92-96).
    boli = h5.create_group("dynamic_objects/fecal_boli")
    points = np.zeros((4, 3, 2), dtype=np.uint16)
    counts = np.array([0, 2, 3, 1], dtype=np.uint8)
    points[1, :2] = [[210, 610], [230, 655]]
    points[2, :3] = [[205, 612], [228, 650], [330, 470]]
    points[3, :1] = [[331, 468]]
    boli.create_dataset("points", data=points)
    boli.create_dataset("counts", data=counts)
    boli.create_dataset("sample_indices", data=np.array([0, 30, 60, 90], dtype=np.uint32))
    # lixit and food_hopper are stored (y, x) (pose_est_v5.py:11-14); lixit
    # here has two lixits of three keypoints (tip, left, right).
    lixit = np.array(
        [[[120, 380], [110, 372], [111, 389]], [[640, 702], [631, 694], [632, 711]]],
        dtype=np.float32,
    )
    h5.create_dataset("static_objects/lixit", data=lixit)
    hopper = np.array([[300, 20], [300, 90], [500, 90], [500, 20]], dtype=np.uint16)
    h5.create_dataset("static_objects/food_hopper", data=hopper)


def _edge_cases(h5: h5py.File) -> None:
    """Put each placement edge case in its own frame of a 3-animal, 4-slot file."""
    pose = h5["poseest"]

    # Embed id 0 on a valid instance, then one above the identity count.
    pose["instance_embed_id"][10, 0] = 0
    pose["longterm_seg_id"][10, 0] = 0
    pose["instance_embed_id"][20, 0] = 4
    pose["longterm_seg_id"][20, 0] = 4

    # The tail slot 3 is empty in the clean file. A masked instance with
    # keypoint confidence, and one with a contour but no confidence.
    rng = np.random.default_rng(SEED + 1)
    pose["points"][30, 3] = np.stack(
        [rng.integers(*Y_RANGE, KEYPOINTS), rng.integers(*X_RANGE, KEYPOINTS)], axis=-1
    )
    pose["confidence"][30, 3] = rng.uniform(0.4, 1.0, KEYPOINTS).astype(np.float32)
    _set_contours(h5, 40, 3)

    # Two valid instances, distinct embed ids, sharing one longterm_seg_id.
    pose["longterm_seg_id"][70, 1] = pose["longterm_seg_id"][70, 0]

    # The threshold: a stored float32 0.3 (0.30000001192, which counts as
    # valid), and in another frame a keypoint at 0 < confidence <= 0.3.
    pose["confidence"][80, 0, 0] = np.float32(0.3)
    pose["confidence"][85, 0, 1] = np.float32(0.15)

    # A contour gap: contour 1 present, contour 0 empty.
    _set_contours(h5, 90, 0, contour=1)

    # Slot 1's tracklet leaves for frames 100-102 and returns at 103.
    for frame in range(100, 103):
        _empty(h5, frame, 1)

    # A valid instance with contours and every keypoint at confidence 0. It is
    # in the last frame, so its tracklet ends rather than gaps.
    pose["confidence"][119, 2] = 0
    pose["points"][119, 2] = 0
    pose["bbox"][119, 2] = 0
    _set_contours(h5, 119, 2)


def _no_contours(h5: h5py.File) -> None:
    """Empty every contour, leaving the padded extents in place."""
    pose = h5["poseest"]
    pose["seg_data"][...] = -1
    pose["seg_external_flag"][...] = False
    pose["instance_seg_id"][...] = 0


def _duplicate_embed(h5: h5py.File) -> None:
    """Give a second valid instance in frame 100 embed id 2, keeping its tracklet."""
    other = 0 if _slot_of_embed(h5, 2) != 0 else 1
    h5["poseest/instance_embed_id"][100, other] = 2


def _tracklet_two_slots(h5: h5py.File) -> None:
    """Put slot 0's tracklet id on slot 1 too, in frame 50 only."""
    track = h5["poseest/instance_track_id"]
    track[50, 1] = track[50, 0]


def _model_metadata(h5: h5py.File) -> None:
    """Record the producing model as the Hub's v8 files do."""
    h5["poseest"].attrs["model_metadata_json"] = json.dumps(MODEL_METADATA, sort_keys=True)


def _untrusted_bbox(h5: h5py.File) -> None:
    """Drop ``bboxes_generated``, which the v8 reader needs to trust ``bbox``."""
    del h5["poseest/bbox"].attrs["bboxes_generated"]


def _mystery(h5: h5py.File) -> None:
    """Add a dataset no catalog names."""
    h5["poseest"].create_dataset("mystery", data=np.arange(5, dtype=np.int32))


def _bad_dynamic_object(h5: h5py.File) -> None:
    """Add a fecal_boli object without ``counts``, which the v7 reader skips."""
    boli = h5.create_group("dynamic_objects/fecal_boli")
    boli.create_dataset("points", data=np.zeros((2, 3, 2), dtype=np.uint16))
    boli.create_dataset("sample_indices", data=np.array([0, 30], dtype=np.uint32))


def main(out: Path = HERE) -> None:
    """Write every corpus file into ``out``.

    Args:
        out: The directory to write into: this script's own by default, or
            another, so a test can regenerate the corpus and compare bytes.
    """
    out.mkdir(parents=True, exist_ok=True)
    small = {"frames": 120, "instances": 3, "identities": 3}
    write_legacy(
        out / "v7-dynamic-objects.h5", **small, version=V7, seg=False, edits=_dynamic_objects
    )
    write_legacy(out / "v8-with-centers.h5", **small, version=V8, seg=False, edits=_model_metadata)
    write_legacy(
        out / "v8-without-centers.h5",
        **small,
        version=V8,
        seg=False,
        edits=_untrusted_bbox,
        omit={"instance_id_center", "instance_track_id", "identity_embeds"},
    )
    write_legacy(
        out / "v8-edge-cases.h5",
        frames=120,
        instances=4,
        identities=3,
        version=V8,
        seg=True,
        edits=_edge_cases,
    )
    write_legacy(out / "v8-no-contours.h5", **small, version=V8, seg=True, edits=_no_contours)
    write_legacy(
        out / "v8-duplicate-embed.h5", **small, version=V8, seg=False, edits=_duplicate_embed
    )
    write_legacy(
        out / "v8-tracklet-two-slots.h5", **small, version=V8, seg=False, edits=_tracklet_two_slots
    )
    write_legacy(out / "v8-unknown-dataset.h5", **small, version=V8, seg=False, edits=_mystery)
    write_legacy(
        out / "v7-bad-dynamic-object.h5", **small, version=V7, seg=False, edits=_bad_dynamic_object
    )

    for path in sorted(out.glob("*.h5")):
        print(f"{path.name:28s} {path.stat().st_size / 1024:6.1f} KiB")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
