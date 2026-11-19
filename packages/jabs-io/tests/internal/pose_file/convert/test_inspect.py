"""Inspect identifies a legacy pose file by its contents, or refuses it."""

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file.convert import LegacyConversionError, inspect_legacy

from .conftest import SAMPLES, generator, legacy, requires_samples

RETIRED = (
    "poseest/instance_seg_id",
    "poseest/longterm_seg_id",
    "poseest/instance_count",
    "poseest/id_mask",
    "poseest/instance_embed_id",
)


def _write(path: Path, edits: Callable[[h5py.File], None], **overrides: object) -> Path:
    """Write a small clean v8 file with ``edits`` applied, and return its path."""
    gen = generator()
    options: dict[str, object] = {**gen.SMALL, "version": gen.V8, "seg": False, **overrides}
    gen.write_legacy(path, edits=edits, **options)
    return path


@requires_samples
@pytest.mark.parametrize("version", [4, 5, 6])
def test_samples_are_recognized(version: int) -> None:
    """The public samples are recognized, with the version each file records."""
    layout = inspect_legacy(SAMPLES / f"sample_pose_est_v{version}.h5")

    assert layout.keypoints == 12
    assert layout.dynamic_objects == ()
    if version == 4:
        assert layout.identity == 5
        assert layout.instances == 6
        assert layout.frames == 9000
        assert layout.version == (4, 0)
        assert not layout.has_segmentation
        assert (layout.contour_axis, layout.point_axis) == (0, 0)
        assert layout.static_objects == ()
    if version == 5:
        assert layout.version == (3, 0)
        assert layout.static_objects == ("corners",)
    if version == 6:
        assert layout.has_segmentation
        assert layout.contour_axis == 4
        assert layout.point_axis == 319
        assert layout.has_tracklets
        assert layout.has_embeddings
        assert not layout.bbox_generated
    assert all(isinstance(part, int) for part in layout.version)


def test_identity_count() -> None:
    """Centers give the identity count; without them, the largest valid embed id does."""
    gen = generator()
    with_centers = inspect_legacy(legacy("v8-with-centers.h5"))
    assert with_centers.identity == 3
    assert with_centers.version == (8, 0, 0)

    without = inspect_legacy(legacy("v8-without-centers.h5"))
    with h5py.File(legacy("v8-without-centers.h5"), "r") as h5:
        valid = ~h5["poseest/id_mask"][()].astype(bool)
        largest = int(h5["poseest/instance_embed_id"][()][valid].max())
    assert without.identity == largest == gen.SMALL["identities"]


def test_identity_ignores_masked_embed_ids(tmp_path: Path) -> None:
    """A masked instance's embed id does not count toward the identity count."""

    def mask_high_embed(h5: h5py.File) -> None:
        """Give a masked instance an embed id above every valid one."""
        del h5["poseest/instance_id_center"]
        h5["poseest/instance_embed_id"][0, 3] = 9

    path = _write(tmp_path / "masked.h5", mask_high_embed, instances=4)
    assert inspect_legacy(path).identity == 3


def test_identity_may_be_zero(tmp_path: Path) -> None:
    """A file with no centers and no valid embed id has identity 0, not a refusal."""

    def mask_everything(h5: h5py.File) -> None:
        """Remove the centers and mask every instance."""
        del h5["poseest/instance_id_center"]
        h5["poseest/id_mask"][...] = True

    layout = inspect_legacy(_write(tmp_path / "anonymous.h5", mask_everything))
    assert layout.identity == 0


def test_tracklet_and_embedding_flags() -> None:
    """A ``pose_v4`` v8 file carries neither tracklets nor embeddings."""
    layout = inspect_legacy(legacy("v8-without-centers.h5"))
    assert not layout.has_tracklets
    assert not layout.has_embeddings
    with_centers = inspect_legacy(legacy("v8-with-centers.h5"))
    assert with_centers.has_tracklets
    assert with_centers.has_embeddings


def test_v7_objects() -> None:
    """Static and dynamic objects are listed by name."""
    layout = inspect_legacy(legacy("v7-dynamic-objects.h5"))
    assert layout.static_objects == ("corners", "food_hopper", "lixit")
    assert layout.dynamic_objects == ("fecal_boli",)
    assert layout.version == (7, 0)


def test_segmentation_extents() -> None:
    """The padded contour and point extents come from ``seg_data``'s shape."""
    gen = generator()
    layout = inspect_legacy(legacy("v8-edge-cases.h5"))
    assert layout.has_segmentation
    assert (layout.contour_axis, layout.point_axis) == (gen.CONTOURS, gen.CONTOUR_POINTS)
    assert (layout.frames, layout.instances) == (gen.EDGE_FRAMES, gen.EDGE_INSTANCES)
    assert layout.identity == gen.EDGE_IDENTITIES
    assert layout.bbox_generated


def _delete_id_mask(h5: h5py.File) -> None:
    """Delete ``id_mask``, as a file below v4 lacks it."""
    del h5["poseest/id_mask"]


def _bad_axis_order(h5: h5py.File) -> None:
    """Add a dynamic object whose ``axis_order`` is neither ``xy`` nor ``yx``."""
    obj = h5.create_group("dynamic_objects/fecal_boli")
    obj.create_dataset("points", data=np.zeros((2, 3, 2), dtype=np.uint16))
    obj.create_dataset("counts", data=np.zeros(2, dtype=np.uint8))
    obj.create_dataset("sample_indices", data=np.array([0, 30], dtype=np.uint32))
    obj["points"].attrs["axis_order"] = "zz"


def _bad_points_ndim(h5: h5py.File) -> None:
    """Add a dynamic object whose ``points`` is 2-D."""
    obj = h5.create_group("dynamic_objects/fecal_boli")
    obj.create_dataset("points", data=np.zeros((2, 2), dtype=np.uint16))
    obj.create_dataset("counts", data=np.zeros(2, dtype=np.uint8))
    obj.create_dataset("sample_indices", data=np.array([0, 30], dtype=np.uint32))


def _dynamic_object(
    points: npt.NDArray[np.generic],
    counts: npt.NDArray[np.generic],
    sample_indices: npt.NDArray[np.generic],
) -> Callable[[h5py.File], None]:
    """Return edits that add a ``fecal_boli`` dynamic object with these datasets."""

    def edits(h5: h5py.File) -> None:
        """Add the dynamic object."""
        obj = h5.create_group("dynamic_objects/fecal_boli")
        obj.create_dataset("points", data=points)
        obj.create_dataset("counts", data=counts)
        obj.create_dataset("sample_indices", data=sample_indices)

    return edits


_POINTS = np.zeros((2, 3, 2), dtype=np.uint16)
_COUNTS = np.array([1, 2], dtype=np.uint8)
_INDICES = np.array([0, 30], dtype=np.uint32)


def _eleven_keypoints(h5: h5py.File) -> None:
    """Replace ``points`` and ``confidence`` with 11-keypoint versions."""
    pose = h5["poseest"]
    points = pose["points"][:, :, :11]
    confidence = pose["confidence"][:, :, :11]
    del pose["points"], pose["confidence"]
    pose.create_dataset("points", data=points)
    pose.create_dataset("confidence", data=confidence)


def _three_coordinates(h5: h5py.File) -> None:
    """Give ``points`` three coordinates per keypoint rather than two."""
    points = h5["poseest/points"][()]
    _replace(h5, "points", np.concatenate([points, points[..., :1]], axis=-1))


def _unknown_root(h5: h5py.File) -> None:
    """Add a top-level group no catalog names."""
    h5.create_group("extras")


def _replace(h5: h5py.File, name: str, data: npt.NDArray[np.generic]) -> None:
    """Replace one ``poseest`` dataset with ``data``."""
    del h5["poseest"][name]
    h5["poseest"].create_dataset(name, data=data)


def _narrow_id_mask(h5: h5py.File) -> None:
    """Give ``id_mask`` one fewer instance column than ``points``."""
    _replace(h5, "id_mask", h5["poseest/id_mask"][:, :-1])


def _short_confidence(h5: h5py.File) -> None:
    """Give ``confidence`` one fewer frame than ``points``."""
    _replace(h5, "confidence", h5["poseest/confidence"][:-1])


def _short_external_flag(h5: h5py.File) -> None:
    """Give ``seg_external_flag`` one fewer contour than ``seg_data``."""
    _replace(h5, "seg_external_flag", h5["poseest/seg_external_flag"][:, :, :-1])


def _flat_bbox(h5: h5py.File) -> None:
    """Store ``bbox`` as four numbers per instance rather than two corners."""
    bbox = h5["poseest/bbox"][()]
    _replace(h5, "bbox", bbox.reshape(*bbox.shape[:2], 4))


def _narrow_embeddings(h5: h5py.File) -> None:
    """Make ``identity_embeds`` narrower than ``instance_id_center``."""
    _replace(h5, "identity_embeds", h5["poseest/identity_embeds"][..., :-1])


def _text_version(h5: h5py.File) -> None:
    """Record the version as text."""
    h5["poseest"].attrs["version"] = "8.0"


def _short_external_ids(h5: h5py.File) -> None:
    """Name one fewer identity than the file has."""
    h5["poseest"].create_dataset("external_identity_mapping", data=np.array([b"a", b"b"]))


def _two_dimensional_external_ids(h5: h5py.File) -> None:
    """Store the identity names as a column rather than a list."""
    h5["poseest"].create_dataset("external_identity_mapping", data=np.array([[b"a"], [b"b"]]))


def _short_longterm_seg_id(h5: h5py.File) -> None:
    """Give ``longterm_seg_id`` one fewer frame than ``points``."""
    _replace(h5, "longterm_seg_id", h5["poseest/longterm_seg_id"][:-1])


def _wide_instance_seg_id(h5: h5py.File) -> None:
    """Give ``instance_seg_id`` one more instance column than ``points``."""
    seg_id = h5["poseest/instance_seg_id"][()]
    _replace(h5, "instance_seg_id", np.concatenate([seg_id, seg_id[:, :1]], axis=1))


def _flat_instance_embedding(h5: h5py.File) -> None:
    """Add an ``instance_embedding`` with no per-keypoint axis."""
    h5["poseest"].create_dataset("instance_embedding", data=np.zeros((120, 3), dtype=np.float32))


def _per_instance_count(h5: h5py.File) -> None:
    """Add an ``instance_count`` with a column per instance rather than one per frame."""
    h5["poseest"].create_dataset("instance_count", data=np.zeros((120, 3), dtype=np.uint8))


def _unknown_static_object(h5: h5py.File) -> None:
    """Add a static object no catalog names."""
    h5.create_dataset("static_objects/water_bottle", data=np.zeros((2, 2), dtype=np.uint16))


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("v8-unknown-dataset.h5", "poseest/mystery"),
        ("v7-bad-dynamic-object.h5", "counts"),
        (_delete_id_mask, "id_mask"),
        (_bad_axis_order, "axis_order"),
        (_eleven_keypoints, "12"),
        (
            _three_coordinates,
            r"poseest/points has shape \(120, 3, 12, 3\); expected \(frame, instance, 12, 2\)",
        ),
        (_bad_points_ndim, "ndim"),
        (
            _dynamic_object(_POINTS, _COUNTS, _INDICES[:, None]),
            r"dynamic_objects/fecal_boli/sample_indices has shape \(2, 1\); expected 1 axis",
        ),
        (
            _dynamic_object(_POINTS, _COUNTS[:, None], _INDICES),
            r"dynamic_objects/fecal_boli/counts has shape \(2, 1\); expected 1 axis",
        ),
        (
            _dynamic_object(_POINTS, np.array([1, 2, 0], dtype=np.uint8), _INDICES),
            r"dynamic_objects/fecal_boli has 2 sample_indices, 3 counts and 2 point samples",
        ),
        (
            _dynamic_object(np.zeros((3, 3, 2), dtype=np.uint16), _COUNTS, _INDICES),
            r"dynamic_objects/fecal_boli has 2 sample_indices, 2 counts and 3 point samples",
        ),
        (
            _dynamic_object(np.zeros((2, 3, 3), dtype=np.uint16), _COUNTS, _INDICES),
            r"dynamic_objects/fecal_boli/points has shape \(2, 3, 3\); expected 2 coordinates",
        ),
        (
            _dynamic_object(_POINTS, np.array([1, 4], dtype=np.uint8), _INDICES),
            r"dynamic_objects/fecal_boli/counts holds 4, more than the 3 objects",
        ),
        (
            _dynamic_object(_POINTS, _COUNTS, np.array([30, 0], dtype=np.uint32)),
            r"dynamic_objects/fecal_boli/sample_indices decreases at sample 1",
        ),
        (
            _dynamic_object(_POINTS, _COUNTS, np.array([0, 120], dtype=np.uint32)),
            r"dynamic_objects/fecal_boli/sample_indices holds frame 120, past the last frame, 119",
        ),
        (_unknown_root, "extras"),
        (_unknown_static_object, "static_objects/water_bottle"),
        (
            _narrow_id_mask,
            r"poseest/id_mask has shape \(120, 2\).*poseest/points \(120, 3, 12, 2\)",
        ),
        (_short_confidence, r"poseest/confidence has shape \(119, 3, 12\)"),
        (_short_external_flag, r"poseest/seg_external_flag has shape \(120, 3, 2\).*seg_data"),
        (_flat_bbox, r"poseest/bbox has shape \(120, 3, 4\)"),
        (_narrow_embeddings, r"poseest/identity_embeds has shape .*instance_id_center"),
        (_text_version, "version"),
        (
            _short_external_ids,
            r"poseest/external_identity_mapping has 2 names.*3 identities",
        ),
        (_two_dimensional_external_ids, r"poseest/external_identity_mapping has shape \(2, 1\)"),
        (_short_longterm_seg_id, r"poseest/longterm_seg_id has shape \(119, 3\)"),
        (_wide_instance_seg_id, r"poseest/instance_seg_id has shape \(120, 4\)"),
        (_flat_instance_embedding, r"poseest/instance_embedding has shape \(120, 3\)"),
        (_per_instance_count, r"poseest/instance_count has shape \(120, 3\)"),
    ],
    ids=[
        "unknown-dataset",
        "bad-dynamic-object",
        "no-id-mask",
        "axis-order",
        "eleven-keypoints",
        "three-coordinates",
        "points-ndim",
        "dynamic-indices-2d",
        "dynamic-counts-2d",
        "dynamic-counts-length",
        "dynamic-points-length",
        "dynamic-points-coordinates",
        "dynamic-counts-above-objects",
        "dynamic-indices-unsorted",
        "dynamic-indices-past-frames",
        "unknown-root",
        "unknown-static-object",
        "id-mask-shape",
        "confidence-shape",
        "external-flag-shape",
        "bbox-shape",
        "embedding-width",
        "text-version",
        "external-ids-length",
        "external-ids-shape",
        "longterm-seg-id-shape",
        "instance-seg-id-shape",
        "instance-embedding-shape",
        "instance-count-shape",
    ],
)
def test_refusals(tmp_path: Path, source: str | Callable[[h5py.File], None], match: str) -> None:
    """Anything outside the catalog, or malformed within it, is refused by name.

    ``source`` is a corpus file's name, or edits to a clean small v8 file.
    """
    if isinstance(source, str):
        path = legacy(source)
    else:
        # Segmentation is on so its shape checks have something to check.
        path = _write(tmp_path / "x.h5", source, seg=True)
    with pytest.raises(LegacyConversionError, match=match):
        inspect_legacy(path)


def test_dropped_lists_retired_datasets() -> None:
    """Retired datasets are listed when present, and an untrusted bbox with them."""
    assert inspect_legacy(legacy("v8-without-centers.h5")).dropped == (
        "poseest/id_mask",
        "poseest/instance_embed_id",
        "poseest/bbox",
    )
    assert "poseest/bbox" not in inspect_legacy(legacy("v8-with-centers.h5")).dropped


@requires_samples
def test_dropped_on_samples() -> None:
    """v6 drops its segmentation ids and identity bookkeeping; v4 its old embedding too."""
    # The v6 sample has no instance_embedding, so it is not listed.
    assert inspect_legacy(SAMPLES / "sample_pose_est_v6.h5").dropped == RETIRED
    v4 = inspect_legacy(SAMPLES / "sample_pose_est_v4.h5").dropped
    assert v4 == (
        "poseest/instance_count",
        "poseest/id_mask",
        "poseest/instance_embed_id",
        "poseest/instance_embedding",
    )


def test_empty_lixit_is_dropped(tmp_path: Path) -> None:
    """An empty lixit is not a static object, and is listed as dropped."""

    def empty_lixit(h5: h5py.File) -> None:
        """Add a lixit with no objects in it."""
        h5.create_dataset("static_objects/lixit", data=np.zeros((0, 2), dtype=np.float32))

    layout = inspect_legacy(_write(tmp_path / "lixit.h5", empty_lixit))
    assert layout.static_objects == ("corners",)
    assert layout.dropped[-1] == "static_objects/lixit"


def test_missing_version_is_empty(tmp_path: Path) -> None:
    """A file without a ``version`` attribute is identified by its contents."""

    def no_version(h5: h5py.File) -> None:
        """Remove the version attribute."""
        del h5["poseest"].attrs["version"]

    assert inspect_legacy(_write(tmp_path / "unversioned.h5", no_version)).version == ()
