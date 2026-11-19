"""Identify a legacy pose file by its contents, before anything is converted.

A legacy file's ``version`` attribute does not say what it holds (the public
v5 sample records ``3.0``), so the converter reads what is there instead and
refuses anything outside the catalog of ``pose_est_v4``..``v8`` paths, naming
it. A path it does not know is one it would silently lose.

Inspection reads shapes and attributes only, except ``instance_embed_id`` and
``id_mask``, which it reads in frame blocks when the identity count has to
come from the largest valid embed id.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import h5py
import numpy as np

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.reader import attr_text

KEYPOINTS = 12

# The least common multiple of the writer's 300-frame chunk and legacy
# seg_data's 450-frame chunk.
BLOCK_FRAMES = 900

_POSEEST = frozenset(
    {
        "points",
        "confidence",
        "id_mask",
        "instance_embed_id",
        "instance_track_id",
        "identity_embeds",
        "instance_id_center",
        "instance_embedding",
        "instance_count",
        "instance_seg_id",
        "longterm_seg_id",
        "seg_data",
        "seg_external_flag",
        "bbox",
    }
)
# What a v4 file has and older ones lack; without them nothing can be placed.
_REQUIRED = ("points", "confidence", "id_mask", "instance_embed_id")
_STATIC_OBJECTS = frozenset({"corners", "lixit", "food_hopper"})
_DYNAMIC_DATASETS = ("points", "counts", "sample_indices")
_AXIS_ORDERS = frozenset({"xy", "yx"})
_ROOTS = frozenset({"poseest", "static_objects", "dynamic_objects"})

# Legacy datasets the format retires: segmentation is indexed by slot, so the
# seg ids go; occupancy and identity are stated directly, so the counts, masks
# and embed ids go; instance_embedding is the old per-keypoint association
# embedding, which nothing reads.
_RETIRED = (
    "instance_seg_id",
    "longterm_seg_id",
    "instance_count",
    "id_mask",
    "instance_embed_id",
    "instance_embedding",
)


@dataclass(frozen=True)
class LegacyLayout:
    """What a legacy pose file holds, as far as conversion needs to know.

    Attributes:
        path: The legacy file.
        frames: ``poseest/points.shape[0]``.
        instances: ``poseest/points.shape[1]``, the legacy instance slots.
        keypoints: ``poseest/points.shape[2]``; always 12.
        identity: Rows of ``instance_id_center`` when it has any, otherwise the
            largest embed id of a valid instance (ADR 0002:1294-1295); 0 when
            there is none.
        version: The ``poseest`` ``version`` attribute, as stored; empty when
            the file has none, since the contents, not the version, identify it.
        has_segmentation: Whether ``poseest/seg_data`` is present.
        contour_axis: ``seg_data.shape[2]``, the padded contour extent; 0
            without segmentation.
        point_axis: ``seg_data.shape[3]``, the padded point extent; 0 without
            segmentation.
        has_tracklets: Whether ``poseest/instance_track_id`` is present.
        has_embeddings: Whether ``poseest/identity_embeds`` is present.
        bbox_generated: Whether ``poseest/bbox`` is present and its
            ``bboxes_generated`` attribute is true, as the v8 reader requires.
        static_objects: Names under ``static_objects``, an empty lixit excluded.
        dynamic_objects: Names under ``dynamic_objects``.
        dropped: Present legacy paths the format retires, which the converter
            leaves behind.
    """

    path: Path
    frames: int
    instances: int
    keypoints: int
    identity: int
    version: tuple[int, ...]
    has_segmentation: bool
    contour_axis: int
    point_axis: int
    has_tracklets: bool
    has_embeddings: bool
    bbox_generated: bool
    static_objects: tuple[str, ...]
    dynamic_objects: tuple[str, ...]
    dropped: tuple[str, ...]


def inspect_legacy(path: Path) -> LegacyLayout:
    """Identify a legacy ``pose_est_v4``..``v8`` file by its contents.

    Args:
        path: The legacy HDF5 file.

    Returns:
        Its layout.

    Raises:
        LegacyConversionError: If the file holds a path outside the catalog,
            lacks a dataset v4 introduced, has other than 12 keypoints, has
            datasets whose shapes disagree, has a ``version`` that is not a
            list of whole numbers, or has a malformed dynamic object. The
            message names the path.
    """
    with h5py.File(path, "r") as h5:
        _check_paths(path, h5)
        pose = h5["poseest"]
        missing = [name for name in _REQUIRED if name not in pose]
        if missing:
            raise LegacyConversionError(
                f"{path}: not a pose_est_v4 or later file; missing poseest/{', poseest/'.join(missing)}"
            )
        if pose["points"].ndim != 4:
            raise LegacyConversionError(
                f"{path}: poseest/points has shape {pose['points'].shape}; expected 4 axes"
            )
        frames, instances, keypoints = pose["points"].shape[:3]
        if keypoints != KEYPOINTS:
            raise LegacyConversionError(
                f"{path}: poseest/points has {keypoints} keypoints; expected {KEYPOINTS}"
            )
        _check_shapes(path, pose, frames, instances)
        version = _version(path, pose)

        seg = pose.get("seg_data")
        bbox = pose.get("bbox")
        bbox_generated = bbox is not None and bool(bbox.attrs.get("bboxes_generated", False))
        statics = h5.get("static_objects", {})
        empty_lixit = "lixit" in statics and statics["lixit"].size == 0
        dynamics = h5.get("dynamic_objects", {})
        for name in dynamics:
            _check_dynamic_object(path, dynamics[name])

        dropped = [f"poseest/{name}" for name in _RETIRED if name in pose]
        if bbox is not None and not bbox_generated:
            dropped.append("poseest/bbox")
        if empty_lixit:
            dropped.append("static_objects/lixit")

        return LegacyLayout(
            path=path,
            frames=int(frames),
            instances=int(instances),
            keypoints=int(keypoints),
            identity=_identity(pose),
            version=version,
            has_segmentation=seg is not None,
            contour_axis=int(seg.shape[2]) if seg is not None else 0,
            point_axis=int(seg.shape[3]) if seg is not None else 0,
            has_tracklets="instance_track_id" in pose,
            has_embeddings="identity_embeds" in pose,
            bbox_generated=bbox_generated,
            static_objects=tuple(
                sorted(name for name in statics if not (name == "lixit" and empty_lixit))
            ),
            dynamic_objects=tuple(sorted(dynamics)),
            dropped=tuple(dropped),
        )


def _check_paths(path: Path, h5: h5py.File) -> None:
    """Refuse any group or dataset outside the legacy catalog, naming it."""
    if "poseest" not in h5:
        _refuse(path, "poseest", "is missing")
    for name in h5:
        if name not in _ROOTS or not isinstance(h5[name], h5py.Group):
            _refuse(path, name, "is not a legacy pose path")
    for name in h5["poseest"]:
        if name not in _POSEEST or not isinstance(h5["poseest"][name], h5py.Dataset):
            _refuse(path, f"poseest/{name}", "is not a legacy pose dataset")
    for name in h5.get("static_objects", {}):
        if name not in _STATIC_OBJECTS or not isinstance(h5["static_objects"][name], h5py.Dataset):
            _refuse(path, f"static_objects/{name}", "is not a legacy static object")
    dynamics = h5.get("dynamic_objects", {})
    for name in dynamics:
        if not isinstance(dynamics[name], h5py.Group):
            _refuse(path, f"dynamic_objects/{name}", "is not a dynamic object group")
        for member in dynamics[name]:
            obj = dynamics[name][member]
            if member not in _DYNAMIC_DATASETS or not isinstance(obj, h5py.Dataset):
                _refuse(
                    path, f"dynamic_objects/{name}/{member}", "is not a dynamic object dataset"
                )


def _check_shapes(path: Path, pose: h5py.Group, frames: int, instances: int) -> None:
    """Refuse ``poseest`` datasets whose shapes disagree with ``points`` or each other.

    A disagreement would otherwise surface mid-conversion as an index error,
    or not at all, with data silently misaligned.
    """
    # None stands for an axis whose extent is the dataset's own.
    expected: dict[str, tuple[int | None, ...]] = {
        "confidence": (frames, instances, KEYPOINTS),
        "id_mask": (frames, instances),
        "instance_embed_id": (frames, instances),
        "instance_track_id": (frames, instances),
        "identity_embeds": (frames, instances, None),
        "seg_data": (frames, instances, None, None, 2),
        "seg_external_flag": (frames, instances, None),
        "bbox": (frames, instances, 2, 2),
        "instance_id_center": (None, None),
    }
    for name, axes in expected.items():
        if name in pose and not _matches(pose[name].shape, axes):
            _refuse_shapes(path, name, pose[name].shape, "poseest/points", pose["points"].shape)
    if "seg_data" in pose and "seg_external_flag" in pose:
        flag, seg = pose["seg_external_flag"].shape, pose["seg_data"].shape
        if flag != seg[:3]:
            _refuse_shapes(path, "seg_external_flag", flag, "poseest/seg_data", seg)
    if "identity_embeds" in pose and "instance_id_center" in pose:
        embeds, centers = pose["identity_embeds"].shape, pose["instance_id_center"].shape
        # Both hold vectors in one embedding space, so their widths agree.
        if embeds[2] != centers[1]:
            _refuse_shapes(path, "identity_embeds", embeds, "poseest/instance_id_center", centers)


def _matches(shape: tuple[int, ...], axes: tuple[int | None, ...]) -> bool:
    """Return whether ``shape`` has ``axes``' rank and every fixed extent."""
    return len(shape) == len(axes) and all(
        a is None or a == s for s, a in zip(shape, axes, strict=True)
    )


def _refuse_shapes(
    path: Path, name: str, shape: tuple[int, ...], other: str, other_shape: tuple[int, ...]
) -> NoReturn:
    """Raise the refusal for a ``poseest`` dataset whose shape disagrees with another's."""
    raise LegacyConversionError(
        f"{path}: poseest/{name} has shape {shape}, which disagrees with {other} {other_shape}"
    )


def _version(path: Path, pose: h5py.Group) -> tuple[int, ...]:
    """Return the ``version`` attribute as whole numbers; empty when it is absent."""
    if "version" not in pose.attrs:
        return ()
    stored = np.atleast_1d(np.asarray(pose.attrs["version"])).ravel()
    whole = stored.dtype.kind in "iu" or (
        stored.dtype.kind == "f"
        and bool(np.all(np.isfinite(stored) & (stored == np.round(stored))))
    )
    if not whole:
        raise LegacyConversionError(
            f"{path}: poseest version attribute {pose.attrs['version']!r} is not a list of whole numbers"
        )
    return tuple(int(v) for v in stored)


def _check_dynamic_object(path: Path, group: h5py.Group) -> None:
    """Refuse a dynamic object the v7 reader would skip or misread."""
    name = group.name.lstrip("/")
    missing = [member for member in _DYNAMIC_DATASETS if member not in group]
    if missing:
        raise LegacyConversionError(f"{path}: {name} is missing {', '.join(missing)}")
    points = group["points"]
    if points.ndim not in (3, 4):
        raise LegacyConversionError(
            f"{path}: {name}/points has ndim {points.ndim}; expected 3 or 4"
        )
    # Absent means "yx" (pose_est_v7.py:15-19); the v7 reader flips anything
    # else it does not know, which would be a guess.
    axis_order = attr_text(points.attrs.get("axis_order", "yx"))
    if axis_order not in _AXIS_ORDERS:
        raise LegacyConversionError(
            f"{path}: {name}/points has axis_order {axis_order!r}; expected 'xy' or 'yx'"
        )


def _identity(pose: h5py.Group) -> int:
    """Return the identity count: center rows, else the largest valid embed id."""
    centers = pose.get("instance_id_center")
    if centers is not None and centers.shape[0] > 0:
        return int(centers.shape[0])
    embed, mask = pose["instance_embed_id"], pose["id_mask"]
    largest = 0
    for start in range(0, embed.shape[0], BLOCK_FRAMES):
        stop = start + BLOCK_FRAMES
        # id_mask true marks an instance with no identity.
        valid = ~mask[start:stop].astype(bool)
        ids = embed[start:stop][valid]
        if ids.size:
            largest = max(largest, int(ids.max()))
    return largest


def _refuse(path: Path, name: str, why: str) -> NoReturn:
    """Raise the refusal for one HDF5 path of the legacy file ``path``."""
    raise LegacyConversionError(f"{path}: {name} {why}")
