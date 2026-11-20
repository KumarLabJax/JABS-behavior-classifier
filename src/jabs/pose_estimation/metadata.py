"""Metadata-only scan of a pose file, unified or legacy.

The project scan needs a handful of small facts about every pose file (frame
count, identities, static objects, scale, capabilities) without loading any
keypoint data. A unified file answers from its manifest and provenance; a
legacy file from :func:`~jabs.io.internal.pose_file.convert.inspect_legacy`,
which checks shapes and attributes, plus its ``poseest`` attributes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from jabs.core.abstract.pose_est import parse_model_metadata
from jabs.core.enums import PoseCapability
from jabs.core.utils.static_objects import lixit_keypoint_count
from jabs.io.internal.pose_file import ComponentReader, PoseFileError
from jabs.io.internal.pose_file.convert import LegacyConversionError, inspect_legacy
from jabs.io.internal.pose_file.convert.provenance import legacy_scale

from .discovery import POSE_FILE_GLOBS, is_legacy_pose_file, is_unified_pose_file
from .errors import PoseOpenError
from .pose import (
    BBOX_COMPONENT,
    STATIC_OBJECT_PREFIX,
    _last_convert_entry,
    converted_from_format,
    has_segmentation_components,
    identity_count,
    resolve_source_format,
)

logger = logging.getLogger(__name__)

__all__ = ["PoseMetadata", "scan_model_metadata", "scan_pose_metadata"]

_LIXIT = "lixit"


@dataclass(frozen=True)
class PoseMetadata:
    """What the project scan needs to know about one pose file.

    Attributes:
        frames: Number of frames.
        num_identities: Number of identities, as ``Pose`` counts them.
        static_objects: Names of the static objects, sorted; an empty lixit
            is not one.
        lixit_keypoints: Keypoints per lixit, 0 without a lixit.
        cm_per_pixel: Centimeters per pixel, or None when unknown.
        capabilities: Optional data the file provides.
        source_format: ``"pose_est_v<N>"`` or ``"jabs.pose-file"``, by the
            ``PoseLike`` rule.
    """

    frames: int
    num_identities: int
    static_objects: tuple[str, ...]
    lixit_keypoints: int
    cm_per_pixel: float | None
    capabilities: frozenset[PoseCapability]
    source_format: str


def _capabilities(
    num_identities: int, segmentation: bool, bounding_boxes: bool
) -> frozenset[PoseCapability]:
    """Return the capabilities for the given facts."""
    capabilities = set()
    if num_identities > 1:
        capabilities.add(PoseCapability.SOCIAL)
    if segmentation:
        capabilities.add(PoseCapability.SEGMENTATION)
    if bounding_boxes:
        capabilities.add(PoseCapability.BOUNDING_BOXES)
    return frozenset(capabilities)


def _scan_unified(path: Path) -> PoseMetadata:
    """Scan a unified pose file from its manifest and provenance."""
    with ComponentReader(path) as reader:
        manifest = reader.manifest
        history = reader.read_provenance().history
    converted_from = converted_from_format(_last_convert_entry(history))
    num_identities = identity_count(manifest, converted_from)

    component_ids = {spec["id"] for spec in manifest["components"]}
    static_objects: list[str] = []
    lixit_keypoints = 0
    for spec in manifest["components"]:
        if spec["id"].startswith(STATIC_OBJECT_PREFIX):
            name = spec["id"][len(STATIC_OBJECT_PREFIX) :]
            static_objects.append(name)
            if name == _LIXIT:
                # (O, P, 2): keypoints per lixit is P
                lixit_keypoints = int(spec["shape"][1])

    scale = manifest["video"].get("cm_per_pixel")
    return PoseMetadata(
        frames=int(manifest["dimensions"]["frame"]),
        num_identities=num_identities,
        static_objects=tuple(sorted(static_objects)),
        lixit_keypoints=lixit_keypoints,
        cm_per_pixel=None if scale is None else float(scale),
        capabilities=_capabilities(
            num_identities,
            has_segmentation_components(component_ids),
            BBOX_COMPONENT in component_ids,
        ),
        source_format=resolve_source_format(path, converted_from),
    )


def _scan_legacy(path: Path) -> PoseMetadata:
    """Scan a legacy pose file from its layout and ``poseest`` attributes."""
    layout = inspect_legacy(path)
    if layout.placed_by_track:
        # pose_est_v3: one identity per track slot, as its conversion has
        num_identities = layout.track_slots
    elif not layout.instance_axis:
        # pose_est_v2: the one animal
        num_identities = 1
    else:
        num_identities = layout.identity

    lixit_keypoints = 0
    with h5py.File(path, "r") as h5:
        cm_per_pixel, _, _ = legacy_scale(h5["poseest"].attrs)
        if _LIXIT in layout.static_objects:
            # shape only, no data read
            lixit_keypoints = lixit_keypoint_count(h5["static_objects"][_LIXIT].shape)

    return PoseMetadata(
        frames=layout.frames,
        num_identities=num_identities,
        static_objects=tuple(sorted(layout.static_objects)),
        lixit_keypoints=lixit_keypoints,
        cm_per_pixel=cm_per_pixel,
        capabilities=_capabilities(num_identities, layout.has_segmentation, layout.bbox_generated),
        source_format=resolve_source_format(path, None),
    )


def _scan_unified_model_metadata(path: Path) -> dict[str, Any] | None:
    """Read the model metadata a unified file's last convert entry preserved."""
    with ComponentReader(path) as reader:
        convert = _last_convert_entry(reader.read_provenance().history)
    extra = (convert.extra or {}) if convert is not None else {}
    return parse_model_metadata(extra.get("model_metadata_json"))


def _scan_legacy_model_metadata(path: Path) -> dict[str, Any] | None:
    """Read a legacy file's ``poseest`` ``model_metadata_json`` attribute."""
    with h5py.File(path, "r") as h5:
        return parse_model_metadata(h5["poseest"].attrs.get("model_metadata_json"))


def _reason(path: Path, error: Exception) -> str:
    """Return an error's message without the ``<path>: `` prefix it may start with."""
    return str(error).removeprefix(f"{path}: ") or type(error).__name__


def scan_pose_metadata(path: Path) -> PoseMetadata:
    """Read a pose file's metadata without reading keypoint data.

    A unified ``*_pose.h5`` file is described by its manifest and provenance;
    a legacy ``*_pose_est_v<N>.h5`` file by the converter's inspection of its
    shapes and attributes, so a legacy file and its conversion scan alike
    (except ``source_format``, which follows the file's name).

    Args:
        path: The pose file the user has.

    Returns:
        Its metadata.

    Raises:
        PoseOpenError: If the name follows neither pattern, or the file cannot
            be read, is not a pose file, is malformed, or would be refused by
            the converter.
    """
    try:
        if is_unified_pose_file(path):
            return _scan_unified(path)
        if is_legacy_pose_file(path):
            return _scan_legacy(path)
    except (
        LegacyConversionError,
        PoseFileError,
        NotImplementedError,
        OSError,
        # a malformed manifest or attribute that passes the schema
        LookupError,
        ValueError,
    ) as error:
        raise PoseOpenError(path, _reason(path, error)) from error
    raise PoseOpenError(
        path, f"not a pose file name; expected one of {', '.join(POSE_FILE_GLOBS)}"
    )


def scan_model_metadata(path: Path) -> dict[str, Any] | None:
    """Read a pose file's model metadata without reading keypoint data.

    Gives what ``model_metadata`` of the opened pose gives: for a unified file,
    the metadata its conversion preserved; for a legacy file, its ``poseest``
    ``model_metadata_json`` attribute.

    Args:
        path: The pose file the user has.

    Returns:
        The parsed metadata, or None when the file has none.

    Raises:
        PoseOpenError: If the name follows neither pattern, or the file cannot
            be read or is malformed.
    """
    try:
        if is_unified_pose_file(path):
            return _scan_unified_model_metadata(path)
        if is_legacy_pose_file(path):
            return _scan_legacy_model_metadata(path)
    except (PoseFileError, OSError, LookupError, ValueError) as error:
        raise PoseOpenError(path, _reason(path, error)) from error
    raise PoseOpenError(
        path, f"not a pose file name; expected one of {', '.join(POSE_FILE_GLOBS)}"
    )
