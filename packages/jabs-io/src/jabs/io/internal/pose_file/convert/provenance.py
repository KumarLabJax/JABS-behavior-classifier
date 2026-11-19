"""The history a converted file carries: what the legacy file was and said.

The Hub deletes a legacy file once it is converted, so the ``convert`` entry is
the only record of it. It names the source by content hash, lists what the
converter invented and what it left behind, and keeps every legacy attribute
verbatim. No provenance record is written (spec 0029-B §5.1): the schema
requires a producer, version and creation time, and no legacy file records
them all.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.convert.inspect import LegacyLayout
from jabs.io.internal.pose_file.convert.plan import PlacementPlan
from jabs.io.internal.pose_file.convert.sources import CONFIDENCE_THRESHOLD
from jabs.io.internal.pose_file.reader import attr_text
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12
from jabs.io.internal.pose_file.types import Component, HistoryEntry

TOOL = "jabs-io"

_HASH_CHUNK = 2**20
_VIDEO_FIELDS = ("width", "height", "fps")


def legacy_hash(path: Path) -> str:
    """Return a legacy file's content hash, as the Hub computes it.

    Args:
        path: The legacy file.

    Returns:
        ``"blake2b:"`` and the lowercase hex of its 256-bit BLAKE2b digest.
    """
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as stream:
        # In chunks: a legacy file can be gigabytes.
        while chunk := stream.read(_HASH_CHUNK):
            digest.update(chunk)
    return f"blake2b:{digest.hexdigest()}"


def legacy_attributes(h5: h5py.File) -> dict[str, dict]:
    """Return every attribute of the legacy file, keyed by HDF5 path, as JSON values.

    Args:
        h5: The open legacy file.

    Returns:
        ``{path: {name: value}}``, ``"/"`` for the root and otherwise the path
        without its leading slash, for each group or dataset that has
        attributes.
    """
    found: dict[str, dict] = {}

    def collect(name: str, obj: h5py.HLObject) -> None:
        """Record one object's attributes, if it has any."""
        if len(obj.attrs):
            found[name] = {key: _json_value(value) for key, value in obj.attrs.items()}

    collect("/", h5)
    h5.visititems(collect)
    return found


def convert_entry(
    *,
    h5: h5py.File,
    layout: LegacyLayout,
    plan: PlacementPlan,
    content_hash: str,
    version: str,
    time: str,
    video: dict[str, object],
) -> HistoryEntry:
    """Return the ``convert`` history entry for one legacy file.

    Args:
        h5: The open legacy file, for its attributes.
        layout: What ``inspect_legacy`` found in it.
        plan: Where ``plan_placement`` put each legacy instance.
        content_hash: ``legacy_hash`` of the file.
        version: This package's version.
        time: The conversion's ISO 8601 timestamp.
        video: The written ``width``, ``height`` and ``fps``; None where unknown.

    Returns:
        The entry.
    """
    synthesized = [
        f"skeletons.{JABS_MOUSE12}",
        "jabs.pose.point_valid",
        "jabs.pose.slot_occupied",
        "dimensions.identity",
    ]
    synthesized += [f"video.{name}" for name in _VIDEO_FIELDS if video[name] is not None]
    if plan.tracklet_splits:
        synthesized.append("jabs.pose.tracklet_id")
    unknown = [f"video.{name}" for name in _VIDEO_FIELDS if video[name] is None]
    notes = (
        f"{_join(unknown)} {'is' if len(unknown) == 1 else 'are'} null: "
        "not recorded in legacy pose files."
        if unknown
        else None
    )

    extra: dict = {
        "tail_slots": list(plan.tail_slots),
        "longterm_mismatches": list(plan.longterm_mismatches),
        "tracklet_splits": list(plan.tracklet_splits),
        # The policy point_valid was derived by (ADR 0002:752); the converter
        # writes no record to carry it.
        "point_valid": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "requires_finite_coordinates": True,
        },
        "legacy_attributes": legacy_attributes(h5),
    }
    metadata = h5["poseest"].attrs.get("model_metadata_json")
    if metadata is not None:
        extra["model_metadata_json"] = attr_text(metadata)

    return HistoryEntry(
        operation="convert",
        tool=TOOL,
        version=version,
        time=time,
        source={
            "format": source_format(layout),
            "content_hash": content_hash,
            "filename": layout.path.name,
        },
        synthesized=tuple(synthesized),
        dropped=layout.dropped,
        notes=notes,
        extra=extra,
    )


def encode_entry(
    components: Sequence[Component], *, version: str, time: str
) -> HistoryEntry | None:
    """Return the ``encode`` history entry naming every alternate added, or None if none was.

    Args:
        components: The converted file's components.
        version: This package's version.
        time: The conversion's ISO 8601 timestamp.

    Returns:
        The entry, or None when no component carries an alternate.
    """
    added = tuple(
        f"{component.id}:alternate:{alternate.name}"
        for component in components
        for alternate in component.alternates
    )
    if not added:
        return None
    return HistoryEntry(
        operation="encode", tool=TOOL, version=version, time=time, synthesized=added
    )


def source_format(layout: LegacyLayout) -> str:
    """Return the legacy dialect as its ``version`` attribute names it.

    Args:
        layout: What ``inspect_legacy`` found.

    Returns:
        ``"pose_est_v<major>"``, or ``"pose_est"`` when the file has no version.
    """
    return f"pose_est_v{layout.version[0]}" if layout.version else "pose_est"


def _join(names: list[str]) -> str:
    """Join names as prose: ``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _json_value(value: object) -> object:
    """Return an HDF5 attribute value as plain JSON: numbers, strings, bools and lists."""
    # np.bytes_ is a bytes subclass.
    if isinstance(value, bytes):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        # A 0-d array's tolist() is a scalar, not a list.
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value
