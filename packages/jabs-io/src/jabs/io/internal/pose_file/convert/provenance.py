"""The history a converted file carries: what the legacy file was and said.

The Hub deletes a legacy file once it is converted, so the ``convert`` entry is
the only record of it. It names the source by content hash, lists what the
converter invented and what it left behind, and keeps every legacy attribute
verbatim. No provenance record is written (spec 0029-B §5.1): the schema
requires a producer, version and creation time, and no legacy file records
them all.
"""

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import LegacyLayout
from jabs.io.internal.pose_file.convert.plan import PlacementPlan
from jabs.io.internal.pose_file.convert.sources import CONFIDENCE_THRESHOLD
from jabs.io.internal.pose_file.reader import attr_text
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12
from jabs.io.internal.pose_file.types import Component, HistoryEntry

TOOL = "jabs-io"

_HASH_CHUNK = 2**20
_VIDEO_FIELDS = ("width", "height", "fps")
# The manifest schema's cm_per_pixel_source values, null aside.
_SCALE_SOURCES = frozenset({"corner_detection", "manually_set", "default_alignment"})


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

    The values are strict JSON, which every consumer of the provenance
    document can parse: a non-finite float becomes the string ``"NaN"``,
    ``"Infinity"`` or ``"-Infinity"``, an empty (null-dataspace) attribute
    ``None``, and an object reference ``"<reference:{target path}>"``, or
    ``"<reference:null>"`` for a null one.

    Args:
        h5: The open legacy file.

    Returns:
        ``{path: {name: value}}``, ``"/"`` for the root and otherwise the path
        without its leading slash, for each group or dataset that has
        attributes.

    Raises:
        LegacyConversionError: If an attribute cannot be read or has a kind
            with no JSON form, such as a compound value, naming its path and
            name.
    """
    found: dict[str, dict] = {}

    def collect(name: str, obj: h5py.HLObject) -> None:
        """Record one object's attributes, if it has any."""
        if not len(obj.attrs):
            return
        values = {}
        for key in obj.attrs:
            try:
                values[key] = _json_value(obj.attrs[key], h5)
            except (TypeError, ValueError, OSError) as error:
                # Leaving it out would lose it for good once the Hub deletes
                # the original.
                raise LegacyConversionError(
                    f"{h5.filename}: attribute {key!r} of {name} cannot be recorded: {error}"
                ) from error
        found[name] = values

    collect("/", h5)
    h5.visititems(collect)
    return found


def legacy_scale(attrs: h5py.AttributeManager) -> tuple[float | None, str | None, tuple[str, ...]]:
    """Return the ``poseest`` scale and its source as the manifest can hold them.

    A value the manifest cannot hold is written as unknown rather than
    refusing the file: the converter only passes the scale through, and ADR
    0002 prefers an explicit unknown. The raw value stays in
    ``legacy_attributes``.

    Args:
        attrs: The ``poseest`` group's attributes.

    Returns:
        ``cm_per_pixel`` (a finite number above 0, else None),
        ``cm_per_pixel_source`` (a schema value, else None, and None whenever
        the scale is) and one note per value that was present but nulled.
    """
    notes = []
    raw_scale = attrs.get("cm_per_pixel")
    scale = None
    if raw_scale is not None:
        try:
            number = _json_value(raw_scale, None)
        except (TypeError, ValueError):
            # legacy_attributes refuses it, naming it.
            number = repr(raw_scale)
        if isinstance(number, list) and len(number) == 1:
            number = number[0]
        if (
            isinstance(number, int | float)
            and not isinstance(number, bool)
            and math.isfinite(number)
            and number > 0
        ):
            scale = float(number)
        else:
            notes.append(
                f"video.cm_per_pixel is null: the legacy value {number!r} is not a positive number."
            )

    raw_source = attrs.get("cm_per_pixel_source")
    source = None
    if raw_source is not None:
        text = attr_text(raw_source)
        if text not in _SCALE_SOURCES:
            notes.append(
                f"video.cm_per_pixel_source is null: the legacy value "
                f"{text if text is not None else repr(raw_source)!r} is not a recognized "
                "source."
            )
        elif scale is None:
            notes.append(
                f"video.cm_per_pixel_source is null: the legacy value {text!r} is the source "
                "of a scale that is not written, and means nothing without it."
            )
        else:
            source = text
    return scale, source, tuple(notes)


def convert_entry(
    *,
    h5: h5py.File,
    layout: LegacyLayout,
    plan: PlacementPlan,
    content_hash: str,
    version: str,
    time: str,
    video: dict[str, object],
    scale_notes: Sequence[str] = (),
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
        scale_notes: ``legacy_scale``'s notes on scale values written as null.

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
    sentences = list(scale_notes)
    if unknown:
        sentences.insert(
            0,
            f"{_join(unknown)} {'is' if len(unknown) == 1 else 'are'} null: "
            "not recorded in legacy pose files.",
        )
    notes = " ".join(sentences) or None

    extra: dict = {
        "tail_slots": list(plan.tail_slots),
        "longterm_mismatches": list(plan.longterm_mismatches),
        "tracklet_splits": list(plan.tracklet_splits),
        # The policy point_valid was derived by (ADR 0002:752); the converter
        # writes no record to carry it.
        "point_valid": {"confidence_threshold": CONFIDENCE_THRESHOLD},
        "legacy_attributes": legacy_attributes(h5),
    }
    if plan.tracklet_id is not None:
        # The id that means no tracklet: legacy tracklet 0 and every
        # unoccupied (frame, slot) carry it.
        extra["tracklet_id_none"] = 0
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


def _json_value(value: object, h5: h5py.File | None) -> object:
    """Return an HDF5 attribute value as strict JSON; raise TypeError when it has no JSON form."""
    if isinstance(value, h5py.Empty):
        return None
    if isinstance(value, h5py.Reference):
        return _reference(value, h5)
    # np.bytes_ is a bytes subclass.
    if isinstance(value, bytes):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        # Strict JSON has no NaN or infinity; strings keep the value.
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, np.ndarray):
        if value.dtype.names is not None:
            raise TypeError(f"a compound value of dtype {value.dtype} has no JSON form")
        # A 0-d array's tolist() is a scalar, not a list; an object array's
        # holds the objects (strings, references), converted in turn.
        return _json_value(value.tolist(), h5)
    if isinstance(value, np.void):
        raise TypeError(f"a compound value of dtype {value.dtype} has no JSON form")
    if isinstance(value, np.generic):
        return _json_value(value.item(), h5)
    if isinstance(value, list | tuple):
        return [_json_value(item, h5) for item in value]
    raise TypeError(f"a value of type {type(value).__name__} has no JSON form")


def _reference(reference: h5py.Reference, h5: h5py.File | None) -> str:
    """Return an object reference as ``"<reference:{target path}>"``, or ``"<reference:null>"``."""
    if not reference:
        return "<reference:null>"
    if h5 is None:
        raise TypeError("an object reference cannot be resolved without its file")
    return f"<reference:{h5[reference].name}>"
