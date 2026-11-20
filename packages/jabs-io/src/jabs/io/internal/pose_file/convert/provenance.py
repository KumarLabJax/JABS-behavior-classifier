"""The history a converted file carries: what the legacy file was and said.

The Hub deletes a legacy file once it is converted, so the ``convert`` entry is
the only record of it. It names the source by content hash, lists what the
converter invented and what it left behind, and keeps every legacy attribute
verbatim. No provenance record is written (spec 0029-B §5.1): the schema
requires a producer, version and creation time, and no legacy file records
them all.
"""

import base64
import hashlib
import json
import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.convert import memory
from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import LegacyLayout
from jabs.io.internal.pose_file.convert.plan import PlacementPlan
from jabs.io.internal.pose_file.convert.sources import CONFIDENCE_THRESHOLD
from jabs.io.internal.pose_file.reader import attr_text
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12
from jabs.io.internal.pose_file.types import Component, HistoryEntry

logger = logging.getLogger(__name__)

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

    The values are strict JSON, which every consumer of the provenance
    document can parse: a non-finite float becomes the string ``"NaN"``,
    ``"Infinity"`` or ``"-Infinity"``, an empty (null-dataspace) attribute
    ``None``, and an object reference ``"<reference:{target path}>"``, or
    ``"<reference:null>"`` for a null one. Text that is not UTF-8, whether
    stored as bytes or handed back by h5py with surrogate escapes, becomes
    ``{"base64": "<the stored bytes, base64>"}``, so the bytes are kept
    exactly; that object is the only one a value can hold, and marks it.

    The attributes are capped at ``memory.MAX_ATTRIBUTE_BYTES`` (16 MiB) in
    total, measured as their JSON text plus their names: each attribute's
    declared size is checked against what is left before its value is read,
    and its JSON size, which counts the expansion of references to paths and
    of bytes to base64, is added once it is converted. A reference's path is
    counted as it is resolved, so an array of them stops at the cap.

    Args:
        h5: The open legacy file.

    Returns:
        ``{path: {name: value}}``, ``"/"`` for the root and otherwise the path
        without its leading slash, for each group or dataset that has
        attributes.

    Raises:
        LegacyConversionError: If an attribute cannot be read or has a kind
            with no JSON form, such as a compound value, or if the attributes
            pass ``memory.MAX_ATTRIBUTE_BYTES``, naming the attribute's path
            and name (and the cap).
    """
    found: dict[str, dict] = {}
    cap = memory.MAX_ATTRIBUTE_BYTES
    # The JSON bytes recorded so far, and those of the value being converted.
    total = 0
    pending = 0

    def too_large(name: str, key: str, why: str) -> LegacyConversionError:
        """Return the refusal naming the attribute where the cap was crossed."""
        return LegacyConversionError(
            f"{h5.filename}: attribute {key!r} of {name} {why}; refusing legacy attributes "
            f"over the cap of {cap} bytes in total, which the conversion records verbatim"
        )

    def spend(size: int) -> None:
        """Count part of a value as it expands; raise once the total passes the cap."""
        nonlocal pending
        pending += size
        if total + pending > cap:
            raise _OverCap

    def collect(name: str, obj: h5py.HLObject) -> None:
        """Record one object's attributes, if it has any."""
        nonlocal total, pending
        if not len(obj.attrs):
            return
        values = {}
        for key in obj.attrs:
            pending = 0
            try:
                declared = memory.attribute_bytes(obj.attrs, key)
                if declared > cap - total:
                    raise too_large(name, key, f"declares {declared} bytes")
                value = _json_value(obj.attrs[key], h5, spend)
            except _OverCap:
                raise too_large(name, key, "expands past the cap") from None
            except (TypeError, ValueError, OSError) as error:
                # Leaving it out would lose it for good once the Hub deletes
                # the original.
                raise LegacyConversionError(
                    f"{h5.filename}: attribute {key!r} of {name} cannot be recorded: {error}"
                ) from error
            total += len(json.dumps(value, allow_nan=False)) + len(key)
            if total > cap:
                raise too_large(name, key, f"takes them to {total} bytes")
            values[key] = value
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

    Its ``extra`` holds the plan's rows as columns, one equal-length list per
    field: ``tail_slots`` has ``frame``, ``slot``, ``origin``
    (``"embed_out_of_range"`` or ``"masked"``) and ``legacy_index``;
    ``longterm_mismatches`` has ``frame``, ``legacy_index``, ``embed_id`` and
    ``longterm_seg_id``; ``tracklet_splits`` has ``id``, ``from``,
    ``first_frame`` and ``last_frame``. It also holds ``point_valid``'s
    threshold, ``legacy_attributes``, ``tracklet_id_none`` when the file has
    tracklets, and ``model_metadata_json`` when the file has one.

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
    if len(plan.tracklet_splits):
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

    tails, mismatches, splits = plan.tail_slots, plan.longterm_mismatches, plan.tracklet_splits
    # Columns, one list per field, rather than a record per row: an unusual
    # file can put an instance in a tail slot in most of its frames.
    extra: dict = {
        "tail_slots": {
            "frame": tails.frame.tolist(),
            "slot": tails.slot.tolist(),
            "origin": np.where(tails.masked, "masked", "embed_out_of_range").tolist(),
            "legacy_index": tails.legacy_index.tolist(),
        },
        "longterm_mismatches": {
            "frame": mismatches.frame.tolist(),
            "legacy_index": mismatches.legacy_index.tolist(),
            "embed_id": mismatches.embed_id.tolist(),
            "longterm_seg_id": mismatches.longterm_seg_id.tolist(),
        },
        "tracklet_splits": {
            "id": splits.id.tolist(),
            "from": splits.from_id.tolist(),
            "first_frame": splits.first_frame.tolist(),
            "last_frame": splits.last_frame.tolist(),
        },
        # The policy point_valid was derived by (ADR 0002:752); the converter
        # writes no record to carry it.
        "point_valid": {"confidence_threshold": CONFIDENCE_THRESHOLD},
        "legacy_attributes": legacy_attributes(h5),
    }
    if plan.tracklet_id is not None:
        # The id that means no tracklet: legacy tracklet 0 and every
        # unoccupied (frame, slot) carry it.
        extra["tracklet_id_none"] = 0
    if "model_metadata_json" in h5["poseest"].attrs:
        # Recorded whole, as legacy_attributes records it; legacy_attributes
        # above has already refused a value with no JSON form.
        extra["model_metadata_json"] = _json_value(h5["poseest"].attrs["model_metadata_json"], h5)

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
        ``"pose_est_v<major>"``, or ``"pose_est"`` when the file has no
        version ``inspect_legacy`` could read as whole numbers.
    """
    return f"pose_est_v{layout.version[0]}" if layout.version else "pose_est"


def _join(names: list[str]) -> str:
    """Join names as prose: ``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


class _OverCap(Exception):
    """A value's expansion passed ``memory.MAX_ATTRIBUTE_BYTES``."""


def _json_value(
    value: object, h5: h5py.File | None, spend: Callable[[int], None] | None = None
) -> object:
    """Return an HDF5 attribute value as strict JSON; raise TypeError when it has no JSON form.

    ``spend``, when given, is called with each reference's path length as it
    is resolved, so a caller can stop an array of them at a cap.
    """
    if isinstance(value, h5py.Empty):
        return None
    if isinstance(value, h5py.Reference):
        text = _reference(value, h5)
        if spend is not None:
            spend(len(text))
        return text
    # np.bytes_ is a bytes subclass.
    if isinstance(value, bytes):
        return _text(bytes(value))
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            # h5py decodes a variable-length string's undecodable bytes as
            # surrogate escapes; any other lone surrogate passes through.
            try:
                return _text(value.encode("utf-8", "surrogateescape"))
            except UnicodeEncodeError:
                return _text(value.encode("utf-8", "surrogatepass"))
        return value
    if isinstance(value, bool | int) or value is None:
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
        return _json_value(value.tolist(), h5, spend)
    if isinstance(value, np.void):
        raise TypeError(f"a compound value of dtype {value.dtype} has no JSON form")
    if isinstance(value, np.generic):
        return _json_value(value.item(), h5, spend)
    if isinstance(value, list | tuple):
        return [_json_value(item, h5, spend) for item in value]
    raise TypeError(f"a value of type {type(value).__name__} has no JSON form")


def _text(raw: bytes) -> str | dict[str, str]:
    """Return bytes as their UTF-8 text, or ``{"base64": ...}`` when they are not UTF-8."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # Replacing what does not decode would lose it for good.
        return {"base64": base64.b64encode(raw).decode("ascii")}


def _reference(reference: h5py.Reference, h5: h5py.File | None) -> str:
    """Return an object reference as ``"<reference:{target path}>"``, or ``"<reference:null>"``."""
    if not reference:
        return "<reference:null>"
    if h5 is None:
        raise TypeError("an object reference cannot be resolved without its file")
    return f"<reference:{h5[reference].name}>"
