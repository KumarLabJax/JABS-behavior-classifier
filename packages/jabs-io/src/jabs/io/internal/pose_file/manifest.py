"""Construction and parsing of the contents manifest (ADR 0002).

The manifest is one JSON document at a fixed path rather than a tree of HDF5
attributes, for three reasons: a range reader discovers the whole contents in
one small read; JSON is order-independent and trivially extensible, which is
what additive-only evolution needs; and it makes validation a JSON Schema
rather than bespoke traversal code.

Two conventions this module enforces:

* Optional *component* declarations are **omitted** when unset, because the
  schema forbids nulls there.
* Unknown *video* fields are emitted as explicit ``null``, because "we do not
  know the frame dimensions" is a fact the file must be able to state.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from jabs.io.internal.pose_file.schema import FORMAT_ID, SCHEMA_REVISION
from jabs.io.internal.pose_file.types import (
    Component,
    HistoryEntry,
    PoseFile,
    Provenance,
    ProvenanceRecord,
    Skeleton,
    VideoInfo,
)

# Video fields that are always present, as null when unknown. `clip_of` is not
# among them: the schema types it as an object, so it is omitted when absent.
_NULLABLE_VIDEO_FIELDS = (
    "width",
    "height",
    "fps",
    "cm_per_pixel",
    "cm_per_pixel_source",
    "start_time",
    "filename",
    "content_hash",
)


@dataclass(frozen=True)
class ParsedManifest:
    """The typed view of a manifest that a reader works from.

    Attributes:
        dimensions: Named dimension sizes.
        video: What the file knows about its video.
        skeletons: Skeletons by id.
        component_specs: The raw component entries, left as dicts so the reader
            can decide what to load rather than being forced to load everything.
        attachment_specs: The raw attachment entries.
        extra: File-level namespaced metadata, or None.
    """

    dimensions: dict[str, int]
    video: VideoInfo
    skeletons: dict[str, Skeleton]
    component_specs: tuple[dict, ...]
    attachment_specs: tuple[dict, ...] = ()
    extra: dict | None = None


def _component_entry(component: Component, layout: dict | None) -> dict:
    """Build one component's manifest entry.

    Args:
        component: The component to describe.
        layout: The storage layout the writer will actually use, or None.

    Returns:
        The manifest entry.
    """
    entry: dict = {
        "id": component.id,
        "path": component.path,
        "axes": list(component.axes),
        "dtype": component.dtype,
        "shape": [int(n) for n in component.data.shape],
        "encoding": dict(component.encoding),
        "missing": dict(component.missing),
    }
    optional = {
        "units": component.units,
        "coord_order": component.coord_order,
        "skeleton": component.skeleton,
        "provenance": component.provenance,
        "description": component.description,
    }
    entry.update({k: v for k, v in optional.items() if v is not None})
    if component.extra is not None:
        entry["extra"] = dict(component.extra)
    if component.sparse_index is not None:
        entry["sparse"] = {"index": component.sparse_index}
    resolved_layout = layout if layout is not None else component.layout
    if resolved_layout is not None:
        entry["layout"] = dict(resolved_layout)
    return entry


def build_manifest(
    pose_file: PoseFile,
    layouts: dict[str, dict] | None = None,
    created: str | None = None,
) -> dict:
    """Build the manifest document for a pose file.

    Args:
        pose_file: The file's contents.
        layouts: Storage layout per component id, as the writer will apply it.
            Supplied by the writer so the manifest reports what is actually on
            disk rather than an intention. This deliberately overrides any
            layout already on a component: layout describes the storage a file
            really has, so the writer's decision is the truth and a caller's
            preference is not.
        created: Creation timestamp. Defaults to now.

    Returns:
        The manifest, ready to validate and serialize.
    """
    layouts = layouts or {}
    video = {"frame_count": int(pose_file.video.frame_count)}
    video.update({name: getattr(pose_file.video, name) for name in _NULLABLE_VIDEO_FIELDS})
    if pose_file.video.clip_of is not None:
        video["clip_of"] = dict(pose_file.video.clip_of)

    skeletons = {}
    for skeleton_id, skeleton in pose_file.skeletons.items():
        entry: dict = {
            "body_parts": list(skeleton.body_parts),
            "edges": [[int(a), int(b)] for a, b in skeleton.edges],
        }
        if skeleton.description is not None:
            entry["description"] = skeleton.description
        skeletons[skeleton_id] = entry

    manifest: dict = {
        "format": FORMAT_ID,
        "schema_revision": SCHEMA_REVISION,
        "created": created or datetime.now(timezone.utc).isoformat(),
        "dimensions": {k: int(v) for k, v in pose_file.dimensions.items()},
        "video": video,
        "components": [
            _component_entry(component, layouts.get(component.id))
            for component in pose_file.components
        ],
    }
    if skeletons:
        manifest["skeletons"] = skeletons
    if pose_file.attachments:
        manifest["attachments"] = [
            {
                key: value
                for key, value in (
                    ("path", attachment.path),
                    ("description", attachment.description),
                    ("content_type", attachment.content_type),
                )
                if value is not None
            }
            for attachment in pose_file.attachments
        ]
    if pose_file.extra is not None:
        manifest["extra"] = dict(pose_file.extra)
    return manifest


def build_provenance(provenance: Provenance) -> dict:
    """Build the provenance document.

    Args:
        provenance: Records and history.

    Returns:
        The provenance document, ready to validate and serialize.
    """
    records = {}
    for key, record in provenance.records.items():
        entry: dict = {
            "producer": record.producer,
            "version": record.version,
            "created": record.created,
        }
        for name in ("model", "algorithm", "parameters"):
            value = getattr(record, name)
            if value is not None:
                entry[name] = dict(value)
        records[key] = entry

    history = []
    for item in provenance.history:
        entry = {
            "operation": item.operation,
            "tool": item.tool,
            "version": item.version,
            "time": item.time,
        }
        if item.source is not None:
            entry["source"] = dict(item.source)
        if item.synthesized is not None:
            entry["synthesized"] = list(item.synthesized)
        if item.dropped is not None:
            entry["dropped"] = list(item.dropped)
        if item.notes is not None:
            entry["notes"] = item.notes
        history.append(entry)

    return {"records": records, "history": history}


def parse_manifest(manifest: dict) -> ParsedManifest:
    """Parse a manifest into its typed view.

    Args:
        manifest: A validated manifest document.

    Returns:
        The dimensions, video metadata, skeletons and raw component entries.
    """
    video_fields = {
        name: manifest["video"].get(name)
        for name in ("frame_count", *_NULLABLE_VIDEO_FIELDS, "clip_of")
    }
    skeletons = {
        skeleton_id: Skeleton(
            body_parts=tuple(entry["body_parts"]),
            edges=tuple((int(a), int(b)) for a, b in entry.get("edges", ())),
            description=entry.get("description"),
        )
        for skeleton_id, entry in manifest.get("skeletons", {}).items()
    }
    return ParsedManifest(
        dimensions=dict(manifest["dimensions"]),
        video=VideoInfo(**video_fields),
        skeletons=skeletons,
        component_specs=tuple(manifest["components"]),
        attachment_specs=tuple(manifest.get("attachments", ())),
        extra=manifest.get("extra"),
    )


def parse_provenance(provenance: dict) -> Provenance:
    """Parse a provenance document into its typed view.

    Args:
        provenance: A validated provenance document.

    Returns:
        The records and history.
    """
    records = {
        key: ProvenanceRecord(
            producer=entry["producer"],
            version=entry["version"],
            created=entry["created"],
            model=entry.get("model"),
            algorithm=entry.get("algorithm"),
            parameters=entry.get("parameters"),
        )
        for key, entry in provenance.get("records", {}).items()
    }
    history = tuple(
        HistoryEntry(
            operation=entry["operation"],
            tool=entry["tool"],
            version=entry["version"],
            time=entry["time"],
            source=entry.get("source"),
            synthesized=tuple(entry["synthesized"]) if "synthesized" in entry else None,
            dropped=tuple(entry["dropped"]) if "dropped" in entry else None,
            notes=entry.get("notes"),
        )
        for entry in provenance.get("history", ())
    )
    return Provenance(records=records, history=history)
