"""Validation of a pose file against the specification (ADR 0002).

A valid file is a well-formed manifest: every component is optional, and
consumers declare their own requirements on top. This module implements the
ADR's validation table, and nothing more opinionated than it.

Two properties matter as much as the checks themselves. **This function
reports, it never raises** — it is the tool aimed at untrusted and damaged
files, so every malformed input must come back as a Finding. And **one problem
does not hide the rest**: only a manifest that fails its schema stops the pass,
because every structural check below assumes a well-formed manifest.

Findings carry a stable ``check`` name so callers and tests can assert on a
specific rule rather than on message text.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.reader import attr_text
from jabs.io.internal.pose_file.schema import (
    FORMAT_ID,
    SCHEMA_REVISION,
    validate_manifest,
    validate_provenance,
)
from jabs.io.internal.pose_file.types import RESERVED_NAMESPACE

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """One validation result.

    Attributes:
        severity: ``"error"`` or ``"warning"``.
        check: Stable name of the rule that produced this finding.
        message: Human-readable explanation.
    """

    severity: str
    check: str
    message: str


def _check_root(h5: h5py.File, findings: list[Finding]) -> bool:
    """Validate the root attributes.

    A missing or unreadable ``schema_revision`` is recorded and the pass
    continues: it tells us nothing about whether the payloads are sound, and
    stopping here would skip every check that does.

    Args:
        h5: The open file.
        findings: Accumulator.

    Returns:
        True when the file identifies as a pose file and can be checked further.
    """
    declared = attr_text(h5.attrs.get("jabs_format"))
    if declared != FORMAT_ID:
        findings.append(
            Finding(
                ERROR,
                "root_attrs",
                f"jabs_format is {declared!r}, expected {FORMAT_ID!r}",
            )
        )
        return False

    if "schema_revision" not in h5.attrs:
        findings.append(Finding(ERROR, "root_attrs", "schema_revision attribute is missing"))
        return True

    revision = _int_attr(h5.attrs["schema_revision"])
    if revision is None:
        findings.append(
            Finding(
                ERROR,
                "root_attrs",
                f"schema_revision is not an integer: {h5.attrs['schema_revision']!r}",
            )
        )
    elif revision > SCHEMA_REVISION:
        # Recorded, never acted on: the format is additive-only, so a newer
        # revision is readable. This is provenance, not a branch.
        findings.append(
            Finding(
                WARNING,
                "root_attrs",
                f"file declares schema_revision {revision}, this build knows {SCHEMA_REVISION}",
            )
        )
    return True


def _int_attr(value: object) -> int | None:
    """Coerce an HDF5 attribute to ``int`` without raising.

    A shape-``(1,)`` integer attribute is what legacy JABS writes, and ``int()``
    on one raises in numpy 2.

    Args:
        value: The raw attribute value.

    Returns:
        The integer, or None when the value is not one.
    """
    try:
        flat = np.atleast_1d(np.asarray(value)).ravel()
        if flat.size != 1:
            return None
        return int(flat[0])
    except (TypeError, ValueError):
        return None


def _read_json(h5: h5py.File, name: str) -> tuple[object | None, str | None]:
    """Read and parse one of the file's JSON documents.

    Args:
        h5: The open file.
        name: Dataset name.

    Returns:
        A ``(document, error)`` pair; exactly one is None.
    """
    try:
        raw = h5[name][()]
    except (KeyError, TypeError) as error:
        return None, f"/{name} is missing or unreadable: {error}"
    try:
        return json.loads(raw), None
    except (TypeError, ValueError) as error:
        return None, f"/{name} is not valid JSON: {error}"


def _check_documents(h5: h5py.File, findings: list[Finding]) -> tuple[dict | None, dict]:
    """Validate the manifest and provenance documents.

    Args:
        h5: The open file.
        findings: Accumulator.

    Returns:
        A ``(manifest, provenance)`` pair. The manifest is None when it is
        unusable for further checks; the provenance is ``{}`` when unusable, so
        that later reference checks degrade rather than crash.
    """
    manifest, error = _read_json(h5, "manifest")
    if error is not None:
        findings.append(Finding(ERROR, "manifest_schema", error))
        manifest = None
    elif not isinstance(manifest, dict):
        findings.append(Finding(ERROR, "manifest_schema", "/manifest is not a JSON object"))
        manifest = None
    else:
        for message in validate_manifest(manifest):
            findings.append(Finding(ERROR, "manifest_schema", message))

    provenance, error = _read_json(h5, "provenance")
    if error is not None:
        findings.append(Finding(ERROR, "provenance_schema", error))
        provenance = {}
    elif not isinstance(provenance, dict):
        findings.append(Finding(ERROR, "provenance_schema", "/provenance is not a JSON object"))
        provenance = {}
    else:
        for message in validate_provenance(provenance):
            findings.append(Finding(ERROR, "provenance_schema", message))

    if manifest is None or any(f.check == "manifest_schema" for f in findings):
        return None, provenance if isinstance(provenance, dict) else {}
    return manifest, provenance


def _axis_length(spec: dict, axis: str) -> int | None:
    """The declared length of one named axis, if the entry is self-consistent.

    Args:
        spec: A component's manifest entry.
        axis: The axis name.

    Returns:
        The length, or None when the axis is absent or the entry's axes and
        shape disagree in arity.
    """
    if axis not in spec["axes"] or len(spec["axes"]) != len(spec["shape"]):
        return None
    return spec["shape"][spec["axes"].index(axis)]


def _check_component(
    h5: h5py.File,
    spec: dict,
    manifest: dict,
    provenance_records: set[str],
    findings: list[Finding],
) -> None:
    """Validate one component entry against the file.

    Args:
        h5: The open file.
        spec: The component's manifest entry.
        manifest: The whole manifest, for cross-references.
        provenance_records: Keys present in the provenance document.
        findings: Accumulator.
    """
    component_id = spec["id"]
    by_id = {c["id"]: c for c in manifest["components"]}

    node = h5.get(spec["path"])
    if node is None:
        findings.append(
            Finding(
                ERROR, "component_path_exists", f"{component_id}: no dataset at {spec['path']}"
            )
        )
        return
    if not isinstance(node, h5py.Dataset):
        findings.append(
            Finding(
                ERROR,
                "component_path_exists",
                f"{component_id}: {spec['path']} is a {type(node).__name__}, not a dataset",
            )
        )
        return

    if list(node.shape) != list(spec["shape"]) or not _dtype_matches(node, spec["dtype"]):
        findings.append(
            Finding(
                ERROR,
                "dtype_shape_match",
                f"{component_id}: declares {spec['dtype']}{tuple(spec['shape'])} but the "
                f"dataset is {node.dtype.name}{node.shape}",
            )
        )

    # Every check below indexes shape by a position taken from axes, so an
    # arity mismatch must stop this component rather than raise an IndexError
    # out of the function that exists to diagnose it.
    if len(spec["axes"]) != len(spec["shape"]):
        findings.append(
            Finding(
                ERROR,
                "axes_arity",
                f"{component_id}: {len(spec['axes'])} axes for {len(spec['shape'])} dimensions",
            )
        )
        return

    _check_missing_reference(spec, by_id, findings)

    if spec.get("provenance") is not None and spec["provenance"] not in provenance_records:
        findings.append(
            Finding(
                ERROR,
                "provenance_reference",
                f"{component_id}: references undeclared provenance record {spec['provenance']!r}",
            )
        )

    has_sample = "sample" in spec["axes"]
    if has_sample != ("sparse" in spec):
        findings.append(
            Finding(
                ERROR,
                "sample_sparse_pairing",
                f"{component_id}: a sample axis and a sparse declaration must accompany "
                "each other",
            )
        )
    elif has_sample:
        _check_sparse_reference(component_id, spec, by_id, findings)

    if "coord" in spec["axes"] and ("units" not in spec or "coord_order" not in spec):
        findings.append(
            Finding(
                ERROR,
                "coord_declarations",
                f"{component_id}: a coord axis requires units and coord_order",
            )
        )

    frame_axis = _axis_length(spec, "frame")
    declared_frames = manifest["dimensions"]["frame"]
    if frame_axis is not None and frame_axis != declared_frames:
        findings.append(
            Finding(
                ERROR,
                "frame_axis_length",
                f"{component_id}: frame axis is {frame_axis} but dimensions.frame is "
                f"{declared_frames}",
            )
        )

    _check_component_skeleton(component_id, spec, manifest, findings)

    if spec.get("encoding", {}).get("kind") == "rle":
        video = manifest["video"]
        if video.get("width") is None or video.get("height") is None:
            findings.append(
                Finding(
                    ERROR,
                    "rle_needs_dimensions",
                    f"{component_id}: an RLE mask needs video.width and video.height",
                )
            )

    _check_layout(component_id, spec, node, findings)


def _dtype_matches(node: h5py.Dataset, declared: str) -> bool:
    """Whether a dataset's dtype matches a declared one.

    Args:
        node: The payload dataset.
        declared: The manifest's dtype name.

    Returns:
        True when they agree. ``"string"`` matches h5py's variable-length and
        fixed-length string representations, which report as object, bytes or
        unicode dtypes rather than "string".
    """
    if declared == "string":
        return node.dtype.kind in "OSU"
    return node.dtype.name == declared


def _check_missing_reference(spec: dict, by_id: dict[str, dict], findings: list[Finding]) -> None:
    """Validate a mask or length reference, including its shape.

    The ADR requires the reference to resolve "to an existing component with a
    compatible shape". Compatible means aligning with the leading axes of what
    it describes: a per-slot mask over per-keypoint data is meaningful, a mask
    of higher rank than its target is not.

    Args:
        spec: The component's manifest entry.
        by_id: Every component entry, keyed by id.
        findings: Accumulator.
    """
    policy = spec["missing"].get("policy")
    if policy not in ("mask", "length"):
        return
    reference = spec["missing"].get(policy)
    target = by_id.get(reference)
    if target is None:
        findings.append(
            Finding(
                ERROR,
                "mask_reference",
                f"{spec['id']}: missing policy references undeclared {reference!r}",
            )
        )
        return
    own = list(spec["shape"])
    other = list(target["shape"])
    if len(other) > len(own) or other != own[: len(other)]:
        findings.append(
            Finding(
                ERROR,
                "mask_reference",
                f"{spec['id']}: {policy} reference {reference!r} has shape {tuple(other)}, "
                f"which does not align with the leading axes of {tuple(own)}",
            )
        )


def _check_sparse_reference(
    component_id: str, spec: dict, by_id: dict[str, dict], findings: list[Finding]
) -> None:
    """Validate that a sparse index reference is usable.

    Args:
        component_id: The referencing component.
        spec: Its manifest entry.
        by_id: Every component entry, keyed by id.
        findings: Accumulator.
    """
    index_id = spec["sparse"]["index"]
    index_spec = by_id.get(index_id)
    if index_spec is None:
        findings.append(
            Finding(
                ERROR,
                "sparse_index_valid",
                f"{component_id}: sparse index {index_id!r} is not declared",
            )
        )
        return
    if index_spec["axes"] != ["sample"]:
        findings.append(
            Finding(
                ERROR,
                "sparse_index_valid",
                f"{component_id}: sparse index {index_id!r} has axes "
                f"{index_spec['axes']}, expected a one-dimensional sample axis",
            )
        )
        return
    sample_length = _axis_length(spec, "sample")
    if sample_length is not None and sample_length != index_spec["shape"][0]:
        findings.append(
            Finding(
                ERROR,
                "sparse_index_length",
                f"{component_id}: sample axis is {sample_length} but index {index_id!r} has "
                f"{index_spec['shape'][0]} entries",
            )
        )


def _check_sparse_index_values(h5: h5py.File, manifest: dict, findings: list[Finding]) -> None:
    """Validate each sparse index's values, once per index.

    Checked once per index rather than once per referencing component, so a
    shared index does not produce duplicate findings.

    Args:
        h5: The open file.
        manifest: The validated manifest.
        findings: Accumulator.
    """
    by_id = {c["id"]: c for c in manifest["components"]}
    index_ids = {c["sparse"]["index"] for c in manifest["components"] if "sparse" in c}
    frame_count = manifest["video"]["frame_count"]
    for index_id in sorted(index_ids):
        spec = by_id.get(index_id)
        if spec is None or spec["axes"] != ["sample"]:
            continue
        node = h5.get(spec["path"])
        if not isinstance(node, h5py.Dataset):
            continue
        # int64 before differencing: np.diff wraps on unsigned dtypes, so a
        # decreasing uint32 index would otherwise validate clean -- and uint32
        # is exactly what the specification prescribes for a frame index.
        values = np.asarray(node[()]).astype(np.int64, copy=False)
        if values.ndim != 1:
            findings.append(
                Finding(ERROR, "sparse_index_valid", f"{index_id}: index must be one-dimensional")
            )
            continue
        if values.size and not np.all(np.diff(values) > 0):
            findings.append(
                Finding(
                    ERROR,
                    "sparse_index_valid",
                    f"{index_id}: index must be strictly increasing",
                )
            )
        if values.size and (values.min() < 0 or values.max() >= frame_count):
            findings.append(
                Finding(
                    ERROR,
                    "sparse_index_valid",
                    f"{index_id}: index values fall outside [0, {frame_count})",
                )
            )


def _check_component_skeleton(
    component_id: str, spec: dict, manifest: dict, findings: list[Finding]
) -> None:
    """Validate a component's skeleton reference and keypoint axis.

    Args:
        component_id: The component.
        spec: Its manifest entry.
        manifest: The validated manifest.
        findings: Accumulator.
    """
    skeleton_id = spec.get("skeleton")
    if skeleton_id is None:
        return
    skeleton = manifest.get("skeletons", {}).get(skeleton_id)
    if skeleton is None:
        findings.append(
            Finding(
                ERROR,
                "skeleton_reference",
                f"{component_id}: references undeclared skeleton {skeleton_id!r}",
            )
        )
        return
    axis = _axis_length(spec, "keypoint")
    if axis is not None and axis != len(skeleton["body_parts"]):
        findings.append(
            Finding(
                ERROR,
                "keypoint_axis_length",
                f"{component_id}: keypoint axis is {axis} but skeleton {skeleton_id!r} has "
                f"{len(skeleton['body_parts'])} body parts",
            )
        )


def _check_skeletons(manifest: dict, findings: list[Finding]) -> None:
    """Validate every skeleton's edges.

    The ADR's row reads "skeleton references resolve; every edge index <
    len(body_parts)" -- one row, two clauses. The schema cannot express the
    bound, so it has to be checked here or not at all.

    Args:
        manifest: The validated manifest.
        findings: Accumulator.
    """
    for skeleton_id, skeleton in manifest.get("skeletons", {}).items():
        limit = len(skeleton["body_parts"])
        bad = [
            edge
            for edge in skeleton.get("edges", [])
            if any(index >= limit or index < 0 for index in edge)
        ]
        if bad:
            findings.append(
                Finding(
                    ERROR,
                    "skeleton_edge_range",
                    f"skeleton {skeleton_id!r}: edges {bad} name keypoints outside [0, {limit})",
                )
            )


def _check_layout(
    component_id: str, spec: dict, node: h5py.Dataset, findings: list[Finding]
) -> None:
    """Compare a declared layout against the dataset's actual storage.

    Args:
        component_id: The component.
        spec: Its manifest entry.
        node: The payload dataset.
        findings: Accumulator.
    """
    declared = spec.get("layout")
    if declared is None:
        return
    actual_storage = "contiguous" if node.chunks is None else "chunked"
    actual_compression = node.compression or "none"
    if (
        declared.get("storage", actual_storage) != actual_storage
        or declared.get("compression", actual_compression) != actual_compression
    ):
        findings.append(
            Finding(
                WARNING,
                "layout_matches_file",
                f"{component_id}: declares {declared} but the dataset is "
                f"{actual_storage}/{actual_compression}",
            )
        )


def _check_manifest_wide(h5: h5py.File, manifest: dict, findings: list[Finding]) -> None:
    """Validate rules that span the whole manifest.

    Args:
        h5: The open file.
        manifest: The parsed manifest.
        findings: Accumulator.
    """
    declared_format = manifest.get("format")
    declared_revision = manifest.get("schema_revision")
    root_revision = _int_attr(h5.attrs.get("schema_revision"))
    if declared_format != attr_text(h5.attrs.get("jabs_format")) or (
        root_revision is not None and declared_revision != root_revision
    ):
        findings.append(
            Finding(
                ERROR,
                "manifest_matches_root",
                f"manifest declares {declared_format!r}/{declared_revision} but the root "
                f"attributes say {attr_text(h5.attrs.get('jabs_format'))!r}/{root_revision}",
            )
        )

    dimensions = manifest["dimensions"]
    if dimensions["identity"] > dimensions["slot"]:
        findings.append(
            Finding(
                ERROR,
                "identity_le_slot",
                f"dimensions.identity ({dimensions['identity']}) exceeds dimensions.slot "
                f"({dimensions['slot']})",
            )
        )

    frame_count = manifest["video"]["frame_count"]
    if dimensions["frame"] != frame_count:
        findings.append(
            Finding(
                ERROR,
                "dimensions_match_video",
                f"dimensions.frame ({dimensions['frame']}) disagrees with video.frame_count "
                f"({frame_count})",
            )
        )

    ids = [c["id"] for c in manifest["components"]]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        findings.append(
            Finding(ERROR, "component_id_unique", f"duplicate component ids: {duplicates}")
        )

    for component_id in ids:
        segments = component_id.split(".")
        if segments[0] != RESERVED_NAMESPACE and len(segments) < 3:
            findings.append(
                Finding(
                    ERROR,
                    "namespace_well_formed",
                    f"{component_id}: a non-jabs namespace needs a reverse-DNS root of at "
                    "least two segments",
                )
            )

    video = manifest["video"]
    if video.get("width") is None or video.get("height") is None:
        findings.append(
            Finding(
                WARNING,
                "video_dimensions_null",
                "video.width or video.height is unknown, so the coordinate space is not "
                "recorded in the file",
            )
        )

    _check_attachments(h5, manifest, findings)


def _check_attachments(h5: h5py.File, manifest: dict, findings: list[Finding]) -> None:
    """Report payloads under /attachments that the manifest never declared.

    Args:
        h5: The open file.
        manifest: The parsed manifest.
        findings: Accumulator.
    """
    declared_paths = {c["path"] for c in manifest["components"]}
    declared_attachments = {a["path"] for a in manifest.get("attachments", [])}
    for spec in manifest.get("attachments", []):
        if not isinstance(h5.get(spec["path"]), h5py.Dataset):
            findings.append(
                Finding(
                    ERROR,
                    "attachment_path_exists",
                    f"declared attachment {spec['path']} is missing or is not a dataset",
                )
            )

    root = h5.get("attachments")
    if not isinstance(root, h5py.Group):
        # A name test would treat a dataset called "attachments" as a group and
        # then fail inside visititems.
        return

    def note_undeclared(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        path = f"/attachments/{name}"
        if path not in declared_attachments and path not in declared_paths:
            findings.append(
                Finding(
                    WARNING,
                    "attachment_undeclared",
                    f"{path} is present but not listed in the manifest's attachments",
                )
            )

    root.visititems(note_undeclared)


def validate(path: str | Path) -> list[Finding]:
    """Validate a pose file.

    Never raises for a malformed file: every problem comes back as a Finding.

    Args:
        path: The file to validate.

    Returns:
        Findings, errors first. An empty list means the file conforms.
    """
    findings: list[Finding] = []
    with h5py.File(path, "r") as h5:
        if not _check_root(h5, findings):
            return findings
        manifest, provenance = _check_documents(h5, findings)
        if manifest is None:
            return findings

        provenance_records = set(provenance.get("records", {}))
        _check_manifest_wide(h5, manifest, findings)
        _check_skeletons(manifest, findings)
        _check_sparse_index_values(h5, manifest, findings)
        for spec in manifest["components"]:
            _check_component(h5, spec, manifest, provenance_records, findings)

    return sorted(findings, key=lambda f: (f.severity != ERROR, f.check))
