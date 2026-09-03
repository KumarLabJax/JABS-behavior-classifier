"""Validation of a pose file against the specification (ADR 0002).

A valid file is a well-formed manifest: every component is optional, and
consumers declare their own requirements on top. This module implements the
ADR's validation table, and nothing more opinionated than it.

Findings carry a stable ``check`` name so callers and tests can assert on a
specific rule rather than on message text.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.schema import (
    FORMAT_ID,
    SCHEMA_REVISION,
    validate_manifest,
    validate_provenance,
)

ERROR = "error"
WARNING = "warning"

_RESERVED_NAMESPACE = "jabs"


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

    Args:
        h5: The open file.
        findings: Accumulator.

    Returns:
        True when the file identifies as a pose file and can be checked further.
    """
    if h5.attrs.get("jabs_format") != FORMAT_ID:
        findings.append(
            Finding(
                ERROR,
                "root_attrs",
                f"jabs_format is {h5.attrs.get('jabs_format')!r}, expected {FORMAT_ID!r}",
            )
        )
        return False
    if "schema_revision" not in h5.attrs:
        findings.append(Finding(ERROR, "root_attrs", "schema_revision attribute is missing"))
        return False
    revision = int(h5.attrs["schema_revision"])
    if revision > SCHEMA_REVISION:
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


def _check_documents(h5: h5py.File, findings: list[Finding]) -> dict | None:
    """Validate the manifest and provenance documents.

    Args:
        h5: The open file.
        findings: Accumulator.

    Returns:
        The parsed manifest, or None when it is unusable for further checks.
    """
    try:
        manifest = json.loads(h5["manifest"][()])
    except (KeyError, ValueError) as error:
        findings.append(Finding(ERROR, "manifest_schema", f"/manifest is unreadable: {error}"))
        return None
    for message in validate_manifest(manifest):
        findings.append(Finding(ERROR, "manifest_schema", message))

    try:
        provenance = json.loads(h5["provenance"][()])
    except (KeyError, ValueError) as error:
        findings.append(Finding(ERROR, "provenance_schema", f"/provenance is unreadable: {error}"))
    else:
        for message in validate_provenance(provenance):
            findings.append(Finding(ERROR, "provenance_schema", message))

    # Every structural check below assumes a well-formed manifest, so a schema
    # failure stops the pass rather than producing cascading noise.
    if any(f.check == "manifest_schema" for f in findings):
        return None
    return manifest


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
    declared_ids = {c["id"] for c in manifest["components"]}

    if spec["path"] not in h5:
        findings.append(
            Finding(
                ERROR,
                "component_path_exists",
                f"{component_id}: no dataset at {spec['path']}",
            )
        )
        return

    dataset = h5[spec["path"]]
    if list(dataset.shape) != list(spec["shape"]) or dataset.dtype.name != spec["dtype"]:
        findings.append(
            Finding(
                ERROR,
                "dtype_shape_match",
                f"{component_id}: declares {spec['dtype']}{tuple(spec['shape'])} but the "
                f"dataset is {dataset.dtype.name}{dataset.shape}",
            )
        )
    if len(spec["axes"]) != len(spec["shape"]):
        findings.append(
            Finding(
                ERROR,
                "axes_arity",
                f"{component_id}: {len(spec['axes'])} axes for {len(spec['shape'])} dimensions",
            )
        )

    reference = spec["missing"].get("mask") or spec["missing"].get("length")
    if reference is not None and reference not in declared_ids:
        findings.append(
            Finding(
                ERROR,
                "mask_reference",
                f"{component_id}: missing policy references undeclared {reference!r}",
            )
        )

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
        index_id = spec["sparse"]["index"]
        if index_id not in declared_ids:
            findings.append(
                Finding(
                    ERROR,
                    "sparse_index_valid",
                    f"{component_id}: sparse index {index_id!r} is not declared",
                )
            )
        else:
            _check_sparse_index(h5, component_id, index_id, spec, manifest, findings)

    if "coord" in spec["axes"] and ("units" not in spec or "coord_order" not in spec):
        findings.append(
            Finding(
                ERROR,
                "coord_declarations",
                f"{component_id}: a coord axis requires units and coord_order",
            )
        )

    skeleton_id = spec.get("skeleton")
    if skeleton_id is not None:
        skeleton = manifest.get("skeletons", {}).get(skeleton_id)
        if skeleton is None:
            findings.append(
                Finding(
                    ERROR,
                    "skeleton_reference",
                    f"{component_id}: references undeclared skeleton {skeleton_id!r}",
                )
            )
        elif "keypoint" in spec["axes"]:
            axis = spec["shape"][spec["axes"].index("keypoint")]
            if axis != len(skeleton["body_parts"]):
                findings.append(
                    Finding(
                        ERROR,
                        "keypoint_axis_length",
                        f"{component_id}: keypoint axis is {axis} but skeleton "
                        f"{skeleton_id!r} has {len(skeleton['body_parts'])} body parts",
                    )
                )

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

    declared_layout = spec.get("layout")
    if declared_layout is not None:
        actual_storage = "contiguous" if dataset.chunks is None else "chunked"
        actual_compression = dataset.compression or "none"
        if (
            declared_layout.get("storage", actual_storage) != actual_storage
            or declared_layout.get("compression", actual_compression) != actual_compression
        ):
            findings.append(
                Finding(
                    WARNING,
                    "layout_matches_file",
                    f"{component_id}: declares {declared_layout} but the dataset is "
                    f"{actual_storage}/{actual_compression}",
                )
            )


def _check_sparse_index(
    h5: h5py.File,
    component_id: str,
    index_id: str,
    spec: dict,
    manifest: dict,
    findings: list[Finding],
) -> None:
    """Validate a sparse component's frame index.

    Args:
        h5: The open file.
        component_id: The component referencing the index.
        index_id: The index component's id.
        spec: The referencing component's manifest entry.
        manifest: The whole manifest.
        findings: Accumulator.
    """
    index_spec = next(c for c in manifest["components"] if c["id"] == index_id)
    if index_spec["path"] not in h5:
        return
    values = np.asarray(h5[index_spec["path"]][()])
    if values.ndim != 1:
        findings.append(
            Finding(ERROR, "sparse_index_valid", f"{index_id}: index must be one-dimensional")
        )
        return
    if values.size and not np.all(np.diff(values) > 0):
        findings.append(
            Finding(ERROR, "sparse_index_valid", f"{index_id}: index must be strictly increasing")
        )
    frame_count = manifest["video"]["frame_count"]
    if values.size and (values.min() < 0 or values.max() >= frame_count):
        findings.append(
            Finding(
                ERROR,
                "sparse_index_valid",
                f"{index_id}: index values fall outside [0, {frame_count})",
            )
        )
    sample_length = spec["shape"][spec["axes"].index("sample")]
    if values.size != sample_length:
        findings.append(
            Finding(
                ERROR,
                "sparse_index_length",
                f"{component_id}: sample axis is {sample_length} but index {index_id!r} has "
                f"{values.size} entries",
            )
        )


def _check_manifest_wide(h5: h5py.File, manifest: dict, findings: list[Finding]) -> None:
    """Validate rules that span the whole manifest.

    Args:
        h5: The open file.
        manifest: The parsed manifest.
        findings: Accumulator.
    """
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

    ids = [c["id"] for c in manifest["components"]]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        findings.append(
            Finding(ERROR, "component_id_unique", f"duplicate component ids: {duplicates}")
        )

    for component_id in ids:
        segments = component_id.split(".")
        if segments[0] != _RESERVED_NAMESPACE and len(segments) < 3:
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

    declared_paths = {c["path"] for c in manifest["components"]}
    declared_attachments = {a["path"] for a in manifest.get("attachments", [])}
    if "attachments" in h5:

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

        h5["attachments"].visititems(note_undeclared)


def validate(path: str | Path) -> list[Finding]:
    """Validate a pose file.

    Args:
        path: The file to validate.

    Returns:
        Findings, errors first. An empty list means the file conforms.
    """
    findings: list[Finding] = []
    with h5py.File(path, "r") as h5:
        if not _check_root(h5, findings):
            return findings
        manifest = _check_documents(h5, findings)
        if manifest is None:
            return findings

        provenance_records = set(json.loads(h5["provenance"][()]).get("records", {}))
        _check_manifest_wide(h5, manifest, findings)
        for spec in manifest["components"]:
            _check_component(h5, spec, manifest, provenance_records, findings)

    return sorted(findings, key=lambda f: (f.severity != ERROR, f.check))
