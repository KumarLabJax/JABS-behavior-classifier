"""The JABS pose file format, revision 1 (ADR 0002).

A single self-describing HDF5 file per video whose contents are declared in a
JSON manifest rather than implied by a version number. A reader asks what a
file contains, never how old it is.

Typical use::

    from jabs.io.internal.pose_file import read_component, read_manifest, read_pose_file, validate

    findings = validate(path)
    pose = read_pose_file(path)                       # everything
    window = read_component(path, "jabs.pose.points", frames=slice(0, 1800))

    # Reading many windows: read and validate the manifest once.
    manifest = read_manifest(path)
    for start in range(0, frame_count, 1800):
        window = read_component(
            path, "jabs.pose.points", frames=slice(start, start + 1800), manifest=manifest
        )

``dense`` is the only baseline encoding written or read: the specification
makes the baseline encoding the mandatory one and ``ragged`` and ``rle``
optional. ``ragged`` alternates (:class:`Alternate`, :class:`RaggedIndex`) are
written and read, but not decoded in Python. A :class:`BlockSource` lets a
writer stream a component's data to disk a block at a time.
"""

from jabs.io.internal.pose_file.adapter import PoseFileHDF5Adapter
from jabs.io.internal.pose_file.alternates import Alternate, RaggedIndex
from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.manifest import (
    ParsedManifest,
    build_manifest,
    build_provenance,
    parse_manifest,
    parse_provenance,
)
from jabs.io.internal.pose_file.reader import (
    NotAPoseFileError,
    PoseFileError,
    read_component,
    read_manifest,
    read_pose_file,
)
from jabs.io.internal.pose_file.schema import (
    FORMAT_ID,
    MANIFEST_SCHEMA,
    PROVENANCE_SCHEMA,
    SCHEMA_REVISION,
    validate_manifest,
    validate_provenance,
)
from jabs.io.internal.pose_file.types import (
    Attachment,
    Component,
    HistoryEntry,
    PoseFile,
    Provenance,
    ProvenanceRecord,
    Skeleton,
    VideoInfo,
)
from jabs.io.internal.pose_file.validate import Finding, validate
from jabs.io.internal.pose_file.writer import write_pose_file

__all__ = [
    "FORMAT_ID",
    "MANIFEST_SCHEMA",
    "PROVENANCE_SCHEMA",
    "SCHEMA_REVISION",
    "Alternate",
    "Attachment",
    "BlockSource",
    "Component",
    "Finding",
    "HistoryEntry",
    "NotAPoseFileError",
    "ParsedManifest",
    "PoseFile",
    "PoseFileError",
    "PoseFileHDF5Adapter",
    "Provenance",
    "ProvenanceRecord",
    "RaggedIndex",
    "Skeleton",
    "VideoInfo",
    "build_manifest",
    "build_provenance",
    "parse_manifest",
    "parse_provenance",
    "read_component",
    "read_manifest",
    "read_pose_file",
    "validate",
    "validate_manifest",
    "validate_provenance",
    "write_pose_file",
]
