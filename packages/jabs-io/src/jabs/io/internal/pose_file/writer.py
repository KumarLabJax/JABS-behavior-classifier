"""Writing revision-1 pose files (ADR 0002)."""

import json
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.manifest import build_manifest, build_provenance
from jabs.io.internal.pose_file.schema import (
    FORMAT_ID,
    SCHEMA_REVISION,
    validate_manifest,
    validate_provenance,
)
from jabs.io.internal.pose_file.types import Component, PoseFile

# Keypoint-scale components are stored contiguous and uncompressed, because
# contiguous storage is the only layout under which a frame range really is one
# byte range -- the access pattern an in-browser overlay depends on. Chunking
# would only bound read amplification, since HDF5 chunks need not be adjacent
# or ordered on disk. These arrays are at most tens of MB, so compression buys
# little and costs the access pattern.
_CONTIGUOUS_ID_PREFIXES = ("jabs.pose.", "jabs.identity.", "jabs.time.")

_COMPRESSION = "gzip"
_COMPRESSION_OPTS = 6


def _is_keypoint_scale(component_id: str) -> bool:
    """Whether a component is stored contiguous and uncompressed.

    Args:
        component_id: The component's namespaced id.

    Returns:
        True when the component should use contiguous storage.
    """
    return component_id.startswith(_CONTIGUOUS_ID_PREFIXES)


def _layout_for(component: Component) -> dict:
    """The storage layout this writer will apply to a component.

    Args:
        component: The component about to be written.

    Returns:
        The layout to record in the manifest.
    """
    if _is_keypoint_scale(component.id):
        return {"storage": "contiguous", "compression": "none"}
    return {
        "storage": "chunked",
        "compression": _COMPRESSION,
        "compression_opts": _COMPRESSION_OPTS,
    }


def write_pose_file(pose_file: PoseFile, path: str | Path) -> None:
    """Write a pose file.

    The manifest and provenance documents are built and validated before the
    file is opened, because ``h5py.File(path, "w")`` truncates at open time and
    a validation failure must not destroy an existing file.

    Args:
        pose_file: The file's contents.
        path: Destination path.

    Raises:
        ValueError: If the manifest or provenance document would be invalid.
    """
    layouts = {component.id: _layout_for(component) for component in pose_file.components}
    manifest = build_manifest(pose_file, layouts=layouts)
    provenance = build_provenance(pose_file.provenance)
    errors = validate_manifest(manifest) + validate_provenance(provenance)
    if errors:
        raise ValueError("refusing to write an invalid pose file: " + "; ".join(errors))

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["jabs_format"] = FORMAT_ID
        h5.attrs["schema_revision"] = np.int32(SCHEMA_REVISION)
        h5.create_dataset("manifest", data=json.dumps(manifest), dtype=string_dtype)
        h5.create_dataset("provenance", data=json.dumps(provenance), dtype=string_dtype)
        for component in pose_file.components:
            if _is_keypoint_scale(component.id):
                h5.create_dataset(component.path, data=component.data, chunks=None)
            else:
                h5.create_dataset(
                    component.path,
                    data=component.data,
                    chunks=True,
                    compression=_COMPRESSION,
                    compression_opts=_COMPRESSION_OPTS,
                )
