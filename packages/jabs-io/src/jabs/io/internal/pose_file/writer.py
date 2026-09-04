"""Writing revision-1 pose files (ADR 0002).

The writer is **atomic**: it builds and serializes both JSON documents, then
writes a temporary file in the destination's directory and moves it into place.
``h5py.File(path, "w")`` truncates at open, so anything that can fail after
that open — an unserializable provenance value, a component path collision, a
full disk — would otherwise cost the caller the file they already had.
"""

import itertools
import json
import os
import tempfile
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

# Only the baseline encoding is implemented. The specification permits ragged
# and RLE, and a reader must implement the baseline and may implement others --
# so writing a ragged payload as though it were dense is not an option.
_SUPPORTED_ENCODINGS = frozenset({"dense"})

# Keypoint-scale components are stored contiguous and uncompressed, because
# contiguous storage is the only layout under which a frame range really is one
# byte range -- the access pattern an in-browser overlay depends on. Chunking
# would only bound read amplification, since HDF5 chunks need not be adjacent
# or ordered on disk. These arrays are at most tens of MB, so compression buys
# little and costs the access pattern.
_CONTIGUOUS_ID_PREFIXES = ("jabs.pose.", "jabs.identity.", "jabs.time.")

_COMPRESSION = "gzip"
# Level 1 rather than 6: measured on segmentation-shaped data, level 6 costs
# several times the write CPU for a few percent less disk.
_COMPRESSION_OPTS = 1

# Target chunk size for compressed components, in frames. Chunks are shaped
# along the frame axis and span every other axis whole, so a frame window maps
# onto whole chunks. Left to h5py, `chunks=True` picks a shape that splits the
# trailing axes and multiplies the reads a one-frame lookup costs.
_FRAME_CHUNK = 300


def _is_keypoint_scale(component_id: str) -> bool:
    """Whether a component is stored contiguous and uncompressed.

    Args:
        component_id: The component's namespaced id.

    Returns:
        True when the component should use contiguous storage.
    """
    return component_id.startswith(_CONTIGUOUS_ID_PREFIXES)


def _frame_major_chunks(component: Component) -> tuple[int, ...] | None:
    """A chunk shape that keeps a frame window contiguous within chunks.

    Args:
        component: The component about to be written.

    Returns:
        The chunk shape, or None when the payload is empty and cannot be
        chunked.
    """
    shape = component.data.shape
    if not shape or any(dimension == 0 for dimension in shape):
        return None
    leading = min(_FRAME_CHUNK, shape[0])
    return (leading, *shape[1:])


def _layout_for(component: Component) -> dict:
    """The storage layout this writer will apply to a component.

    Args:
        component: The component about to be written.

    Returns:
        The layout to record in the manifest.
    """
    if _is_keypoint_scale(component.id):
        return {"storage": "contiguous", "compression": "none"}
    chunks = _frame_major_chunks(component)
    if chunks is None:
        return {"storage": "contiguous", "compression": "none"}
    return {
        "storage": "chunked",
        "chunks": list(chunks),
        "compression": _COMPRESSION,
        "compression_opts": _COMPRESSION_OPTS,
    }


def _check_path_collisions(pose_file: PoseFile) -> None:
    """Refuse components whose HDF5 paths cannot coexist.

    Two ids may be individually valid and still name paths where one is a
    parent of the other, which HDF5 cannot represent.

    Args:
        pose_file: The file's contents.

    Raises:
        ValueError: If two component paths collide.
    """
    paths = sorted((component.path, component.id) for component in pose_file.components)
    for (earlier, earlier_id), (later, later_id) in itertools.pairwise(paths):
        if later == earlier or later.startswith(earlier + "/"):
            raise ValueError(
                f"component paths collide: {earlier_id} at {earlier} is a parent of "
                f"{later_id} at {later}; HDF5 cannot hold both"
            )


def _check_encodings(pose_file: PoseFile) -> None:
    """Refuse any encoding this writer cannot actually produce.

    Silently re-labeling a ragged or RLE payload as dense would corrupt it
    beyond recovery while the bytes are still run values or offsets.

    Args:
        pose_file: The file's contents.

    Raises:
        NotImplementedError: If a component declares an unsupported encoding.
    """
    for component in pose_file.components:
        kind = component.encoding.get("kind")
        if kind not in _SUPPORTED_ENCODINGS:
            raise NotImplementedError(
                f"{component.id}: cannot write encoding {kind!r}; only "
                f"{sorted(_SUPPORTED_ENCODINGS)} is implemented, and relabeling the payload "
                "as dense would corrupt it"
            )


def _create_dataset(h5: h5py.File, component: Component) -> None:
    """Create one component's dataset with this writer's layout policy.

    Args:
        h5: The open destination file.
        component: The component to write.
    """
    if component.dtype == "string":
        # Variable-length UTF-8, so a reader on any HDF5 implementation gets
        # text rather than this machine's fixed-width padding.
        h5.create_dataset(
            component.path,
            data=[str(value) for value in component.data.tolist()],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        return
    if _is_keypoint_scale(component.id):
        h5.create_dataset(component.path, data=component.data, chunks=None)
        return
    chunks = _frame_major_chunks(component)
    if chunks is None:
        h5.create_dataset(component.path, data=component.data, chunks=None)
        return
    h5.create_dataset(
        component.path,
        data=component.data,
        chunks=chunks,
        compression=_COMPRESSION,
        compression_opts=_COMPRESSION_OPTS,
    )


def write_pose_file(pose_file: PoseFile, path: str | Path, created: str | None = None) -> None:
    """Write a pose file, atomically.

    Both JSON documents are built, validated **and serialized** before any file
    is opened, and the payloads are written to a temporary file in the
    destination's directory which is then moved into place. A failure at any
    point leaves an existing destination untouched.

    Args:
        pose_file: The file's contents.
        path: Destination path.
        created: The manifest's creation timestamp. Defaults to now; pass a
            fixed value to make output byte-reproducible, as the conformance
            fixtures need.

    Raises:
        ValueError: If the manifest or provenance document would be invalid, or
            if two component paths collide.
        TypeError: If a value in the manifest or provenance cannot be
            serialized to JSON.
    """
    destination = Path(path)
    _check_encodings(pose_file)
    _check_path_collisions(pose_file)

    layouts = {component.id: _layout_for(component) for component in pose_file.components}
    manifest = build_manifest(pose_file, layouts=layouts, created=created)
    provenance = build_provenance(pose_file.provenance)
    errors = validate_manifest(manifest) + validate_provenance(provenance)
    if errors:
        raise ValueError("refusing to write an invalid pose file: " + "; ".join(errors))

    # Serialize before opening anything: a value the schema admits can still be
    # unserializable (numpy integer scalars, for one), and discovering that
    # after the destination has been truncated destroys the caller's file.
    manifest_json = json.dumps(manifest)
    provenance_json = json.dumps(provenance)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    handle, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(handle)
    try:
        with h5py.File(temporary, "w") as h5:
            h5.attrs["jabs_format"] = FORMAT_ID
            h5.attrs["schema_revision"] = np.int32(SCHEMA_REVISION)
            h5.create_dataset("manifest", data=manifest_json, dtype=string_dtype)
            h5.create_dataset("provenance", data=provenance_json, dtype=string_dtype)
            for component in pose_file.components:
                _create_dataset(h5, component)
            for attachment in pose_file.attachments:
                # Carried verbatim: an attachment declares no axes, so no tool
                # can transform it correctly, and dropping one silently is a
                # specification violation.
                h5.create_dataset(attachment.path, data=attachment.data)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
