"""Writing revision-1 pose files (ADR 0002).

The writer is **atomic**: it builds and serializes both JSON documents, then
writes a temporary file in the destination's directory and moves it into place.
``h5py.File(path, "w")`` truncates at open, so anything that can fail after
that open — an unserializable provenance value, a component path collision, a
full disk — would otherwise cost the caller the file they already had.
"""

import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.alternates import Alternate, RaggedIndex, alternate_key
from jabs.io.internal.pose_file.blocks import BlockSource, ComponentData
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

# A leaf-level ragged payload is chunked along its single item axis at this many
# rows (spec 0029-A §2.6): large enough that a whole recording's points are a
# few hundred chunks, small enough that one frame's points cost one chunk read.
# Byte shuffling groups the coordinates' high bytes, which compress far better
# than the interleaved values.
_RAGGED_CHUNK = 65536


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


def _ragged_chunks(alternate: Alternate) -> tuple[int, ...] | None:
    """The chunk shape of an alternate's payload, or None when it is contiguous.

    A group-level payload is one small value per group, read with its offsets,
    so it is contiguous like them. An empty leaf payload cannot be chunked.
    """
    shape = alternate.data.shape
    if alternate.level != "leaf" or shape[0] == 0:
        return None
    return (min(_RAGGED_CHUNK, shape[0]), *shape[1:])


def _alternate_layout(alternate: Alternate) -> dict:
    """The storage layout this writer will apply to an alternate's payload.

    Args:
        alternate: The alternate about to be written.

    Returns:
        The layout to record in the manifest.
    """
    chunks = _ragged_chunks(alternate)
    if chunks is None:
        return {"storage": "contiguous", "compression": "none"}
    return {
        "storage": "chunked",
        "chunks": list(chunks),
        "compression": _COMPRESSION,
        "compression_opts": _COMPRESSION_OPTS,
    }


def _file_paths(pose_file: PoseFile) -> list[tuple[str, str, bool]]:
    """Every dataset path the writer will create, its owner, and whether it holds offsets.

    Two alternates sharing one :class:`RaggedIndex` share its two datasets, so
    those paths are listed once; two different indexes at one path are two
    claims on it.
    """
    paths = [(component.path, component.id, False) for component in pose_file.components]
    listed: set[int] = set()
    for component in pose_file.components:
        for alternate in component.alternates:
            label = alternate_key(component.id, alternate.name)
            paths.append((alternate.path, label, False))
            index = alternate.index
            if id(index) not in listed:
                listed.add(id(index))
                paths.append((index.group_offsets_path, f"{label}'s group_offsets", True))
                paths.append((index.instance_offsets_path, f"{label}'s instance_offsets", True))
    return paths


def _check_path_collisions(pose_file: PoseFile) -> None:
    """Refuse datasets whose HDF5 paths cannot coexist.

    Covers every dataset the writer creates -- component payloads, alternate
    payloads and offsets. Two paths may be individually valid and still be
    equal, or name paths where one is a parent of the other, which HDF5 cannot
    represent.

    Args:
        pose_file: The file's contents.

    Raises:
        ValueError: If two paths collide, or two different
            :class:`RaggedIndex` objects declare offsets at one path.
    """
    owners: dict[str, str] = {}
    offsets_paths: set[str] = set()
    for path, owner, is_offsets in _file_paths(pose_file):
        other = owners.get(path)
        if other is not None and is_offsets and path in offsets_paths:
            # Each index is listed once, so this is a second index at the same
            # offsets: one array declared twice, not two arrays.
            raise ValueError(
                f"{owner} and {other} are both at {path} through different RaggedIndex "
                "objects; alternates using the same offsets must share one RaggedIndex object"
            )
        if other is not None:
            raise ValueError(
                f"paths collide: {owner} and {other} are both at {path}; HDF5 cannot hold both"
            )
        owners[path] = owner
        if is_offsets:
            offsets_paths.add(path)
    for path, owner in owners.items():
        # Every proper ancestor, so "/a" is found above "/a/b" however the
        # paths sort ("/a-c" sorts between them).
        parts = path.split("/")
        for depth in range(2, len(parts)):
            parent = "/".join(parts[:depth])
            if parent in owners:
                raise ValueError(
                    f"paths collide: {owners[parent]} at {parent} is a parent of {owner} at "
                    f"{path}; HDF5 cannot hold both"
                )


def _check_encodings(pose_file: PoseFile) -> None:
    """Refuse any baseline encoding this writer cannot actually produce.

    Silently re-labeling a ragged or RLE payload as dense would corrupt it
    beyond recovery while the bytes are still run values or offsets. Only
    baselines are checked: an alternate is always ``ragged``, and the writer
    writes it as such.

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


def _as_text(value: object) -> str:
    """Decode one element of a string payload.

    Args:
        value: A str, bytes, or anything else.

    Returns:
        The text. Bytes are decoded rather than repr'd, so ``b"a"`` becomes
        ``"a"`` and not ``"b'a'"``.
    """
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_blocks(dataset: h5py.Dataset, source: BlockSource, label: str) -> None:
    """Write a block source's rows into a dataset created at its shape and dtype.

    The blocks must tile the first axis exactly, in order: anything else would
    leave rows holding the dataset's fill value, indistinguishable from data.
    A block's dtype must match the declaration exactly, because h5py would
    otherwise cast it silently -- float64 into float32 loses precision, and
    floats into integers truncate.

    Args:
        dataset: The dataset to fill, created at ``source.shape`` and
            ``source.dtype``.
        source: The rows to write.
        label: Names the payload in error messages, such as a component id.

    Raises:
        ValueError: If a block has the wrong number of dimensions, dtype or
            trailing shape, starts anywhere but the end of the previous block,
            or runs past the declared length, or if the blocks stop short of it.
    """
    expected_rows = source.shape[0]
    trailing = source.shape[1:]
    written = 0
    for start, block in source.blocks():
        # First: a 0-d block has no rows to count, and against a 1-D source
        # its empty trailing shape would pass the check below.
        if block.ndim != source.ndim:
            raise ValueError(
                f"{label}: a block has {block.ndim} dimensions {block.shape}, but the source "
                f"declares {source.ndim} {source.shape}"
            )
        if block.dtype != source.dtype:
            raise ValueError(
                f"{label}: a block has dtype {block.dtype.str!r}, but the source declares "
                f"{source.dtype.str!r}; blocks are not cast"
            )
        if block.shape[1:] != trailing:
            raise ValueError(
                f"{label}: a block has trailing shape {block.shape[1:]}, but the source "
                f"declares {trailing}"
            )
        if start != written:
            raise ValueError(
                f"{label}: a block starts at row {start}, but the next unwritten row is "
                f"{written}; blocks must tile the first axis in order"
            )
        end = written + block.shape[0]
        if end > expected_rows:
            raise ValueError(
                f"{label}: a block ends at row {end}, past the declared {expected_rows} rows"
            )
        dataset[written:end] = block
        written = end
    if written != expected_rows:
        raise ValueError(
            f"{label}: the blocks stopped at row {written} of the declared {expected_rows}"
        )


def _create_dataset(h5: h5py.File, component: Component) -> None:
    """Create one component's dataset with this writer's layout policy.

    Args:
        h5: The open destination file.
        component: The component to write.
    """
    data = component.data
    options: dict = {}
    if component.dtype == "string":
        # Variable-length UTF-8, so a reader on any HDF5 implementation gets
        # text rather than this machine's fixed-width padding. Converted
        # elementwise with the shape preserved: `[str(v) for v in arr.tolist()]`
        # stringifies whole rows of a 2-D array into one element each, and
        # turns b"a" into the literal text "b'a'".
        data = np.asarray(
            [_as_text(value) for value in component.data.ravel().tolist()], dtype=object
        ).reshape(component.data.shape)
        options["dtype"] = h5py.string_dtype(encoding="utf-8")

    source = None
    if isinstance(data, BlockSource):
        # Create the dataset empty at the declared shape and fill it a block at
        # a time, so the whole array is never in memory. The layout is the same
        # as for an in-memory array: it depends only on the id and the shape.
        source, data = data, None
        options.update(shape=source.shape, dtype=source.dtype)

    chunks = None if _is_keypoint_scale(component.id) else _frame_major_chunks(component)
    if chunks is None:
        dataset = h5.create_dataset(component.path, data=data, chunks=None, **options)
    else:
        dataset = h5.create_dataset(
            component.path,
            data=data,
            chunks=chunks,
            compression=_COMPRESSION,
            compression_opts=_COMPRESSION_OPTS,
            **options,
        )
    if source is not None:
        _write_blocks(dataset, source, component.id)


def _write_array(
    h5: h5py.File, path: str, data: ComponentData, label: str, **options: object
) -> None:
    """Create one dataset from an in-memory array or a block source.

    Args:
        h5: The open destination file.
        path: Where the dataset goes.
        data: Its contents.
        label: Names the payload in error messages.
        **options: Storage options for ``create_dataset``.
    """
    if isinstance(data, BlockSource):
        dataset = h5.create_dataset(path, shape=data.shape, dtype=data.dtype, **options)
        _write_blocks(dataset, data, label)
    else:
        h5.create_dataset(path, data=data, **options)


def _write_ragged_set(
    h5: h5py.File, members: list[tuple[str, Alternate]], written_indexes: set[int]
) -> None:
    """Write one alternate name's offsets and payloads, back to back.

    Written as one run, so the set occupies one byte range (spec 0029-A §2.6):
    a reader fetching one frame's segmentation reads the offsets and then the
    payload nearby, rather than seeking across the dense baseline. The order
    is the order a reader needs them -- offsets, then group-level payloads,
    then the leaf payloads they index.

    Args:
        h5: The open destination file.
        members: Every ``(component id, alternate)`` of one name, in component
            order. They share one :class:`RaggedIndex`.
        written_indexes: Ids of the indexes whose offsets are already in the
            file; updated here. Two names may share one index, and its offsets
            are one pair of datasets.
    """
    index: RaggedIndex = members[0][1].index
    name = members[0][1].name
    if id(index) not in written_indexes:
        written_indexes.add(id(index))
        for offsets_name in ("group_offsets", "instance_offsets"):
            _write_array(
                h5,
                getattr(index, f"{offsets_name}_path"),
                getattr(index, offsets_name),
                f"alternate {name!r} {offsets_name}",
                chunks=None,
            )
    for level in ("group", "leaf"):
        for component_id, alternate in members:
            if alternate.level != level:
                continue
            chunks = _ragged_chunks(alternate)
            options: dict = {"chunks": chunks}
            if chunks is not None:
                options.update(
                    compression=_COMPRESSION, compression_opts=_COMPRESSION_OPTS, shuffle=True
                )
            _write_array(
                h5,
                alternate.path,
                alternate.data,
                alternate_key(component_id, alternate.name),
                **options,
            )


def write_pose_file(
    pose_file: PoseFile, path: str | Path, created: str | None = None, *, deep: bool = False
) -> None:
    """Write a pose file, atomically.

    Both JSON documents are built, validated **and serialized** before any file
    is opened, and the payloads are written to a temporary file in the
    destination's directory which is then moved into place. A failure at any
    point leaves an existing destination untouched.

    Datasets are written in component order. Each named ragged set is written
    immediately before the first component that carries an alternate of that
    name: its offsets (unless an earlier set sharing its index already wrote
    them), then its group-level payloads, then its leaf-level payloads. A
    caller that wants spec 0029-A's layout, with the ragged set after the
    keypoint-scale data, lists the keypoint-scale components before the
    segmentation ones.

    Args:
        pose_file: The file's contents.
        path: Destination path.
        created: The manifest's creation timestamp. Defaults to now; pass a
            fixed value to make output byte-reproducible, as the conformance
            fixtures need.
        deep: Also run the validator's deep check on the temporary file, before
            it replaces the destination: decode every alternate and compare it
            with its baseline, so a stale alternate never reaches the
            destination. It reads the baseline in full, so it costs as much as
            reading the payload; a tool that writes alternates should pass it.
            An alternate the deep check cannot decode is refused too, though
            ``validate`` reports it only as a warning: publishing it would
            break the promise that every alternate was compared. Without
            ``deep``, an alternate whose values disagree with its baseline
            is not caught.

    Raises:
        ValueError: If the manifest or provenance document would be invalid, if
            two dataset paths collide, if two different :class:`RaggedIndex`
            objects declare offsets at one path, if a block source's blocks do
            not match its declared shape and dtype, or if the finished file
            fails validation (with ``deep``, including an alternate that
            disagrees with its baseline or that the deep check cannot decode).
        TypeError: If a value in the manifest or provenance cannot be
            serialized to JSON.
    """
    destination = Path(path)
    _check_encodings(pose_file)
    _check_path_collisions(pose_file)

    layouts = {component.id: _layout_for(component) for component in pose_file.components}
    # Every alternate of a name, in component order: the set written together.
    ragged_sets: dict[str, list[tuple[str, Alternate]]] = {}
    for component in pose_file.components:
        for alternate in component.alternates:
            layouts[alternate_key(component.id, alternate.name)] = _alternate_layout(alternate)
            ragged_sets.setdefault(alternate.name, []).append((component.id, alternate))
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
            written_sets: set[str] = set()
            written_indexes: set[int] = set()
            for component in pose_file.components:
                # Each ragged set goes immediately before the first component
                # that carries it, so it sits beside its baseline on disk.
                for alternate in component.alternates:
                    if alternate.name not in written_sets:
                        _write_ragged_set(h5, ragged_sets[alternate.name], written_indexes)
                        written_sets.add(alternate.name)
                _create_dataset(h5, component)
            for attachment in pose_file.attachments:
                # Carried verbatim: an attachment declares no axes, so no tool
                # can transform it correctly, and dropping one silently is a
                # specification violation.
                h5.create_dataset(attachment.path, data=attachment.data)
        # The two documents were validated before the open, but file-level
        # invariants -- offset integrity, sparse index ordering, layout
        # agreement -- can only be checked against the finished file. Checking
        # here is what makes "the writer never publishes a file its own
        # validator rejects" true rather than aspirational, and it is cheap:
        # validate() reads indexes and offsets, never the payloads -- unless
        # the caller asks for the deep check, which decodes every alternate
        # against its baseline and so reads both in full.
        from jabs.io.internal.pose_file.validate import validate

        # Under deep, an undecodable alternate is a refusal: the caller asked
        # for every alternate to be compared, and this one was not.
        errors = [
            f
            for f in validate(temporary, deep=deep)
            if f.severity == "error" or (deep and f.check == "alternate_deep_unchecked")
        ]
        if errors:
            raise ValueError(
                "refusing to write an invalid pose file: "
                + "; ".join(f"{f.check}: {f.message}" for f in errors)
            )
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
