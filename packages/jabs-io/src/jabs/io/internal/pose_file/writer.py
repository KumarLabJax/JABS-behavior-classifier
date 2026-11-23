"""Writing revision-1 pose files (ADR 0002).

The writer is **atomic**: it builds and serializes both JSON documents, then
writes a temporary file in the destination's directory and moves it into place.
``h5py.File(path, "w")`` truncates at open, so anything that can fail after
that open — an unserializable provenance value, a component path collision, a
full disk — would otherwise cost the caller the file they already had.
"""

import json
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.alternates import Alternate, RaggedIndex
from jabs.io.internal.pose_file.blocks import BlockSource, ComponentData
from jabs.io.internal.pose_file.embeddings import is_embedding_id
from jabs.io.internal.pose_file.manifest import build_manifest, build_provenance
from jabs.io.internal.pose_file.schema import (
    FORMAT_ID,
    SCHEMA_REVISION,
    validate_manifest,
    validate_provenance,
)
from jabs.io.internal.pose_file.types import Component, PoseFile, alternate_key

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
# Level 4. Segmentation-scale arrays are mostly padding, and deflate at level 1
# reaches only about 200:1 on long runs of one value where level 4 approaches
# deflate's ~1032:1 ceiling, so on padding-heavy files level 4 stores several
# times less for about twice the compression CPU. Level 6 adds a few percent.
COMPRESSION_LEVEL = 4

# The target size of one chunk of a chunked component, uncompressed. Chunks
# are cut along every axis but the first and one item axis (see
# `chunk_shape`), so this bounds what a read decompresses per chunk and keeps
# every chunk far below HDF5 1.x's 4 GiB chunk limit, whatever the padded
# extents of the other axes.
CHUNK_BYTES = 256 * 2**10

# Whether a compressed component is byte-shuffled first. Shuffling groups the
# values' high bytes, which compress far better than interleaved values. HDF5
# applies the filter on read, so a reader needs nothing new; the layout schema
# has no field for it, so it is not recorded.
_COMPONENT_SHUFFLE = True

# A leaf-level ragged payload is chunked along its single item axis at this many
# rows (spec 0029-A §2.6): large enough that a whole recording's points are a
# few hundred chunks, small enough that one frame's points cost one chunk read.
_RAGGED_CHUNK = 65536

# A ragged payload holds no padding, so the level that suits padding buys it
# little; it keeps level 1, which spec 0029-A's browser reads were measured on.
_RAGGED_COMPRESSION_LEVEL = 1

# Whether a compressed ragged payload is byte-shuffled first. Shuffling groups
# the coordinates' high bytes, which compress far better than the interleaved
# values.
_RAGGED_SHUFFLE = True

# Target chunk size for an embedding, in bytes. Embeddings are read whole rows
# at a time and their float payloads compress poorly, so they are stored
# uncompressed, chunked along frame with every other axis whole, at about one
# MiB a chunk: few enough chunks for a long recording, small enough that a
# window read does not pull in much more than it asked for.
_EMBEDDING_CHUNK_BYTES = 1_048_576

_CONTIGUOUS = {"storage": "contiguous", "compression": "none"}


def _is_keypoint_scale(component_id: str) -> bool:
    """Whether a component is stored contiguous and uncompressed.

    Args:
        component_id: The component's namespaced id.

    Returns:
        True when the component should use contiguous storage.
    """
    return component_id.startswith(_CONTIGUOUS_ID_PREFIXES)


def _ragged_layout(shape: tuple[int, ...]) -> dict:
    """A ragged payload's chunked, compressed layout along its item axis, or contiguous when empty.

    Chunks span every trailing axis whole and at most ``_RAGGED_CHUNK`` rows
    of the first, so a window along that axis maps onto whole chunks. A 0-d
    payload or one with a zero-length axis cannot be chunked -- every chunk
    dimension must be positive -- and holds nothing to compress, so it is
    stored contiguous.

    Args:
        shape: The payload's shape.

    Returns:
        The layout to record in the manifest.
    """
    if not shape or 0 in shape:
        return {"storage": "contiguous", "compression": "none"}
    return {
        "storage": "chunked",
        "chunks": [min(_RAGGED_CHUNK, shape[0]), *shape[1:]],
        "compression": _COMPRESSION,
        "compression_opts": _RAGGED_COMPRESSION_LEVEL,
    }


def _item_axis(axes: Sequence[str]) -> int:
    """The axis a chunk keeps whole: the innermost after the first not named ``coord``.

    For dense contours that is ``point``, so a chunk is one contour's x or y
    values over consecutive frames. When every trailing axis is ``coord`` the
    last one is kept.

    Args:
        axes: The component's axis names.

    Returns:
        The axis index; 0 for a one-dimensional component.
    """
    for axis in range(len(axes) - 1, 0, -1):
        if axes[axis] != "coord":
            return axis
    return len(axes) - 1


def chunk_shape(axes: Sequence[str], shape: Sequence[int], itemsize: int) -> tuple[int, ...]:
    """The chunk shape this writer gives a chunked component.

    A chunk is 1 along every axis but the first and the item axis
    (:func:`_item_axis`), holds the item axis whole, and as many rows of the
    first as fit in ``CHUNK_BYTES``. So a chunk holds one run of related
    values -- one contour's x or y coordinates -- over consecutive frames:
    values that change little from frame to frame and compress far better
    than whole padded frames do. An item axis longer than the budget is cut
    to it. The first axis's extent depends only on the item axis, not on the
    slot or contour count, so a caller can size frame blocks to it before it
    knows those.

    Args:
        axes: The component's axis names.
        shape: Its shape; every extent positive.
        itemsize: Bytes per value.

    Returns:
        The chunk shape.
    """
    chunks = [1] * len(shape)
    budget = max(1, CHUNK_BYTES // itemsize)
    item = _item_axis(axes)
    if item > 0:
        chunks[item] = min(shape[item], budget)
        budget = max(1, budget // chunks[item])
    chunks[0] = min(shape[0], budget)
    return tuple(chunks)


def _embedding_chunks(component: Component) -> tuple[int, ...]:
    """A frame-major chunk shape of about ``_EMBEDDING_CHUNK_BYTES``.

    Args:
        component: A non-empty embedding component.

    Returns:
        The chunk shape: at least one frame, every other axis whole.
    """
    shape = component.data.shape
    bytes_per_frame = math.prod(shape[1:]) * component.data.dtype.itemsize
    frames = max(1, min(shape[0], _EMBEDDING_CHUNK_BYTES // bytes_per_frame))
    return (frames, *shape[1:])


def _honored_request(component: Component) -> dict:
    """The layout a component requested, with chunk extents clamped to its shape.

    ``Component`` has already validated the request, so only the clamp is left.

    Args:
        component: A non-empty component carrying a ``requested_layout``.

    Returns:
        The layout to apply and record; ``compression_opts`` appears only for
        gzip.
    """
    request = component.requested_layout
    compression = request.get("compression", "none")
    if request["storage"] == "contiguous":
        return dict(_CONTIGUOUS)
    layout = {
        "storage": "chunked",
        "chunks": [
            min(extent, max(1, length))
            for extent, length in zip(request["chunks"], component.data.shape, strict=True)
        ],
        "compression": compression,
    }
    if compression == "gzip":
        layout["compression_opts"] = request["compression_opts"]
    return layout


def _layout_for(component: Component) -> dict:
    """The storage layout this writer will apply to a component.

    The single source of truth: ``_create_dataset`` creates exactly what this
    returns, and the manifest records it. In order: an empty payload cannot be
    chunked; a request is honored; an embedding gets its uncompressed default;
    keypoint-scale components are contiguous; everything else is chunked by
    :func:`chunk_shape`, gzip ``COMPRESSION_LEVEL``.

    Args:
        component: The component about to be written.

    Returns:
        The layout to record in the manifest.
    """
    shape = component.data.shape
    if not shape or any(dimension == 0 for dimension in shape):
        return dict(_CONTIGUOUS)
    if component.requested_layout is not None:
        return _honored_request(component)
    if is_embedding_id(component.id):
        return {
            "storage": "chunked",
            "chunks": list(_embedding_chunks(component)),
            "compression": "none",
        }
    if _is_keypoint_scale(component.id):
        return dict(_CONTIGUOUS)
    chunks = chunk_shape(component.axes, shape, component.data.dtype.itemsize)
    return {
        "storage": "chunked",
        "chunks": list(chunks),
        "compression": _COMPRESSION,
        "compression_opts": COMPRESSION_LEVEL,
    }


def _alternate_layout(alternate: Alternate) -> dict:
    """The storage layout this writer will apply to an alternate's payload.

    A group-level payload is one small value per group, read with its
    offsets, so it is contiguous like them. A leaf-level payload is chunked
    along its item axis.

    Args:
        alternate: The alternate about to be written.

    Returns:
        The layout to record in the manifest.
    """
    if alternate.level != "leaf":
        return {"storage": "contiguous", "compression": "none"}
    return _ragged_layout(alternate.data.shape)


def _create_options(layout: dict, *, shuffle: bool = False) -> dict:
    """The ``create_dataset`` storage options that produce a layout.

    Derived from the layout the manifest records, so the storage, chunk shape
    and compression it records are the ones the dataset gets. The layout does
    not record shuffle: that is a filter the writer adds to every compressed
    payload, and the layout schema has no field for it, so options rebuilt
    from a manifest alone produce unshuffled chunks -- valid, only larger.

    Args:
        layout: The layout from :func:`_layout_for` or :func:`_alternate_layout`.
        shuffle: Byte-shuffle before compressing; applies only when compressed.

    Returns:
        Keyword arguments for ``create_dataset``.
    """
    if layout["storage"] != "chunked":
        return {"chunks": None}
    options: dict = {"chunks": tuple(layout["chunks"])}
    if layout["compression"] != "none":
        options.update(
            compression=layout["compression"], compression_opts=layout.get("compression_opts")
        )
        if shuffle:
            options["shuffle"] = True
    return options


def _file_paths(pose_file: PoseFile) -> list[tuple[str, str]]:
    """Every dataset path the writer will create, and its owner.

    Two alternates sharing one :class:`RaggedIndex` share its two datasets, so
    those paths are listed once. The ``PoseFile`` guarantees that alternates
    naming the same offset paths share one index, so a path listed twice is a
    real collision.
    """
    paths = [("/manifest", "the manifest"), ("/provenance", "the provenance document")]
    paths += [(component.path, component.id) for component in pose_file.components]
    listed: set[int] = set()
    for component in pose_file.components:
        for alternate in component.alternates:
            label = alternate_key(component.id, alternate.name)
            paths.append((alternate.path, label))
            index = alternate.index
            if id(index) not in listed:
                listed.add(id(index))
                paths.append((index.group_offsets_path, f"{label}'s group_offsets"))
                paths.append((index.instance_offsets_path, f"{label}'s instance_offsets"))
    paths += [(a.path, f"the attachment at {a.path}") for a in pose_file.attachments]
    return paths


def _check_path_collisions(pose_file: PoseFile) -> None:
    """Refuse datasets whose HDF5 paths cannot coexist.

    Covers every dataset the writer creates -- the manifest and provenance
    documents, component payloads, alternate payloads and offsets, and
    attachments -- before any is written, so a block source is not consumed
    for a file that cannot be finished. Two paths may be individually valid
    and still be equal, or name paths where one is a parent of the other,
    which HDF5 cannot represent.

    Args:
        pose_file: The file's contents.

    Raises:
        ValueError: If two paths collide.
    """
    owners: dict[str, str] = {}
    for path, owner in _file_paths(pose_file):
        other = owners.get(path)
        if other is not None:
            raise ValueError(
                f"paths collide: {owner} and {other} are both at {path}; HDF5 cannot hold both"
            )
        owners[path] = owner
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
    options = _create_options(_layout_for(component), shuffle=_COMPONENT_SHUFFLE)
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
    _write_array(h5, component.path, data, component.id, **options)


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


def _write_ragged_set(h5: h5py.File, members: list[tuple[str, Alternate]]) -> None:
    """Write one index's offsets and every payload decoded through it, back to back.

    Written as one run, so the set occupies one byte range (spec 0029-A §2.6):
    a reader fetching one frame's segmentation reads the offsets and then the
    payload nearby, rather than seeking across the dense baseline. The order
    is the order a reader needs them -- offsets, then group-level payloads,
    then the leaf payloads they index -- and within a level, the order the
    alternates appear in, component by component. Every name that shares the
    index is in the run, so a second name does not land after other baselines.

    Args:
        h5: The open destination file.
        members: Every ``(component id, alternate)`` whose alternate uses one
            :class:`RaggedIndex`, in component order.
    """
    index: RaggedIndex = members[0][1].index
    for offsets_name in ("group_offsets", "instance_offsets"):
        _write_array(
            h5,
            getattr(index, f"{offsets_name}_path"),
            getattr(index, offsets_name),
            f"ragged {offsets_name} at {getattr(index, f'{offsets_name}_path')}",
            chunks=None,
        )
    for level in ("group", "leaf"):
        for component_id, alternate in members:
            if alternate.level != level:
                continue
            _write_array(
                h5,
                alternate.path,
                alternate.data,
                alternate_key(component_id, alternate.name),
                **_create_options(_alternate_layout(alternate), shuffle=_RAGGED_SHUFFLE),
            )


def write_pose_file(
    pose_file: PoseFile, path: str | Path, created: str | None = None, *, deep: bool = False
) -> None:
    """Write a pose file, atomically.

    Both JSON documents are built, validated **and serialized** before any file
    is opened, and the payloads are written to a temporary file in the
    destination's directory which is then moved into place. A failure at any
    point leaves an existing destination untouched.

    Both documents must be strict JSON, which every consumer can parse: a
    non-finite float (``NaN``, infinity) anywhere in the manifest or the
    provenance, including a history entry's ``extra``, is refused rather than
    written. A caller holding such a value records it some other way, as a
    string.

    Datasets are written in component order. Each ragged set -- one
    :class:`RaggedIndex` and every alternate decoded through it, under any
    name -- is written immediately before the first component that carries
    one of those alternates: the offsets, then the group-level payloads, then
    the leaf-level payloads. A caller that wants spec 0029-A's layout, with
    the ragged set after the keypoint-scale data, lists the keypoint-scale
    components before the segmentation ones.

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
            An alternate the deep check cannot decode is refused too --
            ``validate`` runs with ``require_decoded`` and reports it as an
            error -- because publishing it would break the promise that every
            alternate was compared. Without
            ``deep``, an alternate whose values disagree with its baseline
            is not caught.

    Raises:
        ValueError: If the manifest or provenance document would be invalid or
            is not strict JSON (a NaN or infinity anywhere in it), if
            two dataset paths collide, if a block source's blocks do
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
    # Every alternate decoded through one index, in component order: the set
    # written together. Keyed by identity, as the PoseFile shares indexes.
    ragged_sets: dict[int, list[tuple[str, Alternate]]] = {}
    for component in pose_file.components:
        for alternate in component.alternates:
            layouts[alternate_key(component.id, alternate.name)] = _alternate_layout(alternate)
            ragged_sets.setdefault(id(alternate.index), []).append((component.id, alternate))
    manifest = build_manifest(pose_file, layouts=layouts, created=created)
    provenance = build_provenance(pose_file.provenance)
    errors = validate_manifest(manifest) + validate_provenance(provenance)
    if errors:
        raise ValueError("refusing to write an invalid pose file: " + "; ".join(errors))

    # Serialize before opening anything: a value the schema admits can still be
    # unserializable (numpy integer scalars, for one), and discovering that
    # after the destination has been truncated destroys the caller's file.
    # Strict JSON: NaN and infinity are Python extensions that Go's and
    # JavaScript's parsers reject, so a file carrying them is unreadable there.
    try:
        manifest_json = json.dumps(manifest, allow_nan=False)
    except ValueError as error:
        raise ValueError(
            f"refusing to write an invalid pose file: the manifest is not strict JSON: {error}"
        ) from error
    try:
        provenance_json = json.dumps(provenance, allow_nan=False)
    except ValueError as error:
        raise ValueError(
            f"refusing to write an invalid pose file: the provenance is not strict JSON: {error}"
        ) from error

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
            written_sets: set[int] = set()
            for component in pose_file.components:
                # Each ragged set goes immediately before the first component
                # that carries it, so it sits beside its baseline on disk.
                for alternate in component.alternates:
                    if id(alternate.index) not in written_sets:
                        _write_ragged_set(h5, ragged_sets[id(alternate.index)])
                        written_sets.add(id(alternate.index))
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
        # for every alternate to be compared. The validator decides that
        # severity, so this filters on severity alone.
        errors = [
            f
            for f in validate(temporary, deep=deep, require_decoded=deep)
            if f.severity == "error"
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
