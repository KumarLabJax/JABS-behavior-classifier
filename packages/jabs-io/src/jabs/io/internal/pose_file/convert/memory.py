"""Refuse a legacy file whose declared shapes would exhaust memory, before allocating.

The plan holds arrays sized frames x slots x contours, and pass 2 reads blocks
of at least one chunk row of the dense contours whatever ``block_bytes`` asks,
all from the shapes the file declares. A file of a few kilobytes can declare
billions of frames, or a contour axis no real file has, in chunked storage it
never wrote. So the converter estimates its peak memory from the shapes alone
and refuses a file above the caller's cap, naming both.

The estimate is a bound for the shapes real files have, not a measurement: it
uses the slot count's upper bound (every identity plus every legacy instance),
because the real count is only known once the plan is made. Measured with
tracemalloc, it is 1.7 to 2.7 times the real peak on the public v4 and v6
samples, a real 108,000-frame v8 file and v6-width synthetics of up to
324,000 frames.

A second, independent check refuses a large dataset whose declared size is
far beyond what it stores: data that was never written. It catches what the
estimate does not count, such as a retired dataset the conversion never reads.

A third refuses an attribute declaring more than ``MAX_ATTRIBUTE_BYTES``, from
its metadata, before inspection or the scale reads any attribute's value.
"""

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.validate import deep_check_block_frames
from jabs.io.internal.pose_file.writer import chunk_shape

if TYPE_CHECKING:
    from jabs.io.internal.pose_file.convert.inspect import LegacyLayout

logger = logging.getLogger(__name__)

# A dataset is suspicious when it declares more than this many bytes ...
STORAGE_RATIO_FLOOR = 2**30
# ... and that is more than this many times what it stores. gzip of constant
# data reaches about 914:1 on legacy-shaped chunks, and deflate cannot pass
# about 1032:1; the 733 real legacy files surveyed reach 573:1 at most. So
# only storage that was never written, or a filter no legacy writer uses,
# goes past this.
STORAGE_RATIO = 2000

# The cap on the legacy attributes, in total, that the conversion records
# verbatim in its history (``provenance.legacy_attributes``). Real files hold
# a few kilobytes; dense attribute storage lets a crafted one hold gigabytes,
# and an array of object references expands to one path string per element.
MAX_ATTRIBUTE_BYTES = 16 * 2**20

# The poseest datasets read in frame blocks, by pass 1, pass 2 or inspection.
FRAME_DATASETS = (
    "points",
    "confidence",
    "id_mask",
    "instance_embed_id",
    "instance_track_id",
    "longterm_seg_id",
    "identity_embeds",
    "seg_data",
    "seg_external_flag",
    "bbox",
)
# The deep check holds a block of the dense contours, the valid-point mask,
# the expected points cut from it and the alternate's points: under 3 blocks
# measured, 4 allowed.
_DEEP_CHECK_COPIES = 4

# The poseest datasets read whole.
_WHOLE_DATASETS = ("instance_id_center", "external_identity_mapping")


# The most bytes of a legacy dataset's chunk cache. A block shorter than a
# legacy dataset's chunks would decompress each chunk once per block that
# touches it, so the dataset is opened with a cache holding one chunk row
# (every chunk over the chunks' frames) when that row is at most this. A larger
# row keeps HDF5's default 1 MiB cache: slower, never unbounded.
LEGACY_CACHE_MAX_BYTES = 2 * 2**30


def legacy_cache_bytes(dataset: h5py.Dataset, block_frames: int) -> int:
    """Return the chunk cache a legacy frame dataset is read with, in blocks; 0 for the default.

    Args:
        dataset: A legacy dataset read along its first axis.
        block_frames: The frames read at a time.

    Returns:
        One chunk row's decompressed bytes when blocks split its chunks and
        the row is at most ``LEGACY_CACHE_MAX_BYTES``; otherwise 0, as when
        one block holds every frame and nothing is read twice.
    """
    chunks = dataset.chunks
    if (
        chunks is None
        or dataset.shape is None
        or block_frames % chunks[0] == 0
        or block_frames >= dataset.shape[0]
    ):
        return 0
    across = math.prod(-(-n // c) for n, c in zip(dataset.shape[1:], chunks[1:], strict=True))
    row = across * math.prod(chunks) * dataset.dtype.itemsize
    return row if row <= LEGACY_CACHE_MAX_BYTES else 0


def open_cached(group: h5py.Group, name: str, block_frames: int) -> h5py.Dataset:
    """Open a legacy frame dataset with the chunk cache ``legacy_cache_bytes`` gives it.

    HDF5 shares one open dataset, and its cache, among every handle to it, so
    a caller holds the returned handle while it reads the dataset in blocks;
    other handles opened meanwhile, like ``group[name]``, use the same cache.

    Args:
        group: The group holding the dataset.
        name: Its name in the group.
        block_frames: The frames read at a time.

    Returns:
        The dataset.
    """
    dataset = group[name]
    nbytes = legacy_cache_bytes(dataset, block_frames)
    if not nbytes:
        return dataset
    chunks = nbytes // (math.prod(dataset.chunks) * dataset.dtype.itemsize)
    access = h5py.h5p.create(h5py.h5p.DATASET_ACCESS)
    # One slot a chunk at least: a chunk row is a run of consecutive chunk
    # indices, so a table at least that long has no collisions within it.
    access.set_chunk_cache(2 * chunks + 1, nbytes, 0.75)
    del dataset
    return h5py.Dataset(h5py.h5d.open(group.id, name.encode(), dapl=access))


# The converted dense contours' axes, which decide their chunk shape.
_CONTOUR_AXES = ("frame", "slot", "contour", "point", "coord")


def contour_rows(frames: int, contour_axis: int, point_axis: int) -> int:
    """Return the frames in one chunk of the converted dense contours; 1 without contours.

    The writer chunks contours along frames and the point axis only
    (``writer.chunk_shape``), so this depends on the point axis and not on
    the slot count, which is unknown until the plan is made.

    Args:
        frames: The frame count.
        contour_axis: The padded contour extent; 0 without segmentation.
        point_axis: The padded point extent; 0 without segmentation.

    Returns:
        The chunk's frame extent: the smallest block the converter reads.
    """
    if min(frames, contour_axis, point_axis) <= 0:
        return 1
    return chunk_shape(_CONTOUR_AXES, (frames, 1, contour_axis, point_axis, 2), 4)[0]


def contour_chunk_frames(layout: "LegacyLayout") -> int:
    """Return :func:`contour_rows` for an inspected legacy file.

    Args:
        layout: What ``inspect_legacy`` found in the file.

    Returns:
        The frames in one chunk of its converted dense contours; 1 without
        segmentation.
    """
    if not layout.has_segmentation:
        return 1
    return contour_rows(layout.frames, layout.contour_axis, layout.point_axis)


def estimate_memory(h5: h5py.File, *, identity: int, block_frames: int) -> int:
    """Estimate a conversion's peak memory, in bytes, from the declared shapes.

    With ``F`` frames, ``N`` legacy instances, ``S = identity + N`` slots (the
    bound), ``C`` contours, ``T`` 1 with tracklets and ``G`` 1 with
    segmentation, the plan keeps, while it is made (``plan.py``), per frame:

    - ``N * (2 + 4T + 5GC + 16)``: ``slot_of`` (int16), the legacy tracklets
      (uint32), the legacy contour lengths and sources (uint32 and int8) and
      the int64 ``np.nonzero`` pair that scatters them into slots;
    - ``N * 57``: the tail and longterm-mismatch rows, as int64 columns per
      block, at worst one of each per legacy instance;
    - ``4TS``: ``tracklet_id`` (uint32);
    - the larger of ``48TS``, the tracklet split's int64 index arrays, and
      ``GS * (13C + 12)``: the slot contour lengths and sources, the ``used``
      mask, ``group_offsets`` (at most one uint64 per contour),
      ``contour_count`` (uint32) and ``instance_offsets`` (uint64). They
      are not alive at once.

    Once made, the plan keeps ``33N + 4TS + GS * (13C + 12)`` (``slot_of``,
    the rows as narrower columns, the slot arrays), and the deep check reads
    its own copy of both offsets, ``GS * (8C + 8)``. The per-frame term is
    the larger of the two phases, times ``F``.

    To that it adds the larger of two block terms. Writing: the largest
    frame block's working set across the frame datasets, which for one with
    ``v`` values per instance and item size ``s`` is
    ``N * v * (2s + 12) + 9 * S * v`` bytes a frame (the block as read and
    one copy in its own dtype, a 4-byte cast, the scatter's gather and masks;
    then the 4-byte output block and the ragged alternate's mask and rows),
    times ``min(block_frames, F)``. Checking: four of the deep check's blocks
    of dense contours (``validate.deep_check_block_frames``: 64 MiB, widened
    to whole chunk rows of the contours as the writer chunks them), or the
    whole baseline if smaller. And it adds, at their item size plus a 4-byte
    cast, the datasets read whole: the static and dynamic objects,
    ``instance_id_center`` and ``external_identity_mapping``.

    A file whose ``points`` has no instance axis (``pose_est_v2``) has
    ``N = 1``, and every axis after the frame counts toward ``v``.

    Last, it adds the largest single chunk, ``prod(chunks) * itemsize``,
    across every dataset it reads (the frame datasets, the datasets read
    whole and the objects); a contiguous dataset adds nothing. HDF5
    decompresses a whole chunk to read any part of it, and a chunk can be
    far larger than what it stores on disk, or than the dataset's extent.
    One chunk is counted: the reads are sequential, and a chunk larger than
    the dataset's chunk cache (h5py's default, 1 MiB) is not kept after its
    read, so two large chunks are never decompressed at once.

    And it adds every frame dataset's chunk cache (``legacy_cache_bytes``):
    one chunk row where blocks are shorter than the dataset's chunks.

    Args:
        h5: The open legacy file; only shapes and dtypes are read.
        identity: The identity count, or a lower bound for it.
        block_frames: The frames pass 2 reads at a time.

    Returns:
        The estimate, in bytes.
    """
    pose = h5["poseest"]
    points = pose["points"]
    instance_axis = points.ndim != 3
    frames = int(points.shape[0])
    instances = int(points.shape[1]) if instance_axis else 1
    # Where a frame dataset's per-instance values start.
    value_axis = 2 if instance_axis else 1
    slots = identity + instances
    seg = pose.get("seg_data")
    contours = int(seg.shape[2]) if seg is not None and seg.ndim == 5 else 0
    tracklets = int("instance_track_id" in pose)
    segmentation = int(seg is not None)

    # What the plan keeps once made, and the deep check's offsets beside it.
    kept = (
        instances * (2 + 31) + 4 * tracklets * slots + segmentation * slots * (13 * contours + 12)
    )
    planning = (
        instances * (2 + 4 * tracklets + 5 * segmentation * contours + 16 + 57)
        + 4 * tracklets * slots
        + max(48 * tracklets * slots, segmentation * slots * (13 * contours + 12))
    )
    checking = kept + segmentation * slots * (8 * contours + 8)
    per_frame = max(planning, checking)

    block = 0
    for name in FRAME_DATASETS:
        dataset = pose.get(name)
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim < value_axis:
            continue
        values = math.prod(dataset.shape[value_axis:])
        size = dataset.dtype.itemsize
        block = max(block, instances * values * (2 * size + 12) + 9 * slots * values)
    writing = min(block_frames, frames) * block

    checking_block = 0
    if seg is not None and seg.ndim == 5 and frames > 0:
        # The output contours, int32, for every frame, read in the deep
        # check's blocks of the chunks the writer gives them.
        shape = (frames, slots, *seg.shape[2:])
        dense = math.prod(shape) * 4
        rows = contour_rows(frames, int(seg.shape[2]), int(seg.shape[3]))
        chunks = (rows, 1, 1, int(seg.shape[3]), 1)
        frame_bytes = math.prod(shape[1:]) * 4
        block = deep_check_block_frames(shape, 4, chunks) * frame_bytes
        checking_block = _DEEP_CHECK_COPIES * min(block, dense)

    whole = sum(_whole_bytes(pose.get(name)) for name in _WHOLE_DATASETS)
    read = [pose.get(name) for name in (*FRAME_DATASETS, *_WHOLE_DATASETS)]
    for root in ("static_objects", "dynamic_objects"):
        group = h5.get(root)
        if isinstance(group, h5py.Group):
            whole += _group_bytes(group)
            read.extend(_group_datasets(group))
    chunk = max((_chunk_bytes(dataset) for dataset in read), default=0)
    # The chunk caches of the frame datasets read in blocks: the plan and
    # pass 2 each hold every one they read for the whole pass.
    cache = sum(
        legacy_cache_bytes(dataset, block_frames)
        for dataset in read[: len(FRAME_DATASETS)]
        if isinstance(dataset, h5py.Dataset)
    )

    return frames * per_frame + max(writing, checking_block) + whole + chunk + cache


def require_memory(path: Path, estimate: int, max_memory: int) -> None:
    """Refuse a conversion whose estimated peak memory is above the cap.

    Args:
        path: The legacy file, for the message.
        estimate: What ``estimate_memory`` returned.
        max_memory: The cap, in bytes.

    Raises:
        LegacyConversionError: If ``estimate`` is above ``max_memory``.
    """
    logger.debug("%s: estimated peak memory %d bytes, cap %d", path, estimate, max_memory)
    if estimate > max_memory:
        raise LegacyConversionError(
            f"{path}: converting it needs an estimated {estimate / 2**30:.1f} GiB "
            f"({estimate} bytes) of memory, above the cap of {max_memory / 2**30:.1f} GiB "
            f"({max_memory} bytes); "
            "its declared shapes are larger than any real recording's, or the cap is too low"
        )


def check_storage(path: Path, h5: h5py.File) -> None:
    """Refuse a large dataset that declares far more than it stores.

    Run only on a file already checked to be self-contained: the walk visits
    hard links only then.

    Args:
        path: The legacy file, for the message.
        h5: The open legacy file.

    Raises:
        LegacyConversionError: If a dataset declares more than
            ``STORAGE_RATIO_FLOOR`` bytes and more than ``STORAGE_RATIO``
            times the bytes it stores. The message names it.
    """
    found: list[str] = []

    def visit(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset) or found:
            return
        declared = _declared_bytes(obj)
        if declared <= STORAGE_RATIO_FLOOR:
            return
        stored = obj.id.get_storage_size()
        if declared > STORAGE_RATIO * stored:
            found.append(
                f"{path}: {name} declares {declared} bytes but stores {stored}; refusing "
                f"a dataset more than {STORAGE_RATIO} times larger than its storage, which "
                "was never written"
            )

    h5.visititems(visit)
    if found:
        raise LegacyConversionError(found[0])


def check_attributes(path: Path, h5: h5py.File) -> None:
    """Refuse an attribute whose declared size alone is above ``MAX_ATTRIBUTE_BYTES``.

    Only metadata is read, so it runs before anything reads an attribute's
    value. Attributes are never compressed, so a value read once this passes
    is at most the cap, or for a variable-length string, which declares only
    its pointer, the file's size; ``provenance.legacy_attributes`` then caps
    what they expand to. Run only on a file already checked to be
    self-contained: the walk visits hard links only then.

    Args:
        path: The legacy file, for the message.
        h5: The open legacy file.

    Raises:
        LegacyConversionError: If an attribute declares more than
            ``MAX_ATTRIBUTE_BYTES``, or its metadata cannot be read. The
            message names its object and name.
    """
    found: list[str] = []

    def visit(name: str, obj: h5py.HLObject) -> None:
        if found:
            return
        for key in obj.attrs:
            try:
                declared = attribute_bytes(obj.attrs, key)
            except (TypeError, ValueError, OSError) as error:
                found.append(f"{path}: attribute {key!r} of {name} cannot be recorded: {error}")
                return
            if declared > MAX_ATTRIBUTE_BYTES:
                found.append(
                    f"{path}: attribute {key!r} of {name} declares {declared} bytes; refusing "
                    f"legacy attributes over the cap of {MAX_ATTRIBUTE_BYTES} bytes, which "
                    "the conversion records verbatim"
                )
                return

    visit("/", h5)
    h5.visititems(visit)
    if found:
        raise LegacyConversionError(found[0])


def attribute_bytes(attrs: h5py.AttributeManager, key: str) -> int:
    """Return an attribute's declared size in bytes, from its metadata, without reading it.

    A variable-length string declares the size of its pointer, not its text,
    and an empty (null) attribute 0.

    Args:
        attrs: The attributes of one object.
        key: The attribute's name.

    Returns:
        Its value count times its item size.
    """
    attribute = attrs.get_id(key)
    return 0 if attribute.shape is None else math.prod(attribute.shape) * attribute.dtype.itemsize


def _values(dataset: h5py.Dataset) -> int:
    """Return a dataset's declared value count, exactly; 0 for an empty (null) one."""
    return 0 if dataset.shape is None else math.prod(dataset.shape)


def _declared_bytes(dataset: h5py.Dataset) -> int:
    """Return a dataset's declared size in bytes."""
    return _values(dataset) * dataset.dtype.itemsize


def _whole_bytes(dataset: object) -> int:
    """Return what reading a dataset whole and casting it costs: its size at itemsize plus 4."""
    if not isinstance(dataset, h5py.Dataset):
        return 0
    return _values(dataset) * (dataset.dtype.itemsize + 4)


def _chunk_bytes(dataset: object) -> int:
    """Return a dataset's chunk size in bytes, as decompressed; 0 if it is contiguous or absent."""
    if not isinstance(dataset, h5py.Dataset) or dataset.chunks is None:
        return 0
    return math.prod(dataset.chunks) * dataset.dtype.itemsize


def _group_datasets(group: h5py.Group) -> list[object]:
    """Return a group's members, one level of subgroups expanded into theirs."""
    members: list[object] = []
    for name in group:
        member = group[name]
        if isinstance(member, h5py.Group):
            members.extend(member[inner] for inner in member)
        else:
            members.append(member)
    return members


def _group_bytes(group: h5py.Group) -> int:
    """Return ``_whole_bytes`` summed over a group's datasets, one level of subgroups included."""
    return sum(_whole_bytes(member) for member in _group_datasets(group))
