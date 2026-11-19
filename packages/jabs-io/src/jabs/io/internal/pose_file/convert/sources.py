"""Pass 2: the legacy pose and identity data as components, read in frame blocks.

Every frame-axis component is a :class:`BlockSource`. Its ``blocks()`` opens
the legacy file itself, read-only, reads one block of frames, scatters each
placed legacy instance into the slot the plan gave it and closes the file, so
the components outlive any handle the caller held and memory stays bounded by
one block. Each component reads only the datasets it needs; the writer calls
each ``blocks()`` once.

The values are spec 0029-A §2.3 and ADR 0002:1294-1315: points are flipped
from ``(y, x)``, ``NaN`` only where there is no coordinate, and ``point_valid``
is the confidence threshold alone, compared in float64. Empty slots get
``NaN`` points and bbox, zero confidence, embeddings and tracklets, and false
masks.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import BLOCK_FRAMES, LegacyLayout
from jabs.io.internal.pose_file.convert.plan import PlacementPlan
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12
from jabs.io.internal.pose_file.types import Component

# The browser's threshold (pose-adapter.ts:59), compared in float64 so that a
# stored float32 0.3, which is 0.30000001192, counts as valid.
CONFIDENCE_THRESHOLD = 0.3

_OCCUPIED = "jabs.pose.slot_occupied"
_MASKED = {"policy": "mask", "mask": _OCCUPIED}

# Reads one block of legacy frames, [start, stop), per legacy instance.
_Reader = Callable[[h5py.Group, int, int], npt.NDArray[np.generic]]


def pose_components(
    path: Path, layout: LegacyLayout, plan: PlacementPlan, *, block_frames: int
) -> list[Component]:
    """Return the legacy file's pose and identity data as components.

    In order: ``jabs.pose.points``, ``confidence``, ``point_valid``,
    ``slot_occupied``, ``bbox`` when the legacy bbox is trusted,
    ``tracklet_id`` when the file has tracklets, then
    ``jabs.identity.embeddings`` when it has embeddings,
    ``jabs.identity.centers`` when ``instance_id_center`` has rows and
    ``jabs.identity.external_ids`` when ``external_identity_mapping`` has
    names. No component carries provenance; the caller assigns it.

    Args:
        path: The legacy file. Each block source opens it when read.
        layout: What ``inspect_legacy`` found in it.
        plan: Where ``plan_placement`` put each legacy instance.
        block_frames: Frames per block; a multiple of 900.

    Returns:
        The components.

    Raises:
        ValueError: If ``block_frames`` is not a positive multiple of 900.
        LegacyConversionError: If an ``external_identity_mapping`` name is
            bytes that are not UTF-8.
    """
    check_block_frames(block_frames)
    frames, slots, keypoints = layout.frames, plan.slots, layout.keypoints

    def placed(
        tail: tuple[int, ...], dtype: type[np.generic], fill: object, read: _Reader
    ) -> BlockSource:
        """Return a block source scattering ``read``'s per-instance rows into slots."""
        return _placed_source(path, plan, (frames, slots, *tail), dtype, fill, read, block_frames)

    components = [
        Component(
            id="jabs.pose.points",
            axes=("frame", "slot", "keypoint", "coord"),
            data=placed((keypoints, 2), np.float32, np.nan, _read_points),
            missing={"policy": "nan"},
            units="pixel",
            coord_order="xy",
            skeleton=JABS_MOUSE12,
        ),
        Component(
            id="jabs.pose.confidence",
            axes=("frame", "slot", "keypoint"),
            data=placed((keypoints,), np.float32, 0, _read_confidence),
            missing={"policy": "none"},
            units="unitless",
            skeleton=JABS_MOUSE12,
        ),
        Component(
            id="jabs.pose.point_valid",
            axes=("frame", "slot", "keypoint"),
            data=placed((keypoints,), np.bool_, False, _read_point_valid),
            missing={"policy": "none"},
            skeleton=JABS_MOUSE12,
        ),
        Component(
            id=_OCCUPIED,
            axes=("frame", "slot"),
            data=placed((), np.bool_, False, _read_occupied),
            missing={"policy": "none"},
        ),
    ]
    if layout.bbox_generated:
        components.append(
            Component(
                id="jabs.pose.bbox",
                axes=("frame", "slot", "corner", "coord"),
                # Already (x, y): not flipped.
                data=placed((2, 2), np.float32, np.nan, _reader("bbox", np.float32)),
                missing=_MASKED,
                units="pixel",
                coord_order="xy",
            )
        )
    if plan.tracklet_id is not None:
        components.append(
            Component(
                id="jabs.pose.tracklet_id",
                axes=("frame", "slot"),
                data=_memory_source(plan.tracklet_id, block_frames),
                missing=_MASKED,
            )
        )

    with h5py.File(path, "r") as h5:
        pose = h5["poseest"]
        embedding = pose["identity_embeds"].shape[2] if layout.has_embeddings else 0
        centers = pose.get("instance_id_center")
        centers = centers[...].astype(np.float32) if centers is not None else None
        names = pose.get("external_identity_mapping")
        names = _identity_names(path, names) if names is not None else None

    if layout.has_embeddings:
        components.append(
            Component(
                id="jabs.identity.embeddings",
                axes=("frame", "slot", "embedding"),
                data=placed((embedding,), np.float32, 0, _reader("identity_embeds", np.float32)),
                missing=_MASKED,
            )
        )
    # Its rows are the identity count (ADR 0002:644-646), so an empty one is
    # not written.
    if centers is not None and centers.shape[0] > 0:
        components.append(
            Component(
                id="jabs.identity.centers",
                axes=("identity", "embedding"),
                data=centers,
                missing={"policy": "none"},
            )
        )
    # Inspection refused a length other than the identity count; with no
    # identity there is nothing to name.
    if names is not None and names.size > 0:
        components.append(
            Component(
                id="jabs.identity.external_ids",
                axes=("identity",),
                data=names,
                missing={"policy": "none"},
            )
        )
    return components


def check_block_frames(block_frames: int) -> None:
    """Refuse a block size off the 900-frame grid.

    Args:
        block_frames: Frames per block.

    Raises:
        ValueError: If ``block_frames`` is not a positive multiple of 900.
    """
    if block_frames <= 0 or block_frames % BLOCK_FRAMES:
        raise ValueError(
            f"block_frames must be a positive multiple of {BLOCK_FRAMES}, got {block_frames}"
        )


def _identity_names(path: Path, dataset: h5py.Dataset) -> npt.NDArray[np.object_]:
    """Read ``external_identity_mapping`` as text, as ``pose_est_v4.py:70-80`` does.

    Integers become their decimal strings and bytes their UTF-8 text; bytes
    that are not UTF-8 are refused rather than replaced, since the Hub deletes
    the original.
    """
    raw = dataset[...]
    if np.issubdtype(raw.dtype, np.integer):
        return np.array([str(value) for value in raw.tolist()], dtype=object)
    try:
        names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw.tolist()
        ]
    except UnicodeDecodeError as error:
        raise LegacyConversionError(
            f"{path}: poseest/external_identity_mapping holds a name that is not UTF-8: {error}"
        ) from error
    return np.array(names, dtype=object)


def _placed_source(
    path: Path,
    plan: PlacementPlan,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    fill: object,
    read: _Reader,
    block_frames: int,
) -> BlockSource:
    """Return a block source that scatters each block's placed instances into their slots."""

    def blocks() -> Iterator[tuple[int, npt.NDArray[np.generic]]]:
        """Yield the slot-major blocks, opening the legacy file for their duration."""
        with h5py.File(path, "r") as h5:
            pose = h5["poseest"]
            for start in range(0, shape[0], block_frames):
                stop = min(start + block_frames, shape[0])
                slot_of = plan.slot_of[start:stop]
                frame, instance = np.nonzero(slot_of >= 0)
                block = np.full((stop - start, *shape[1:]), fill, dtype=dtype)
                block[frame, slot_of[frame, instance]] = read(pose, start, stop)[frame, instance]
                yield start, block

    return BlockSource(shape, dtype, blocks)


def _memory_source(array: npt.NDArray[np.generic], block_frames: int) -> BlockSource:
    """Return a block source over an in-memory array, in ``block_frames`` rows."""

    def blocks() -> Iterator[tuple[int, npt.NDArray[np.generic]]]:
        """Yield the array's rows a block at a time."""
        for start in range(0, array.shape[0], block_frames):
            yield start, array[start : start + block_frames]

    return BlockSource(array.shape, array.dtype, blocks)


def _reader(name: str, dtype: type[np.generic]) -> _Reader:
    """Return a reader of one ``poseest`` dataset, cast to ``dtype``."""

    def read(pose: h5py.Group, start: int, stop: int) -> npt.NDArray[np.generic]:
        """Read ``name``'s rows in ``[start, stop)``."""
        return pose[name][start:stop].astype(dtype, copy=False)

    return read


_read_confidence = _reader("confidence", np.float32)


def _read_points(pose: h5py.Group, start: int, stop: int) -> npt.NDArray[np.float32]:
    """Read points flipped to ``(x, y)``, ``NaN`` where confidence is 0 or either value is.

    A keypoint with confidence above 0 and at most the threshold keeps its
    coordinates; ``NaN`` means no coordinate, never low confidence (ADR 0002).
    """
    points = pose["points"][start:stop][..., ::-1].astype(np.float32)
    none = (pose["confidence"][start:stop] == 0) | np.isnan(points).any(axis=-1)
    points[none] = np.nan
    return points


def _read_point_valid(pose: h5py.Group, start: int, stop: int) -> npt.NDArray[np.bool_]:
    """Read whether each keypoint's confidence is above the threshold, compared in float64.

    The threshold alone, as ADR 0002 and spec 0029-A §2.3 define it and the
    Hub's browser marks it, so the Hub's identity check agrees. A ``NaN``
    coordinate with confidence above it stays ``NaN`` in points: no coordinate.
    """
    return pose["confidence"][start:stop].astype(np.float64) > CONFIDENCE_THRESHOLD


def _read_occupied(pose: h5py.Group, start: int, stop: int) -> npt.NDArray[np.bool_]:
    """Read whether any of each instance's confidences is above 0."""
    return (pose["confidence"][start:stop] > 0).any(axis=-1)
