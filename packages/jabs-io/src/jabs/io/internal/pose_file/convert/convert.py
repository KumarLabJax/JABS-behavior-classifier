"""Convert one legacy pose file to a ``jabs.pose-file``.

The order is what makes a refusal leave nothing behind: inspection and pass 1
(the plan) read the whole legacy file and refuse anything they cannot place
before a component is built or a byte written. Pass 2 then streams the
components into ``write_pose_file``, which writes a temporary file, deep-checks
it and only then replaces the destination.
"""

import importlib.metadata
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import BLOCK_FRAMES, LegacyLayout, inspect_legacy
from jabs.io.internal.pose_file.convert.objects import object_components
from jabs.io.internal.pose_file.convert.plan import plan_placement
from jabs.io.internal.pose_file.convert.provenance import (
    convert_entry,
    encode_entry,
    legacy_hash,
    legacy_scale,
    source_format,
)
from jabs.io.internal.pose_file.convert.segmentation import segmentation_components
from jabs.io.internal.pose_file.convert.sources import pose_components
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12, jabs_mouse12
from jabs.io.internal.pose_file.types import PoseFile, Provenance, VideoInfo
from jabs.io.internal.pose_file.writer import write_pose_file

# Dense contours are int32 and points float32: four bytes a value either way.
_VALUE_BYTES = 4


@dataclass(frozen=True)
class VideoOverrides:
    """Video facts the caller knows and the legacy file does not record.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frame rate.
    """

    width: int | None = None
    height: int | None = None
    fps: float | None = None


@dataclass(frozen=True)
class ConversionReport:
    """What one conversion did.

    Attributes:
        source: The legacy file.
        destination: The written pose file.
        format: The legacy dialect, ``"pose_est_v<major>"``.
        dimensions: The written file's dimensions.
        tail_slots: How many instances went to tail slots.
        tracklet_splits: How many tracklet runs took a new id.
        longterm_mismatches: How many valid instances had a
            ``longterm_seg_id`` other than their embed id.
        dropped: The legacy paths left behind.
        seconds: Wall-clock time the conversion took.
    """

    source: Path
    destination: Path
    format: str
    dimensions: dict[str, int]
    tail_slots: int
    tracklet_splits: int
    longterm_mismatches: int
    dropped: tuple[str, ...]
    seconds: float


def block_frames_for(layout: LegacyLayout, block_bytes: int) -> int:
    """Return the frames per block that keep the widest component within a byte budget.

    The widest per-frame component is the dense contours when the file has
    segmentation (``S x C x P x 2`` int32 values a frame) and they are wider
    than the points (``S x K x 2`` float32), otherwise the points. The slot count ``S`` is not known until
    the plan is made, and the plan is itself read in these blocks, so this
    uses its bound: every identity plus every legacy instance in a tail slot.

    Args:
        layout: What ``inspect_legacy`` found.
        block_bytes: The budget for one block of the widest component.

    Returns:
        The largest multiple of 900 frames that fits, and never less than 900.
    """
    slots = layout.identity + layout.instances
    # Without segmentation contour_axis and point_axis are 0, leaving the points.
    frame_values = slots * 2 * max(layout.contour_axis * layout.point_axis, layout.keypoints)
    frame_bytes = max(1, frame_values * _VALUE_BYTES)
    return max(1, block_bytes // (frame_bytes * BLOCK_FRAMES)) * BLOCK_FRAMES


def convert_legacy(
    source: str | Path,
    destination: str | Path,
    *,
    video: VideoOverrides | None = None,
    block_bytes: int = 64 * 2**20,
    created: str | None = None,
) -> ConversionReport:
    """Convert a legacy ``pose_est_v4``..``v8`` file to a ``jabs.pose-file``.

    The legacy file is only read. The destination is written atomically and
    deep-checked before it replaces any existing file, so a refusal or a
    failure leaves an existing destination as it was.

    Args:
        source: The legacy HDF5 file.
        destination: Where to write the pose file.
        video: Video facts the legacy file does not record.
        block_bytes: The memory budget for one block of the widest component.
        created: The timestamp for the manifest and the history, ISO 8601.
            Defaults to now.

    Returns:
        What the conversion did.

    Raises:
        LegacyConversionError: If the file is outside the catalog, cannot be
            placed (a duplicate identity, a tracklet in two slots of a frame),
            or the converted file fails validation. The message says why.
    """
    started = time.perf_counter()
    source, destination = Path(source), Path(destination)
    video = video or VideoOverrides()
    stamp = created or datetime.now(timezone.utc).isoformat()
    version = importlib.metadata.version("jabs-io")

    layout = inspect_legacy(source)
    block_frames = block_frames_for(layout, block_bytes)
    with h5py.File(source, "r") as h5:
        # Pass 1 runs to completion first: every refusal it makes happens with
        # nothing built and nothing written.
        plan = plan_placement(h5, layout, block_frames=block_frames)
        cm_per_pixel, cm_per_pixel_source, scale_notes = legacy_scale(h5["poseest"].attrs)
        written_video = {"width": video.width, "height": video.height, "fps": video.fps}
        convert = convert_entry(
            h5=h5,
            layout=layout,
            plan=plan,
            content_hash=legacy_hash(source),
            version=version,
            time=stamp,
            video=written_video,
            scale_notes=scale_notes,
        )

    # In spec 0029-A §2.6's order: keypoint-scale data before segmentation,
    # so the writer puts the ragged set after it.
    components = (
        *pose_components(source, layout, plan, block_frames=block_frames),
        *segmentation_components(source, layout, plan, block_frames=block_frames),
        *object_components(source, layout),
    )
    encode = encode_entry(components, version=version, time=stamp)
    dimensions = {"frame": layout.frames, "slot": plan.slots, "identity": layout.identity}
    pose_file = PoseFile(
        dimensions=dimensions,
        video=VideoInfo(
            frame_count=layout.frames,
            width=video.width,
            height=video.height,
            fps=video.fps,
            cm_per_pixel=cm_per_pixel,
            cm_per_pixel_source=cm_per_pixel_source,
        ),
        skeletons={JABS_MOUSE12: jabs_mouse12()},
        components=components,
        provenance=Provenance(history=(convert,) if encode is None else (convert, encode)),
    )
    try:
        write_pose_file(pose_file, destination, created=created, deep=True)
    except ValueError as error:
        raise LegacyConversionError(f"{source}: {error}") from error

    return ConversionReport(
        source=source,
        destination=destination,
        format=source_format(layout),
        dimensions=dimensions,
        tail_slots=len(plan.tail_slots),
        tracklet_splits=len(plan.tracklet_splits),
        longterm_mismatches=len(plan.longterm_mismatches),
        dropped=layout.dropped,
        seconds=time.perf_counter() - started,
    )
