"""Builders writing tiny unified pose files for the ``Pose`` tests."""

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file import (
    Component,
    HistoryEntry,
    PoseFile,
    Provenance,
    VideoInfo,
    write_pose_file,
)
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12, jabs_mouse12

KEYPOINTS = 12

# A pinned timestamp keeps the written files reproducible.
_TIME = "2026-11-20T00:00:00Z"

# Contour and point capacity of built segmentation.
SEG_CONTOURS = 2
SEG_POINTS = 3


def make_points(frames: int, slots: int) -> npt.NDArray[np.float32]:
    """Return distinct, finite ``(F, S, K, 2)`` points: ``x = 100 f + 10 s + k``, ``y = x + 0.5``."""
    f, s, k = np.meshgrid(np.arange(frames), np.arange(slots), np.arange(KEYPOINTS), indexing="ij")
    x = (100 * f + 10 * s + k).astype(np.float32)
    return np.stack([x, x + 0.5], axis=-1)


def make_segmentation(frames: int, slots: int) -> dict[str, npt.NDArray[np.generic]]:
    """Return packed segmentation arrays whose values encode their frame and slot.

    Frame ``f``, slot ``s`` holds ``(f + s) % 3`` contours (at most
    ``SEG_CONTOURS``); contour ``c`` has ``c + 2`` points, point ``p`` is
    ``(f, 10 s + p)``. Padding is -1.
    """
    contours = np.full((frames, slots, SEG_CONTOURS, SEG_POINTS, 2), -1, dtype=np.int32)
    count = np.zeros((frames, slots), dtype=np.uint32)
    length = np.zeros((frames, slots, SEG_CONTOURS), dtype=np.uint32)
    flags = np.zeros((frames, slots, SEG_CONTOURS), dtype=bool)
    for f in range(frames):
        for s in range(slots):
            n = min((f + s) % 3, SEG_CONTOURS)
            count[f, s] = n
            for c in range(n):
                length[f, s, c] = c + 2
                flags[f, s, c] = c == 0
                for p in range(c + 2):
                    contours[f, s, c, p] = (f, 10 * s + p)
    return {"contours": contours, "count": count, "length": length, "flags": flags}


def build_pose_file(
    path: Path,
    *,
    frames: int = 6,
    slots: int = 2,
    identity: int = 2,
    points: npt.NDArray[np.float32] | None = None,
    point_valid: npt.NDArray[np.bool_] | None = None,
    slot_usable: npt.NDArray[np.bool_] | None = None,
    tracklet_id: npt.NDArray[np.uint32] | None = None,
    fps: float | None = 30.0,
    cm_per_pixel: float | None = None,
    static_objects: Mapping[str, npt.NDArray[np.float32]] | None = None,
    dynamic_objects: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    segmentation: bool = False,
    bbox: bool = False,
    external_ids: list[str] | None = None,
    convert_extra: dict | None = None,
) -> Path:
    """Write a tiny unified pose file and return its path.

    Args:
        path: Destination.
        frames: Frame count.
        slots: Slot count.
        identity: ``dimensions.identity``.
        points: ``(F, S, K, 2)`` points; defaults to :func:`make_points`.
        point_valid: ``(F, S, K)`` validity; defaults to all valid.
        slot_usable: Optional ``(F, S)`` ``jabs.pose.slot_usable``.
        tracklet_id: Optional ``(F, S)`` ``jabs.pose.tracklet_id``.
        fps: ``video.fps``.
        cm_per_pixel: ``video.cm_per_pixel``.
        static_objects: Name to ``(O, P, 2)`` points.
        dynamic_objects: Name to ``(frame_index, points, counts)``.
        segmentation: Add :func:`make_segmentation`'s components.
        bbox: Add ``jabs.pose.bbox``, slot ``s`` frame ``f`` holding
            ``[[f, s], [f + 1, s + 1]]``.
        external_ids: Optional ``jabs.identity.external_ids``.
        convert_extra: When given, the provenance history is a ``convert``
            entry with this ``extra``; otherwise an ``infer`` entry.

    Returns:
        ``path``.
    """
    points = make_points(frames, slots) if points is None else points
    if point_valid is None:
        point_valid = np.ones((frames, slots, KEYPOINTS), dtype=bool)
    occupied = point_valid.any(axis=-1)
    masked = {"policy": "mask", "mask": "jabs.pose.slot_occupied"}
    components = [
        Component(
            id="jabs.pose.points",
            axes=("frame", "slot", "keypoint", "coord"),
            data=points.astype(np.float32),
            missing={"policy": "nan"},
            units="pixel",
            coord_order="xy",
            skeleton=JABS_MOUSE12,
        ),
        Component(
            id="jabs.pose.point_valid",
            axes=("frame", "slot", "keypoint"),
            data=point_valid,
            missing={"policy": "none"},
            skeleton=JABS_MOUSE12,
        ),
        Component(
            id="jabs.pose.slot_occupied",
            axes=("frame", "slot"),
            data=occupied,
            missing={"policy": "none"},
        ),
    ]
    if slot_usable is not None:
        components.append(
            Component(
                id="jabs.pose.slot_usable",
                axes=("frame", "slot"),
                data=slot_usable,
                missing={"policy": "none"},
            )
        )
    if tracklet_id is not None:
        components.append(
            Component(
                id="jabs.pose.tracklet_id",
                axes=("frame", "slot"),
                data=tracklet_id.astype(np.uint32),
                missing=masked,
            )
        )
    if bbox:
        f, s = np.meshgrid(np.arange(frames), np.arange(slots), indexing="ij")
        corners = np.stack(
            [np.stack([f, s], axis=-1), np.stack([f + 1, s + 1], axis=-1)], axis=-2
        ).astype(np.float32)
        components.append(
            Component(
                id="jabs.pose.bbox",
                axes=("frame", "slot", "corner", "coord"),
                data=corners,
                missing=masked,
                units="pixel",
                coord_order="xy",
            )
        )
    if external_ids is not None:
        components.append(
            Component(
                id="jabs.identity.external_ids",
                axes=("identity",),
                data=np.array(external_ids, dtype=object),
                missing={"policy": "none"},
            )
        )
    for name, data in (static_objects or {}).items():
        components.append(
            Component(
                id=f"jabs.static_objects.{name}",
                axes=("object", "point", "coord"),
                data=data.astype(np.float32),
                missing={"policy": "none"},
                units="pixel",
                coord_order="xy",
            )
        )
    for name, (frame_index, object_points, counts) in (dynamic_objects or {}).items():
        prefix = f"jabs.dynamic_objects.{name}"
        components += [
            Component(
                id=f"{prefix}.frame_index",
                axes=("sample",),
                data=frame_index.astype(np.uint32),
                missing={"policy": "none"},
                units="frame",
                sparse_index=f"{prefix}.frame_index",
            ),
            Component(
                id=f"{prefix}.points",
                axes=("sample", "object", "point", "coord"),
                data=object_points.astype(np.float32),
                missing={"policy": "length", "length": f"{prefix}.counts"},
                units="pixel",
                coord_order="xy",
                sparse_index=f"{prefix}.frame_index",
            ),
            Component(
                id=f"{prefix}.counts",
                axes=("sample",),
                data=counts.astype(np.uint32),
                missing={"policy": "none"},
                units="unitless",
                sparse_index=f"{prefix}.frame_index",
            ),
        ]
    if segmentation:
        seg = make_segmentation(frames, slots)
        components += [
            Component(
                id="jabs.segmentation.contours",
                axes=("frame", "slot", "contour", "point", "coord"),
                data=seg["contours"],
                missing={"policy": "length", "length": "jabs.segmentation.contour_length"},
                units="pixel",
                coord_order="xy",
            ),
            Component(
                id="jabs.segmentation.contour_count",
                axes=("frame", "slot"),
                data=seg["count"],
                missing={"policy": "none"},
            ),
            Component(
                id="jabs.segmentation.contour_length",
                axes=("frame", "slot", "contour"),
                data=seg["length"],
                missing={"policy": "none"},
            ),
            Component(
                id="jabs.segmentation.external_flag",
                axes=("frame", "slot", "contour"),
                data=seg["flags"],
                missing={"policy": "none"},
            ),
        ]

    if convert_extra is None:
        history = HistoryEntry(operation="infer", tool="test", version="1", time=_TIME)
    else:
        history = HistoryEntry(
            operation="convert",
            tool="jabs-io",
            version="1",
            time=_TIME,
            source={"format": "pose_est_v6", "content_hash": "blake2b:00", "filename": "x.h5"},
            synthesized=("jabs.pose.point_valid",),
            extra=convert_extra,
        )
    pose_file = PoseFile(
        dimensions={"frame": frames, "slot": slots, "identity": identity},
        video=VideoInfo(frame_count=frames, fps=fps, cm_per_pixel=cm_per_pixel),
        skeletons={JABS_MOUSE12: jabs_mouse12()},
        components=tuple(components),
        provenance=Provenance(history=(history,)),
    )
    write_pose_file(pose_file, path, created=_TIME)
    return path


@pytest.fixture
def pose_file_factory(tmp_path: Path) -> Callable[..., Path]:
    """Return a function writing a tiny unified pose file under ``tmp_path``.

    Its ``name`` keyword (default ``"vid_pose.h5"``) names the file; every
    other keyword goes to :func:`build_pose_file`.
    """

    def factory(name: str = "vid_pose.h5", **kwargs: object) -> Path:
        return build_pose_file(tmp_path / name, **kwargs)  # type: ignore[arg-type]

    return factory
