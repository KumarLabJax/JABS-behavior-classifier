"""Shared fixtures for pose file format tests."""

import dataclasses

import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file.embeddings import (
    EMBEDDING_AXES,
    EmbeddingDescriptor,
    EmbeddingModel,
    TemporalWindow,
)
from jabs.io.internal.pose_file.skeletons import jabs_mouse12
from jabs.io.internal.pose_file.types import (
    Component,
    HistoryEntry,
    PoseFile,
    Provenance,
    ProvenanceRecord,
    VideoInfo,
)

MOUSE12 = jabs_mouse12()

# The dense segmentation helper's padded widths: up to two contours per
# instance, up to three points per contour.
_SEGMENTATION_CONTOURS = 2
_SEGMENTATION_POINTS = 3


def ragged_segmentation(frames: int = 4, slots: int = 2) -> dict[str, npt.NDArray[np.generic]]:
    """Build dense segmentation arrays and the ragged form that encodes them.

    Instances hold zero, one or two contours, and contours hold one to three
    points, so the offsets are not uniform and some instances own no groups.
    The ragged form follows ADR 0002's decoding rule: ``instance_offsets``
    holds group indices in row-major ``(frame, slot)`` order, and
    ``contour_offsets`` holds payload indices.

    Args:
        frames: Number of frames.
        slots: Number of slots.

    Returns:
        ``contours`` ``(F, S, 2, 3, 2)`` int32 (padding is -1),
        ``contour_count`` ``(F, S)`` uint32, ``contour_length`` ``(F, S, 2)``
        uint32, ``external_flag`` ``(F, S, 2)`` bool, and the ragged
        ``points`` ``(N, 2)`` int32, ``flags`` ``(G,)`` bool,
        ``contour_offsets`` ``(G + 1,)`` uint64 and ``instance_offsets``
        ``(F * S + 1,)`` uint64.
    """
    shape = (frames, slots, _SEGMENTATION_CONTOURS)
    contours = np.full((*shape, _SEGMENTATION_POINTS, 2), -1, dtype=np.int32)
    contour_count = np.zeros((frames, slots), dtype=np.uint32)
    contour_length = np.zeros(shape, dtype=np.uint32)
    external_flag = np.zeros(shape, dtype=bool)
    points: list[list[int]] = []
    flags: list[bool] = []
    contour_offsets = [0]
    instance_offsets = [0]
    for f in range(frames):
        for s in range(slots):
            count = (f + s) % (_SEGMENTATION_CONTOURS + 1)
            contour_count[f, s] = count
            for c in range(count):
                length = 1 + (f + 2 * s + c) % _SEGMENTATION_POINTS
                contour = [[10 * f + p, 100 * s + 10 * c + p] for p in range(length)]
                contours[f, s, c, :length] = contour
                contour_length[f, s, c] = length
                # The first contour of an instance is its outer boundary.
                external_flag[f, s, c] = c == 0
                points.extend(contour)
                flags.append(c == 0)
                contour_offsets.append(len(points))
            instance_offsets.append(len(contour_offsets) - 1)
    return {
        "contours": contours,
        "contour_count": contour_count,
        "contour_length": contour_length,
        "external_flag": external_flag,
        "points": np.array(points, dtype=np.int32).reshape(-1, 2),
        "flags": np.array(flags, dtype=bool),
        "contour_offsets": np.array(contour_offsets, dtype=np.uint64),
        "instance_offsets": np.array(instance_offsets, dtype=np.uint64),
    }


def build_sample_pose_file(width: int | None = 800, height: int | None = 800) -> PoseFile:
    """Build a four-frame, two-slot pose file with the four core pose components.

    Args:
        width: Frame width, or None to model a legacy conversion that could not
            determine the coordinate space.
        height: Frame height, or None for the same reason.

    Returns:
        A valid pose file.
    """
    frames, slots = 4, 2
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 800, (frames, slots, 12, 2)).astype(np.float32)
    confidence = rng.uniform(0, 1, (frames, slots, 12)).astype(np.float32)
    point_valid = confidence > 0.3
    slot_occupied = point_valid.any(axis=2)
    points[~point_valid] = np.nan

    record = ProvenanceRecord(
        producer="test",
        version="0.0.1",
        created="2026-09-03T00:00:00Z",
        parameters={"confidence_threshold": 0.3},
    )
    return PoseFile(
        dimensions={"frame": frames, "slot": slots, "identity": slots},
        video=VideoInfo(frame_count=frames, width=width, height=height, fps=30.0),
        skeletons={"jabs.mouse12": MOUSE12},
        components=(
            Component(
                id="jabs.pose.points",
                axes=("frame", "slot", "keypoint", "coord"),
                data=points,
                missing={"policy": "nan"},
                units="pixel",
                coord_order="xy",
                skeleton="jabs.mouse12",
                provenance="jabs.pose",
            ),
            Component(
                id="jabs.pose.confidence",
                axes=("frame", "slot", "keypoint"),
                data=confidence,
                missing={"policy": "none"},
                units="unitless",
                skeleton="jabs.mouse12",
                provenance="jabs.pose",
            ),
            Component(
                id="jabs.pose.point_valid",
                axes=("frame", "slot", "keypoint"),
                data=point_valid,
                missing={"policy": "none"},
                skeleton="jabs.mouse12",
                provenance="jabs.pose",
            ),
            Component(
                id="jabs.pose.slot_occupied",
                axes=("frame", "slot"),
                data=slot_occupied,
                missing={"policy": "none"},
                provenance="jabs.pose",
            ),
        ),
        provenance=Provenance(
            records={"jabs.pose": record},
            history=(
                HistoryEntry(
                    operation="infer",
                    tool="test",
                    version="0.0.1",
                    time="2026-09-03T00:00:00Z",
                ),
            ),
        ),
    )


EMBEDDING_WEIGHTS = "sha256:" + "0" * 64


def build_embedding_component(
    name: str = "trunk",
    kind: str = "instance",
    *,
    frames: int = 4,
    slots: int = 2,
    dim: int = 8,
    dtype: str = "float16",
    tap: str = "trunk.final",
) -> Component:
    """Build a valid embedding component with random, fully present rows.

    Args:
        name: The ``<name>`` segment of the component id.
        kind: ``"instance"`` or ``"frame"``.
        frames: Length of the frame axis.
        slots: Length of the slot axis, for the instance kind.
        dim: Length of the feature axis.
        dtype: Payload dtype.
        tap: The descriptor's tap.

    Returns:
        A component with id ``jabs.embedding.<name>.<kind>``.
    """
    shape = (frames, slots, dim) if kind == "instance" else (frames, dim)
    rng = np.random.default_rng(11)
    descriptor = EmbeddingDescriptor(
        name=name,
        kind=kind,
        dim=dim,
        model=EmbeddingModel("vit-carry-spal", EMBEDDING_WEIGHTS),
        tap=tap,
        pooling="mask_mean" if kind == "instance" else "mean",
        window=TemporalWindow(9, 9),
    )
    return Component(
        id=f"jabs.embedding.{name}.{kind}",
        axes=EMBEDDING_AXES[kind],
        data=rng.standard_normal(shape).astype(dtype),
        missing={"policy": "nan"},
        embedding=descriptor,
    )


def build_pose_file_with_embeddings() -> PoseFile:
    """Build the sample pose file plus ``trunk.instance`` and ``trunk.frame`` embeddings.

    Instance rows are NaN wherever ``jabs.pose.slot_occupied`` is false.

    Returns:
        A valid pose file.
    """
    base = build_sample_pose_file()
    occupied = np.asarray(base.component("jabs.pose.slot_occupied").data)
    instance = build_embedding_component("trunk", "instance")
    rows = np.array(instance.data, copy=True)
    rows[~occupied] = np.nan
    instance = dataclasses.replace(instance, data=rows)
    frame = build_embedding_component("trunk", "frame")
    return dataclasses.replace(base, components=(*base.components, instance, frame))


@pytest.fixture
def minimal_manifest() -> dict:
    """A manifest with exactly one component, valid against the schema."""
    return {
        "format": "jabs.pose-file",
        "schema_revision": 1,
        "dimensions": {"frame": 4, "slot": 1, "identity": 1},
        "video": {"frame_count": 4, "width": 800, "height": 800, "fps": 30.0},
        "skeletons": {
            "jabs.mouse12": {
                "body_parts": list(MOUSE12.body_parts),
                "edges": [list(e) for e in MOUSE12.edges],
            }
        },
        "components": [
            {
                "id": "jabs.pose.points",
                "path": "/jabs/pose/points",
                "axes": ["frame", "slot", "keypoint", "coord"],
                "dtype": "float32",
                "shape": [4, 1, 12, 2],
                "units": "pixel",
                "coord_order": "xy",
                "encoding": {"kind": "dense"},
                "missing": {"policy": "nan"},
                "skeleton": "jabs.mouse12",
            }
        ],
    }


@pytest.fixture
def sample_pose_file() -> PoseFile:
    """A four-frame, two-slot pose file with the four core pose components."""
    return build_sample_pose_file()


@pytest.fixture
def sample_pose_file_no_dimensions() -> PoseFile:
    """The same file with unknown frame dimensions, as a legacy conversion has."""
    return build_sample_pose_file(width=None, height=None)
