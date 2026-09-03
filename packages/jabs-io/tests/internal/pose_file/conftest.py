"""Shared fixtures for pose file format tests."""

import numpy as np
import pytest

from jabs.io.internal.pose_file.types import (
    Component,
    HistoryEntry,
    PoseFile,
    Provenance,
    ProvenanceRecord,
    Skeleton,
    VideoInfo,
)

MOUSE12 = Skeleton(
    body_parts=(
        "NOSE",
        "LEFT_EAR",
        "RIGHT_EAR",
        "BASE_NECK",
        "LEFT_FRONT_PAW",
        "RIGHT_FRONT_PAW",
        "CENTER_SPINE",
        "LEFT_REAR_PAW",
        "RIGHT_REAR_PAW",
        "BASE_TAIL",
        "MID_TAIL",
        "TIP_TAIL",
    ),
    edges=(
        (4, 6),
        (6, 5),
        (7, 9),
        (9, 8),
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 10),
        (10, 11),
        (1, 0),
        (0, 2),
    ),
    description="JABS 12-keypoint mouse skeleton",
)


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
                provenance="jabs.pose",
            ),
            Component(
                id="jabs.pose.point_valid",
                axes=("frame", "slot", "keypoint"),
                data=point_valid,
                missing={"policy": "none"},
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
