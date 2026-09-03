"""Shared fixtures for pose file format tests."""

import pytest


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
                "body_parts": [
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
                ],
                "edges": [
                    [4, 6],
                    [6, 5],
                    [7, 9],
                    [9, 8],
                    [0, 3],
                    [3, 6],
                    [6, 9],
                    [9, 10],
                    [10, 11],
                    [1, 0],
                    [0, 2],
                ],
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
