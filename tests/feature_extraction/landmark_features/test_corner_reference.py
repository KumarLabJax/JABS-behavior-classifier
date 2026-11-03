"""Check the vectorized corner features against the original per-frame implementation."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from shapely.geometry import Point

import jabs.feature_extraction.landmark_features.corner as corner_module
from jabs.core.utils import signed_angle_degrees
from jabs.pose_estimation import PoseEstimation


def _reference_features(
    poses: PoseEstimation, identity: int, pixel_scale: float
) -> dict[str, npt.NDArray[np.floating]]:
    """Original per-frame loop from CornerDistanceInfo.cache_features, kept as a reference."""
    n = poses.num_frames
    out = {
        "distance to corner": np.full(n, np.nan, dtype=np.float32),
        "distance to center": np.full(n, np.nan, dtype=np.float32),
        "distance to wall": np.full(n, np.nan, dtype=np.float32),
        "bearing to corner": np.full(n, np.nan, dtype=np.float32),
        "bearing to center": np.full(n, np.nan, dtype=np.float32),
        "all walls": np.full([n, 4], np.nan, dtype=np.float32),
        "closest corner": np.full(n, -1, dtype=np.int8),
    }
    hulls = poses.get_identity_convex_hulls(identity)
    idx = PoseEstimation.KeypointIndex
    corners = corner_module.CornerDistanceInfo.sort_points_clockwise(
        poses.static_objects["corners"]
    )
    arena_center_np = np.mean(corners, axis=0)
    arena_center = Point(arena_center_np[0], arena_center_np[1])
    for frame in range(n):
        points, _ = poses.get_points(frame, identity)
        if points is None:
            continue
        shape = hulls[frame]
        distance = float("inf")
        corner_coordinates = (0, 0)
        closest_idx = -1
        for i in range(4):
            d = shape.distance(Point(corners[i, 0], corners[i, 1]))
            if d < distance:
                distance = d
                corner_coordinates = (corners[i, 0], corners[i, 1])
                closest_idx = i
        neck = points[idx.BASE_NECK, :]
        nose = points[idx.NOSE, :]
        p1 = corners.astype(np.float32)
        p2 = np.roll(p1, 1, axis=0)
        centroid = np.asarray(shape.centroid.xy).squeeze()
        c3 = np.hstack([centroid, 0])
        p1_3 = np.hstack([p1, np.zeros((4, 1), dtype=np.float32)])
        p2_3 = np.hstack([p2, np.zeros((4, 1), dtype=np.float32)])
        wall = np.abs(np.cross(c3 - p1_3, p2_3 - p1_3))[:, 2]
        wall_cv2 = cv2.pointPolygonTest(corners.astype(np.float32), centroid, True)
        wall = np.abs(wall * (wall_cv2 / np.min(wall)))
        out["distance to corner"][frame] = distance * pixel_scale
        out["distance to center"][frame] = shape.distance(arena_center) * pixel_scale
        out["distance to wall"][frame] = wall_cv2 * pixel_scale
        out["all walls"][frame] = wall * pixel_scale
        out["bearing to corner"][frame] = signed_angle_degrees(nose, neck, corner_coordinates)
        out["bearing to center"][frame] = signed_angle_degrees(nose, neck, arena_center_np)
        out["closest corner"][frame] = closest_idx
    return out


@pytest.mark.parametrize(
    "pixel_scale",
    [1.0, 0.07, np.float32(0.07928075), np.float64(0.07928075)],
    ids=["unit", "python_float", "numpy_float32", "numpy_float64"],
)
def test_matches_reference_implementation(
    pose_est_v5_with_static_objects: PoseEstimation, pixel_scale: float
) -> None:
    """Every output equals the original per-frame implementation, for every identity."""
    poses = pose_est_v5_with_static_objects
    info = corner_module.CornerDistanceInfo(poses, pixel_scale)
    for identity in poses.identities:
        expected = _reference_features(poses, identity, pixel_scale)
        distances = info.get_distances(identity)
        bearings = info.get_bearings(identity)
        walls = info.get_wall_distances(identity)
        for key in ("distance to corner", "distance to center", "distance to wall"):
            np.testing.assert_array_equal(distances[key], expected[key], err_msg=key)
        for key in ("bearing to corner", "bearing to center"):
            np.testing.assert_array_equal(bearings[key], expected[key], err_msg=key)
        for i in range(4):
            np.testing.assert_array_equal(walls[f"wall_{i}"], expected["all walls"][:, i])
        np.testing.assert_array_equal(
            info.get_closest_corner(identity), expected["closest corner"]
        )


def test_present_frames_without_hull_are_nan(
    pose_est_v5_with_static_objects: PoseEstimation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present identity without a convex hull gets NaN instead of raising."""
    poses = pose_est_v5_with_static_objects
    identity = poses.identities[0]
    hulls = list(poses.get_identity_convex_hulls(identity))
    frame = int(np.flatnonzero(poses.identity_mask(identity))[0])
    hulls[frame] = None
    monkeypatch.setattr(poses, "get_identity_convex_hulls", lambda i: hulls)

    info = corner_module.CornerDistanceInfo(poses, 1.0)

    assert np.isnan(info.get_distances(identity)["distance to corner"][frame])
    assert np.isnan(info.get_bearings(identity)["bearing to corner"][frame])
    assert info.get_closest_corner(identity)[frame] == -1
    assert not np.isnan(info.get_distances(identity)["distance to corner"][frame + 1])


def test_identity_never_present(
    pose_est_v5_with_static_objects: PoseEstimation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identity with no present frames produces all-NaN features."""
    poses = pose_est_v5_with_static_objects
    monkeypatch.setattr(poses, "identity_mask", lambda i: np.zeros(poses.num_frames, dtype=bool))

    info = corner_module.CornerDistanceInfo(poses, 1.0)

    assert np.all(np.isnan(info.get_distances(0)["distance to wall"]))
    assert np.all(np.isnan(info.get_wall_distances(0)["wall_0"]))
    assert np.all(info.get_closest_corner(0) == -1)
