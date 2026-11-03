"""Tests for the ShapeDescriptors segmentation feature."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from jabs.feature_extraction.segmentation_features.moment_cache import MomentInfo
from jabs.feature_extraction.segmentation_features.shape_descriptors import ShapeDescriptors
from jabs.pose_estimation import PoseEstimation


def _reference_per_frame(
    poses: PoseEstimation, moments: MomentInfo, pixel_scale: float
) -> dict[str, npt.NDArray[np.floating]]:
    """Original per-frame loop from ShapeDescriptors.per_frame, kept as a reference."""
    n = poses.num_frames
    names = [
        "x", "y", "ellipse_w", "ellipse_l", "perimeter", "elongation", "rectangularity",
        "convexity", "solidity", "euler_number", "hole_area_ratio",
    ]  # fmt: skip
    v = {name: np.full(n, np.nan, dtype=np.float32) for name in names}
    m = moments.get_moment
    for frame in range(n):
        if np.isnan(m(frame, "m00")):
            continue
        v["x"][frame] = m(frame, "m10") / m(frame, "m00")
        v["y"][frame] = m(frame, "m01") / m(frame, "m00")
        a = m(frame, "m20") / m(frame, "m00") - np.square(v["x"][frame])
        b = 2 * (m(frame, "m11") / m(frame, "m00") - v["x"][frame] * v["y"][frame])
        c = m(frame, "m02") / m(frame, "m00") - np.square(v["y"][frame])
        v["ellipse_w"][frame] = 0.5 * np.sqrt(
            8 * (a + c - np.sqrt(np.square(b) + np.square(a - c)))
        )
        v["ellipse_l"][frame] = 0.5 * np.sqrt(
            8 * (a + c + np.sqrt(np.square(b) + np.square(a - c)))
        )
        contours = moments.get_trimmed_contours(frame)
        flags = moments.get_flags(frame)
        rect = cv2.minAreaRect(np.concatenate(contours, axis=0))
        perimeters = [cv2.arcLength(x, True) for x in contours]
        hull = cv2.convexHull(np.concatenate(contours, axis=0))
        hull_perimeter = cv2.arcLength(hull, True)
        hull_area = cv2.contourArea(hull)
        hole_areas = np.sum(
            [cv2.contourArea(contours[int(x)]) for x in np.where(flags[: len(contours)] == 0)[0]]
        )
        v["perimeter"][frame] = np.sum(perimeters) * pixel_scale
        v["elongation"][frame] = 1 - (np.min(rect[1]) / np.max(rect[1]))
        v["rectangularity"][frame] = m(frame, "m00") / (rect[1][0] * rect[1][1] * pixel_scale**2)
        v["convexity"][frame] = hull_perimeter / np.sum(perimeters)
        v["solidity"][frame] = m(frame, "m00") / (hull_area * pixel_scale**2)
        v["euler_number"][frame] = np.sum(flags[: len(contours)] == 1) - np.sum(
            flags[: len(contours)] == 0
        )
        v["hole_area_ratio"][frame] = (hole_areas * pixel_scale**2) / m(frame, "m00")
    v["centroid_speed"] = np.hypot(np.gradient(v.pop("x")), np.gradient(v.pop("y"))) * poses.fps
    return v


@pytest.mark.parametrize(
    "pixel_scale",
    [1.0, 0.07, np.float32(0.07928075), np.float64(0.07928075)],
    ids=["unit", "python_float", "numpy_float32", "numpy_float64"],
)
def test_matches_reference_implementation(seg_data: dict, pixel_scale: float) -> None:
    """Every shape descriptor equals the original per-frame implementation."""
    poses = seg_data["pose_est_v6"]
    for identity in poses.identities:
        moments = MomentInfo(poses, identity, pixel_scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = _reference_per_frame(poses, moments, pixel_scale)
            actual = ShapeDescriptors(poses, pixel_scale, moments).per_frame(identity)
        assert actual.keys() == expected.keys()
        for key, values in expected.items():
            np.testing.assert_array_equal(actual[key], values, err_msg=f"{identity} {key}")
