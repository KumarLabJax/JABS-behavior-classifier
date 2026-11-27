"""Tests for heading-aligned (rotation-invariant) segmentation Moments features."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from jabs.core.types import SegmentationWindow
from jabs.feature_extraction.segmentation_features import Moments
from jabs.feature_extraction.segmentation_features.moment_cache import MomentInfo
from jabs.feature_extraction.segmentation_features.moments import rotate_central_moments

_ROTATED_MOMENT_KEYS: tuple[str, ...] = (
    "mu20",
    "mu11",
    "mu02",
    "mu30",
    "mu21",
    "mu12",
    "mu03",
    "nu20",
    "nu11",
    "nu02",
    "nu30",
    "nu21",
    "nu12",
    "nu03",
)
_MOMENT_EXPONENTS: tuple[tuple[str, int, int], ...] = (
    ("20", 2, 0),
    ("11", 1, 1),
    ("02", 0, 2),
    ("30", 3, 0),
    ("21", 2, 1),
    ("12", 1, 2),
    ("03", 0, 3),
)


def _asymmetric_rectilinear_contour() -> npt.NDArray[np.int32]:
    """Return an asymmetric axis-aligned stepped polygon centered near (60, 60).

    All edges are horizontal or vertical so cardinal 90-degree rotations map onto the
    exact same integer pixel lattice without Bresenham diagonal rounding drift, while
    breaking both x- and y-symmetry so every 2nd- and 3rd-order central moment is nonzero.
    """
    return np.array(
        [
            [20, 46],
            [70, 46],
            [70, 34],
            [102, 34],
            [102, 68],
            [62, 68],
            [62, 74],
            [20, 74],
        ],
        dtype=np.int32,
    )


def _asymmetric_silhouette_contour() -> npt.NDArray[np.int32]:
    """Return a diagonal-edged asymmetric mouse-like polygon centered near (60, 60)."""
    return np.array(
        [
            [25, 58],
            [45, 42],
            [78, 38],
            [98, 52],
            [92, 66],
            [68, 72],
            [40, 66],
        ],
        dtype=np.int32,
    )


def _rotate_contour_cardinal(
    contour: npt.NDArray[np.int32], k_90: int, center: tuple[int, int] = (200, 200)
) -> npt.NDArray[np.int32]:
    """Rotate an integer contour by ``k_90 * 90`` degrees in y-down image coordinates.

    Matches a 90-degree camera/rig rotation (for example, DAX2 bottom-hopper vs.
    DAX3 right-hopper layouts).
    """
    cx, cy = center
    pts = contour.astype(np.int64) - np.array([60, 60], dtype=np.int64)
    k = k_90 % 4
    dx, dy = pts[:, 0], pts[:, 1]
    if k == 0:
        rx, ry = dx, dy
    elif k == 1:  # +90 deg: (x, y) -> (-y, x)
        rx, ry = -dy, dx
    elif k == 2:  # +180 deg: (x, y) -> (-x, -y)
        rx, ry = -dx, -dy
    else:  # +270 deg: (x, y) -> (y, -x)
        rx, ry = dy, -dx
    return np.column_stack([rx + cx, ry + cy]).astype(np.int32)


def _direct_pixel_rotated_moments(
    contour_list: list[npt.NDArray[np.int32]],
    bearing_deg: float,
    pixel_scale: float,
) -> dict[str, float]:
    """Compute heading-aligned moments directly by rotating rasterized mask pixel coordinates.

    Independent ground-truth reference that does not call ``rotate_central_moments``.
    """
    points = np.concatenate(contour_list).reshape(-1, 2)
    x0, y0 = points.min(axis=0)
    width, height = points.max(axis=0) - (x0, y0) + 1
    render = np.zeros((int(height), int(width), 1), dtype=np.uint8)
    cv2.drawContours(render, contour_list, -1, [1], -1, offset=(-int(x0), -int(y0)))

    ys, xs = np.nonzero(render[:, :, 0])
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    m00_px = float(len(x))
    dx = x - x.mean()
    dy = y - y.mean()

    rad = np.deg2rad(float(bearing_deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    xp = c * dx + s * dy
    yp = -s * dx + c * dy

    result: dict[str, float] = {"m00": m00_px * (pixel_scale**2)}
    for pq, p, q in _MOMENT_EXPONENTS:
        mu_px = float(np.sum((xp**p) * (yp**q)))
        result[f"mu{pq}"] = mu_px * (pixel_scale ** (p + q + 2))
        result[f"nu{pq}"] = mu_px / (m00_px ** (1.0 + 0.5 * (p + q)))
    return result


class _StubSegmentationPose:
    """Minimal PoseLike stub exposing bearings and segmentation windows for Moments tests."""

    def __init__(
        self,
        contours_per_frame: list[list[npt.NDArray[np.int32]]],
        bearings_deg: npt.NDArray[np.float32],
    ) -> None:
        self._contours = contours_per_frame
        self._bearings = np.asarray(bearings_deg, dtype=np.float32)
        self.num_frames = len(contours_per_frame)
        self.fps = 30.0

    def compute_all_bearings(self, identity: int) -> npt.NDArray[np.float32]:
        del identity
        return self._bearings.copy()

    def iter_segmentation(
        self, identity: int, window_size: int | None = None
    ) -> Iterator[SegmentationWindow]:
        del identity, window_size
        max_c = max((len(c) for c in self._contours), default=1)
        max_c = max(max_c, 1)
        max_p = max(
            (len(poly) for frame_c in self._contours for poly in frame_c),
            default=1,
        )
        max_p = max(max_p, 1)
        contours = np.zeros((self.num_frames, max_c, max_p, 2), dtype=np.int32)
        contour_count = np.zeros(self.num_frames, dtype=np.uint32)
        contour_length = np.zeros((self.num_frames, max_c), dtype=np.uint32)
        for f, frame_c in enumerate(self._contours):
            contour_count[f] = len(frame_c)
            for c_idx, poly in enumerate(frame_c):
                contour_length[f, c_idx] = len(poly)
                contours[f, c_idx, : len(poly)] = poly
        yield SegmentationWindow(
            contours=contours,
            contour_count=contour_count,
            contour_length=contour_length,
            external_flag=None,
            start=0,
        )


def test_create_moment(seg_data: dict[str, Any]) -> None:
    """Test Moments feature class initialization.

    Verifies that the Moments feature can be properly initialized with
    all expected moment keys.
    """
    pixel_scale = 1.0
    moments_feature = Moments(seg_data["pose_est_v6"], pixel_scale, seg_data["moment_cache"])

    assert moments_feature._name == "moments"

    moment_keys = {
        "m00",
        *_ROTATED_MOMENT_KEYS,
    }

    assert moment_keys - set(moments_feature._moments_to_use) == set()


def test_moments_per_frame(seg_data: dict[str, Any]) -> None:
    """Test per-frame moment computation on real sample pose data."""
    moments_feature = seg_data["feature_mods"]["moments"]
    moment_values = moments_feature.per_frame(1)
    assert len(moment_values["m00"]) == seg_data["pose_est_v6"].num_frames
    assert set(moment_values.keys()) == {"m00", *_ROTATED_MOMENT_KEYS}
    assert list(moment_values) == moments_feature._moments_to_use
    for arr in moment_values.values():
        assert arr.dtype == np.float32


def test_rotate_central_moments_zero_angle_is_identity() -> None:
    """Rotating by 0 degrees leaves all central moments unchanged."""
    raw = MomentInfo.calculate_moments([_asymmetric_silhouette_contour()])
    inputs = [np.array([raw[k]], dtype=np.float32) for k in _ROTATED_MOMENT_KEYS[:7]]
    rotated = rotate_central_moments(*inputs, bearing_deg=np.array([0.0], dtype=np.float32))
    for inp, out in zip(inputs, rotated, strict=True):
        np.testing.assert_allclose(out, inp, rtol=1e-6, atol=1e-9)


def test_rotate_central_moments_exact_cardinal_rotations() -> None:
    """Cardinal 0/90/180/270-degree rotations produce identical heading-aligned moments."""
    base_contour = _asymmetric_rectilinear_contour()
    base_moments = MomentInfo.calculate_moments([_rotate_contour_cardinal(base_contour, 0)])
    # Confirm the test shape actually has nontrivial asymmetric 2nd & 3rd moments
    assert abs(base_moments["mu20"] - base_moments["mu02"]) > 100.0
    assert abs(base_moments["mu11"]) > 100.0
    assert abs(base_moments["mu30"]) > 100.0
    assert abs(base_moments["mu03"]) > 100.0

    for k in (1, 2, 3):
        rot_contour = _rotate_contour_cardinal(base_contour, k)
        rot_m = MomentInfo.calculate_moments([rot_contour])
        angle = np.array([90.0 * k], dtype=np.float32)

        for prefix in ("mu", "nu"):
            keys = [f"{prefix}{pq}" for pq in ("20", "11", "02", "30", "21", "12", "03")]
            aligned = rotate_central_moments(
                *[np.array([rot_m[key]], dtype=np.float32) for key in keys],
                bearing_deg=angle,
            )
            expected = [base_moments[key] for key in keys]
            for key, actual_arr, exp_val in zip(keys, aligned, expected, strict=True):
                assert actual_arr[0] == pytest.approx(exp_val, rel=1e-5, abs=1e-6), (
                    f"{key} mismatch at {90 * k} deg"
                )


@pytest.mark.parametrize("theta_deg", [23.0, 45.0, 90.0, 137.5, -62.0, -145.0])
def test_rotate_central_moments_arbitrary_angles_on_point_cloud(theta_deg: float) -> None:
    """Analytical rotation recovers canonical body-frame moments for arbitrary continuous angles."""
    rng = np.random.default_rng(42)
    # Asymmetric continuous 2D cloud (stretched along x, skewed in x and y)
    x0 = rng.normal(loc=0.0, scale=4.0, size=2000) + 0.35 * rng.exponential(scale=2.0, size=2000)
    y0 = rng.normal(loc=0.0, scale=1.5, size=2000) + 0.12 * x0**2
    x0 = x0 - x0.mean()
    y0 = y0 - y0.mean()

    def _central_moments(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> list[float]:
        return [
            float(np.sum(x**2)),
            float(np.sum(x * y)),
            float(np.sum(y**2)),
            float(np.sum(x**3)),
            float(np.sum((x**2) * y)),
            float(np.sum(x * (y**2))),
            float(np.sum(y**3)),
        ]

    canonical = _central_moments(x0, y0)
    rad = np.deg2rad(theta_deg)
    c, s = np.cos(rad), np.sin(rad)
    # Rotate body cloud by +theta_deg into image frame
    xr = c * x0 - s * y0
    yr = s * x0 + c * y0
    rot_m = _central_moments(xr, yr)

    recovered = rotate_central_moments(
        *[np.array([v], dtype=np.float64) for v in rot_m],
        bearing_deg=np.array([theta_deg], dtype=np.float64),
    )
    for rec, exp in zip(recovered, canonical, strict=True):
        assert float(rec[0]) == pytest.approx(exp, rel=1e-5, abs=1e-4)


@pytest.mark.parametrize("theta_deg", [15.0, 45.0, 90.0, 143.0, -77.0])
def test_rotate_central_moments_preserves_invariants_and_hu_moments(theta_deg: float) -> None:
    """Rotation preserves trace, determinant, and all seven Hu moments identically."""
    raw = MomentInfo.calculate_moments([_asymmetric_silhouette_contour()])
    angle = np.array([theta_deg], dtype=np.float64)

    mu_rot = rotate_central_moments(
        *[
            np.array([raw[k]], dtype=np.float64)
            for k in ("mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03")
        ],
        bearing_deg=angle,
    )
    nu_rot = rotate_central_moments(
        *[
            np.array([raw[k]], dtype=np.float64)
            for k in ("nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03")
        ],
        bearing_deg=angle,
    )

    # 2nd-order matrix invariants: trace and determinant
    assert float(mu_rot[0][0] + mu_rot[2][0]) == pytest.approx(raw["mu20"] + raw["mu02"], rel=1e-5)
    assert float(mu_rot[0][0] * mu_rot[2][0] - mu_rot[1][0] ** 2) == pytest.approx(
        raw["mu20"] * raw["mu02"] - raw["mu11"] ** 2, rel=1e-5
    )

    # All 7 Hu moments must match raw image-frame Hu moments
    rot_dict = dict(raw)
    for k, arr in zip(
        ("nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"), nu_rot, strict=True
    ):
        rot_dict[k] = float(arr[0])
    hu_orig = cv2.HuMoments(raw).ravel()
    hu_rot = cv2.HuMoments(rot_dict).ravel()
    np.testing.assert_allclose(hu_rot, hu_orig, rtol=1e-5, atol=1e-12)


def test_rotate_central_moments_left_right_reflection_parity() -> None:
    """Mirroring across the animal's heading axis keeps even-q moments and negates odd-q moments."""
    base = _asymmetric_silhouette_contour()
    # Mirror y around center y=60 while keeping heading along +x (theta = 0)
    mirrored = base.copy()
    mirrored[:, 1] = 120 - mirrored[:, 1]

    m_base = MomentInfo.calculate_moments([base])
    m_mir = MomentInfo.calculate_moments([mirrored])

    even_q = {"20", "02", "30", "12"}
    odd_q = {"11", "21", "03"}
    for prefix in ("mu", "nu"):
        keys = [f"{prefix}{pq}" for pq in ("20", "11", "02", "30", "21", "12", "03")]
        r_base = rotate_central_moments(
            *[np.array([m_base[k]], dtype=np.float32) for k in keys],
            bearing_deg=np.array([0.0], dtype=np.float32),
        )
        r_mir = rotate_central_moments(
            *[np.array([m_mir[k]], dtype=np.float32) for k in keys],
            bearing_deg=np.array([0.0], dtype=np.float32),
        )
        for pq, vb, vm in zip(
            ("20", "11", "02", "30", "21", "12", "03"), r_base, r_mir, strict=True
        ):
            if pq in even_q:
                assert float(vm[0]) == pytest.approx(float(vb[0]), rel=1e-5, abs=1e-6)
            elif pq in odd_q:
                assert float(vm[0]) == pytest.approx(-float(vb[0]), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("pixel_scale", [1.0, 0.02])
def test_moments_per_frame_dax2_vs_dax3_90deg_rotation_invariance(pixel_scale: float) -> None:
    """End-to-end Moments.per_frame is invariant to 0/90/180/270 deg arena rotations."""
    base = _asymmetric_rectilinear_contour()
    contours_per_frame = [[_rotate_contour_cardinal(base, k)] for k in range(4)]
    bearings = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)

    stub_pose = _StubSegmentationPose(contours_per_frame, bearings)
    moment_cache = MomentInfo(stub_pose, identity=0, pixel_scale=pixel_scale)
    values = Moments(stub_pose, pixel_scale, moment_cache).per_frame(identity=0)

    # Every moment (m00, mu*, nu*) must agree across all four cardinal orientations
    for key, arr in values.items():
        assert np.all(np.isfinite(arr)), f"{key} has non-finite values"
        np.testing.assert_allclose(
            arr, np.full(4, arr[0], dtype=np.float32), rtol=1e-5, atol=1e-6, err_msg=key
        )


def test_moments_per_frame_nu_is_invariant_to_pixel_scale() -> None:
    """Rotated nu* moments are dimensionless and identical across cm_per_pixel scales."""
    contours_per_frame = [[_asymmetric_silhouette_contour()]]
    bearings = np.array([37.0], dtype=np.float32)
    stub_pose = _StubSegmentationPose(contours_per_frame, bearings)

    vals_by_scale: dict[float, dict[str, npt.NDArray[np.float32]]] = {}
    for scale in (1.0, 0.05, 0.02):
        cache = MomentInfo(stub_pose, identity=0, pixel_scale=scale)
        vals_by_scale[scale] = Moments(stub_pose, scale, cache).per_frame(identity=0)

    ref = vals_by_scale[1.0]
    for scale in (0.05, 0.02):
        cur = vals_by_scale[scale]
        assert float(cur["m00"][0]) == pytest.approx(float(ref["m00"][0]) * scale**2, rel=1e-5)
        for pq, p, q in _MOMENT_EXPONENTS:
            assert float(cur[f"mu{pq}"][0]) == pytest.approx(
                float(ref[f"mu{pq}"][0]) * scale ** (p + q + 2), rel=1e-5, abs=1e-9
            )
            assert float(cur[f"nu{pq}"][0]) == pytest.approx(
                float(ref[f"nu{pq}"][0]), rel=1e-5, abs=1e-9
            )


def test_moments_per_frame_nan_bearing_and_missing_contours() -> None:
    """Missing bearing keeps m00 finite and sets rotated mu*/nu* to NaN; missing mask sets all NaN."""
    base = _asymmetric_silhouette_contour()
    contours_per_frame = [
        [base],  # frame 0: valid contour + valid bearing
        [base],  # frame 1: valid contour + NaN bearing
        [],  # frame 2: no contour + valid bearing
    ]
    bearings = np.array([35.0, np.nan, 35.0], dtype=np.float32)

    stub_pose = _StubSegmentationPose(contours_per_frame, bearings)
    moment_cache = MomentInfo(stub_pose, identity=0, pixel_scale=1.0)
    values = Moments(stub_pose, 1.0, moment_cache).per_frame(identity=0)

    # Frame 0: all finite
    for key, arr in values.items():
        assert np.isfinite(arr[0]), f"frame 0 {key} should be finite"

    # Frame 1: m00 stays finite and equal to frame 0; rotated mu*/nu* are NaN
    assert np.isfinite(values["m00"][1])
    assert values["m00"][1] == pytest.approx(values["m00"][0])
    for key in _ROTATED_MOMENT_KEYS:
        assert np.isnan(values[key][1]), f"frame 1 {key} should be NaN when bearing is NaN"

    # Frame 2: all NaN (no segmentation contour)
    for key, arr in values.items():
        assert np.isnan(arr[2]), f"frame 2 {key} should be NaN without contour"


def test_moments_per_frame_matches_direct_pixel_rotation_on_real_pose_data(
    seg_data: dict[str, Any],
) -> None:
    """Moments.per_frame matches direct rotated-pixel summation on real sample contours."""
    pose = seg_data["pose_est_v6"]
    pixel_scale = 0.05
    moment_cache = MomentInfo(pose, identity=1, pixel_scale=pixel_scale)
    values = Moments(pose, pixel_scale, moment_cache).per_frame(1)
    bearings = pose.compute_all_bearings(1)

    # Check a stride of real segmented frames with finite bearing against independent pixel sums
    checked = 0
    for frame in range(0, pose.num_frames, 120):
        contours = moment_cache.get_trimmed_contours(frame)
        if len(contours) == 0 or np.isnan(bearings[frame]):
            continue
        expected = _direct_pixel_rotated_moments(contours, float(bearings[frame]), pixel_scale)
        for key, exp_val in expected.items():
            assert float(values[key][frame]) == pytest.approx(exp_val, rel=1e-4, abs=1e-6), (
                f"frame {frame} {key}"
            )
        checked += 1
    assert checked >= 10
