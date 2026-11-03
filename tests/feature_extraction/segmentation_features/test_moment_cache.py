"""Tests for MomentInfo image moment computation."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from jabs.feature_extraction.segmentation_features.moment_cache import (
    MomentInfo,
    _shift_raw_moments,
)


def _full_frame_moments(contour_list: list[npt.NDArray[np.int32]]) -> dict[str, float]:
    """Reference implementation: render the contours onto a full frame from (0, 0)."""
    frame_size = int(np.max(np.concatenate(contour_list))) + 1
    render = np.zeros((frame_size, frame_size, 1), dtype=np.uint8)
    cv2.drawContours(render, contour_list, -1, [1], -1)
    return cv2.moments(render)


def _assert_moments_match(actual: dict[str, float], expected: dict[str, float]) -> None:
    """Assert two cv2.moments dicts agree to within float64 rounding."""
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key


def _rect(x: int, y: int, w: int, h: int) -> npt.NDArray[np.int32]:
    """Rectangle contour with top-left corner (x, y)."""
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)


@pytest.mark.parametrize(
    "contours",
    [
        [_rect(0, 0, 10, 5)],
        [_rect(300, 120, 40, 18)],
        [_rect(500, 20, 12, 30), _rect(560, 70, 8, 8)],
        # nested contour, filled in as with the full-frame rendering
        [_rect(200, 200, 50, 50), _rect(210, 210, 10, 10)],
        [np.array([[400, 300], [460, 310], [430, 380]], dtype=np.int32)],
        [np.array([[250, 250]], dtype=np.int32)],
    ],
    ids=["origin", "offset", "two_blobs", "nested", "triangle", "single_point"],
)
def test_calculate_moments_matches_full_frame(contours: list[npt.NDArray[np.int32]]) -> None:
    """Bounding-box rendering gives the same moments as rendering the whole frame."""
    _assert_moments_match(MomentInfo.calculate_moments(contours), _full_frame_moments(contours))


def test_shift_raw_moments_zero_offset_is_identity() -> None:
    """Shifting by (0, 0) leaves the moments unchanged."""
    moments = cv2.moments(np.ones((4, 6), dtype=np.uint8))
    assert _shift_raw_moments(moments, 0.0, 0.0) == moments


def test_shift_raw_moments_leaves_central_moments_unchanged() -> None:
    """Central and normalized moments are translation invariant and are not modified."""
    moments = cv2.moments(np.ones((4, 6), dtype=np.uint8))
    shifted = _shift_raw_moments(moments, 7.0, 3.0)
    for key in moments:
        if key.startswith(("mu", "nu")):
            assert shifted[key] == moments[key]
    assert shifted["m00"] == moments["m00"]
    assert shifted["m10"] == pytest.approx(moments["m10"] + 7.0 * moments["m00"])


@pytest.mark.parametrize("pixel_scale", [1.0, 0.05])
def test_moment_info_matches_full_frame_reference(seg_data: dict, pixel_scale: float) -> None:
    """MomentInfo stores full-frame moments scaled to cm, and NaN where there is no mask."""
    info = MomentInfo(seg_data["pose_est_v6"], 1, pixel_scale)
    keys = list(cv2.moments(np.empty(0)))
    scale = np.array([pixel_scale ** info.get_pixel_power(k) for k in keys])

    frames_with_contours = 0
    for frame in range(info._moments.shape[0]):
        contours = info.get_trimmed_contours(frame)
        if len(contours) == 0:
            assert np.all(np.isnan(info._moments[frame]))
            continue
        frames_with_contours += 1
        reference = _full_frame_moments(contours)
        expected = np.array([reference[k] for k in keys]) * scale
        np.testing.assert_allclose(info._moments[frame], expected, rtol=1e-5, atol=1e-12)
    assert frames_with_contours > 0
