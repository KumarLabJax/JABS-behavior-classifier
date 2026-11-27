"""Tests for HuMoments segmentation features."""

from __future__ import annotations

from typing import Any

import numpy as np

from jabs.feature_extraction.segmentation_features import HuMoments
from jabs.feature_extraction.segmentation_features.moment_cache import MomentInfo


def test_hu_moment_feature_name(seg_data: dict[str, Any]) -> None:
    """Test HuMoment class.

    Verifies that the HuMoments feature has the correct feature names, returns the
    expected number of features per frame, and fills them in for frames with a shape.
    """
    # test that data was read and setup correctly
    hu_moments_feature = seg_data["feature_mods"]["hu_moments"]

    assert hu_moments_feature._feature_names[-2] == "hu6"

    # per_frame ignores its identity argument: the fixture bound these feature modules to
    # identity 1's moment cache, so identity 1 is also the one to read m00 for below.
    identity = 1

    hu_moments_by_frame = hu_moments_feature.per_frame(identity)

    assert len(hu_moments_by_frame) == 7

    # hu1 is nu20 + nu02, which is positive for any shape with area, so a feature left at
    # its zero fill or filled with NaN fails here.
    m00 = seg_data["feature_mods"]["moments"].per_frame(identity)["m00"]
    has_shape = m00 > 0
    assert has_shape.any()
    hu1 = hu_moments_by_frame["hu1"][has_shape]
    assert np.isfinite(hu1).all()
    assert (hu1 != 0).all()


def test_hu_moments_are_invariant_to_pixel_scale(seg_data: dict[str, Any]) -> None:
    """HuMoments are dimensionless and identical across different cm_per_pixel scales."""
    pose = seg_data["pose_est_v6"]
    unit_cache = MomentInfo(pose, 1, 1.0)
    unit_hu = HuMoments(pose, 1.0, unit_cache).per_frame(1)

    for scale in (0.05, 0.02):
        scaled_cache = MomentInfo(pose, 1, scale)
        scaled_hu = HuMoments(pose, scale, scaled_cache).per_frame(1)
        for key in unit_hu:
            np.testing.assert_allclose(
                scaled_hu[key],
                unit_hu[key],
                rtol=1e-5,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"{key} changed with pixel_scale={scale}",
            )
