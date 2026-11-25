"""The segmentation-dependent features on the sample v6 file match their goldens.

The goldens were recorded with the legacy reader (see ``feature_goldens.py``). The
unified ``Pose`` over a conversion of the same file must match them except where spec §2 (docs/superpowers/specs/
2026-11-20-jabs-adopts-pose-file-design.md), the global return-dtype rule, or an explicit
``FEATURE_VERSION`` change modifies a feature. Each allowed difference is named below;
nothing else may differ.

Allowed differences:

* ``closest_fov_angles`` features agree to 1e-3 (degrees, and their sine and
  cosine) instead of float rounding: ``Pose`` returns float32 points, legacy
  readers float64 (global constraint "Return dtypes after the flip"), and the
  bearings and angles derived from them differ by about 1e-4 degrees.
* ``moments/{mu,nu}*`` features (``FEATURE_VERSION = 21``): the goldens recorded
  raw image-frame central and normalized central moments; ``Moments`` now rotates
  those 14 moments into the animal's bearing frame. Expected golden values are
  rotated analytically by ``pose.compute_all_bearings(identity)[::FRAME_STRIDE]``
  so the golden check still verifies every frame and moment against the legacy
  reference (while ``m00``, ``hu_moments``, ``shape_descriptor``, and all social
  features remain exact against the unrotated goldens).
* No other difference occurs on the recorded frames. The converter places
  contours in the instance's pose slot (spec §2 row "Segmentation pairing")
  where the legacy reader paired them by ``longterm_seg_id``; that moves the
  contours of 8 identity-frames of the sample, none of them on a recorded frame.
"""

import numpy as np
import pytest
from feature_goldens import FRAME_STRIDE, compute_features, load_golden

from jabs.feature_extraction.segmentation_features.moments import rotate_central_moments
from jabs.pose_estimation.pose import Pose

# absolute tolerance of the closest_fov_angles features under Pose (float32 points)
_FLOAT32_ANGLE_ATOL = 1e-3
_MOMENT_ORDERS = ("20", "11", "02", "30", "21", "12", "03")


@pytest.fixture(scope="module")
def computed(seg_data: dict) -> dict[str, np.ndarray]:
    """Features computed on the sample v6 file, keyed like the goldens."""
    return compute_features(seg_data["pose_est_v6"])


def _atol(key: str, unified: bool) -> float:
    """Absolute tolerance for one golden feature."""
    if unified and "/closest_fov_angles/" in key:
        return _FLOAT32_ANGLE_ATOL
    return 1e-6


def _expected_goldens(pose: Pose) -> dict[str, np.ndarray]:
    """Return golden feature values with legacy image-frame moments rotated into bearing frame."""
    expected = dict(load_golden())
    for identity in pose.identities:
        bearings = pose.compute_all_bearings(identity)[::FRAME_STRIDE]
        for prefix in ("mu", "nu"):
            keys = [f"{identity}/segmentation/moments/{prefix}{pq}" for pq in _MOMENT_ORDERS]
            rotated = rotate_central_moments(
                *[expected[k] for k in keys],
                bearing_deg=bearings,
            )
            for k, arr in zip(keys, rotated, strict=True):
                expected[k] = arr
    return expected


def test_feature_keys_match_golden(computed: dict[str, np.ndarray]) -> None:
    """Every recorded feature is computed, and nothing else."""
    assert sorted(computed) == sorted(load_golden())


def test_features_match_golden(seg_data: dict, computed: dict[str, np.ndarray]) -> None:
    """Each feature equals its recorded values, NaN where they were NaN."""
    pose = seg_data["pose_est_v6"]
    unified = isinstance(pose, Pose)
    expected = _expected_goldens(pose)
    mismatched = [
        key
        for key, ref in expected.items()
        if not np.allclose(computed[key], ref, rtol=1e-5, atol=_atol(key, unified), equal_nan=True)
    ]
    assert not mismatched, mismatched
