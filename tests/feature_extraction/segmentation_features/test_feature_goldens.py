"""The segmentation-dependent features on the sample v6 file match their goldens.

The goldens were recorded with the legacy reader (see ``feature_goldens.py``). The
unified ``Pose`` over a conversion of the same file must match them except where spec §2
(docs/superpowers/specs/2026-11-20-jabs-adopts-pose-file-design.md), the global return-dtype
rule, or an explicit ``FEATURE_VERSION`` change modifies a feature. Each allowed difference is
named below; nothing else may differ.

Allowed differences:

* ``closest_fov_angles`` features agree to 1e-3 (degrees, and their sine and
  cosine) instead of float rounding: ``Pose`` returns float32 points, legacy
  readers float64 (global constraint "Return dtypes after the flip"), and the
  bearings and angles derived from them differ by about 1e-4 degrees.
* ``nu*`` and ``hu_moments`` (``FEATURE_VERSION = 21``): the goldens recorded
  normalized central moments ``nu_{p,q}`` scaled by ``PIXEL_SCALE**(p + q + 2)``,
  which also scaled ``HuMoments``. ``MomentInfo.get_pixel_power`` now returns
  ``0`` for dimensionless ``nu*`` keys. Expected unscaled ``nu*`` and
  ``HuMoments`` values are derived directly from the legacy golden ``nu*`` by
  dividing out ``PIXEL_SCALE**(p + q + 2)`` and recomputing ``cv2.HuMoments``.
* ``moments/{mu,nu}*`` features (``FEATURE_VERSION = 21``): the goldens recorded
  raw image-frame central and normalized central moments; ``Moments`` now rotates
  those 14 moments into the animal's bearing frame. Rather than applying the
  function under test to the goldens, ``test_moment_cache_matches_legacy_goldens``
  checks raw ``MomentInfo`` directly against the legacy goldens, and
  ``test_rotated_moments_preserve_golden_invariants_and_nan_mask`` checks
  independent rotation invariants (trace, determinant, Hu moments, and NaN mask)
  on the rotated output. Independent accuracy of the rotation itself is verified
  in ``test_moments.py``.
* No other difference occurs on the recorded frames. The converter places
  contours in the instance's pose slot (spec §2 row "Segmentation pairing")
  where the legacy reader paired them by ``longterm_seg_id``; that moves the
  contours of 8 identity-frames of the sample, none of them on a recorded frame.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from feature_goldens import FRAME_STRIDE, PIXEL_SCALE, compute_features, load_golden

from jabs.feature_extraction.segmentation_features.moment_cache import MomentInfo
from jabs.pose_estimation.pose import Pose

# absolute tolerance of the closest_fov_angles features under Pose (float32 points)
_FLOAT32_ANGLE_ATOL = 1e-3
_MOMENT_ORDERS: tuple[str, ...] = ("20", "11", "02", "30", "21", "12", "03")
_NU_LEGACY_SCALE: dict[str, float] = {
    f"nu{pq}": PIXEL_SCALE ** (int(pq[0]) + int(pq[1]) + 2) for pq in _MOMENT_ORDERS
}


@pytest.fixture(scope="module")
def computed(seg_data: dict[str, Any]) -> dict[str, npt.NDArray[np.float32]]:
    """Features computed on the sample v6 file, keyed like the goldens."""
    return compute_features(seg_data["pose_est_v6"])


def _rtol(key: str) -> float:
    """Relative tolerance for one golden feature."""
    if "/segmentation/hu_moments/" in key and key.endswith(("/hu5", "/hu7")):
        return 1e-3
    return 1e-5


def _atol(key: str, unified: bool) -> float:
    """Absolute tolerance for one golden feature."""
    if "/segmentation/hu_moments/" in key:
        return 0.0
    if unified and "/closest_fov_angles/" in key:
        return _FLOAT32_ANGLE_ATOL
    return 1e-6


def _is_rotated_moment_key(key: str) -> bool:
    """Return True if ``key`` is one of the 14 bearing-rotated ``moments/{mu,nu}*`` features."""
    parts = key.split("/")
    return len(parts) == 4 and parts[2] == "moments" and parts[3].startswith(("mu", "nu"))


def _unscaled_golden_nu(
    golden: dict[str, npt.NDArray[Any]], identity: int
) -> dict[str, npt.NDArray[np.float64]]:
    """Undo the legacy ``PIXEL_SCALE**(p+q+2)`` factor on golden ``nu*`` arrays."""
    return {
        name: np.asarray(golden[f"{identity}/segmentation/moments/{name}"], dtype=np.float64)
        / scale
        for name, scale in _NU_LEGACY_SCALE.items()
    }


def _hu_from_nu_arrays(
    nu_by_name: dict[str, npt.NDArray[np.floating]], m00: npt.NDArray[np.floating]
) -> dict[str, npt.NDArray[np.float32]]:
    """Compute per-frame ``hu1..hu7`` directly from ``nu*`` arrays using ``cv2.HuMoments``."""
    n_frames = len(m00)
    out = {f"hu{i}": np.zeros(n_frames, dtype=np.float32) for i in range(1, 8)}
    for frame in range(n_frames):
        if m00[frame] == 0:
            continue
        moments_dict = {name: float(arr[frame]) for name, arr in nu_by_name.items()}
        hu = cv2.HuMoments(moments_dict).ravel()
        for i in range(7):
            out[f"hu{i + 1}"][frame] = hu[i]
    return out


def test_feature_keys_match_golden(computed: dict[str, npt.NDArray[np.float32]]) -> None:
    """Every recorded feature is computed, and nothing else."""
    assert sorted(computed) == sorted(load_golden())


def test_features_match_golden(
    seg_data: dict[str, Any], computed: dict[str, npt.NDArray[np.float32]]
) -> None:
    """Every non-rotated feature matches its legacy golden (with unscaled Hu moments)."""
    pose: Pose = seg_data["pose_est_v6"]
    unified = isinstance(pose, Pose)
    golden = load_golden()

    expected_hu: dict[str, npt.NDArray[np.float32]] = {}
    for identity in pose.identities:
        m00 = golden[f"{identity}/segmentation/moments/m00"]
        nu_unscaled = _unscaled_golden_nu(golden, identity)
        for hu_name, arr in _hu_from_nu_arrays(nu_unscaled, m00).items():
            expected_hu[f"{identity}/segmentation/hu_moments/{hu_name}"] = arr

    mismatched = []
    for key, ref in golden.items():
        if _is_rotated_moment_key(key):
            continue
        target = expected_hu.get(key, ref)
        if not np.allclose(
            computed[key], target, rtol=_rtol(key), atol=_atol(key, unified), equal_nan=True
        ):
            mismatched.append(key)
    assert not mismatched, mismatched


def test_moment_cache_matches_legacy_goldens(seg_data: dict[str, Any]) -> None:
    """Raw MomentInfo agrees directly with legacy goldens (m00/mu exact, nu unscaled)."""
    pose: Pose = seg_data["pose_est_v6"]
    golden = load_golden()

    for identity in pose.identities:
        cache = MomentInfo(pose, identity, PIXEL_SCALE)
        m00_actual = cache.get_moment_array("m00")[::FRAME_STRIDE]
        np.testing.assert_allclose(
            m00_actual,
            golden[f"{identity}/segmentation/moments/m00"],
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        )
        for pq in _MOMENT_ORDERS:
            mu_name = f"mu{pq}"
            np.testing.assert_allclose(
                cache.get_moment_array(mu_name)[::FRAME_STRIDE],
                golden[f"{identity}/segmentation/moments/{mu_name}"],
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
                err_msg=f"{identity}/{mu_name}",
            )
            nu_name = f"nu{pq}"
            expected_nu = (
                golden[f"{identity}/segmentation/moments/{nu_name}"].astype(np.float64)
                / (_NU_LEGACY_SCALE[nu_name])
            )
            np.testing.assert_allclose(
                cache.get_moment_array(nu_name)[::FRAME_STRIDE],
                expected_nu,
                rtol=1e-5,
                atol=0.0,
                equal_nan=True,
                err_msg=f"{identity}/{nu_name}",
            )


def test_rotated_moments_preserve_golden_invariants_and_nan_mask(
    seg_data: dict[str, Any], computed: dict[str, npt.NDArray[np.float32]]
) -> None:
    """Bearing-rotated moments preserve trace, determinant, Hu moments, and NaN semantics."""
    pose: Pose = seg_data["pose_est_v6"]
    golden = load_golden()

    for identity in pose.identities:
        bearings = pose.compute_all_bearings(identity)[::FRAME_STRIDE]
        g_m00 = golden[f"{identity}/segmentation/moments/m00"]
        g_mu20 = golden[f"{identity}/segmentation/moments/mu20"].astype(np.float64)
        g_mu11 = golden[f"{identity}/segmentation/moments/mu11"].astype(np.float64)
        g_mu02 = golden[f"{identity}/segmentation/moments/mu02"].astype(np.float64)
        g_nu = _unscaled_golden_nu(golden, identity)

        c_mu20 = computed[f"{identity}/segmentation/moments/mu20"].astype(np.float64)
        c_mu11 = computed[f"{identity}/segmentation/moments/mu11"].astype(np.float64)
        c_mu02 = computed[f"{identity}/segmentation/moments/mu02"].astype(np.float64)
        c_nu = {
            f"nu{pq}": computed[f"{identity}/segmentation/moments/nu{pq}"].astype(np.float64)
            for pq in _MOMENT_ORDERS
        }

        expected_nan = np.isnan(g_m00) | np.isnan(bearings)
        valid = ~expected_nan
        assert valid.any()

        for prefix in ("mu", "nu"):
            for pq in _MOMENT_ORDERS:
                arr = computed[f"{identity}/segmentation/moments/{prefix}{pq}"]
                np.testing.assert_array_equal(
                    np.isnan(arr), expected_nan, err_msg=f"{identity}/{prefix}{pq} NaN mask"
                )

        # 2nd-order rotation invariants: trace and determinant (independent of rotate_central_moments)
        np.testing.assert_allclose(
            c_mu20[valid] + c_mu02[valid],
            g_mu20[valid] + g_mu02[valid],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            c_mu20[valid] * c_mu02[valid] - c_mu11[valid] ** 2,
            g_mu20[valid] * g_mu02[valid] - g_mu11[valid] ** 2,
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            c_nu["nu20"][valid] + c_nu["nu02"][valid],
            g_nu["nu20"][valid] + g_nu["nu02"][valid],
            rtol=1e-5,
            atol=0.0,
        )

        # All 7 Hu moments recomputed from rotated float32 nu* must match unscaled golden Hu
        # moments (hu5/hu7 are degree-4 polynomials of nu3*, so float32 rounding is ~1e-3 rel).
        hu_from_rotated = _hu_from_nu_arrays(c_nu, g_m00)
        hu_from_golden = _hu_from_nu_arrays(g_nu, g_m00)
        for i in range(1, 8):
            key = f"hu{i}"
            rtol = 1e-3 if i in (5, 7) else 1e-4
            np.testing.assert_allclose(
                hu_from_rotated[key][valid],
                hu_from_golden[key][valid],
                rtol=rtol,
                atol=0.0,
                err_msg=f"{identity}/{key}",
            )
