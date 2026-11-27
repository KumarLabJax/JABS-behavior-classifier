"""Heading-aligned (egocentric) segmentation image moment features."""

import logging
import typing

import numpy as np
import numpy.typing as npt

from jabs.core.enums import PoseCapability
from jabs.core.pose_protocol import PoseLike
from jabs.feature_extraction.feature_base_class import Feature

if typing.TYPE_CHECKING:
    from .moment_cache import MomentInfo

logger = logging.getLogger(__name__)

# The (p, q) orders of the rotated mu*/nu* moments, in rotate_central_moments' argument order.
_ROTATED_ORDERS: typing.Final[tuple[str, ...]] = ("20", "11", "02", "30", "21", "12", "03")
_ROTATED_PREFIXES: typing.Final[tuple[str, ...]] = ("mu", "nu")


def rotate_central_moments(
    mu20: npt.NDArray[np.floating],
    mu11: npt.NDArray[np.floating],
    mu02: npt.NDArray[np.floating],
    mu30: npt.NDArray[np.floating],
    mu21: npt.NDArray[np.floating],
    mu12: npt.NDArray[np.floating],
    mu03: npt.NDArray[np.floating],
    bearing_deg: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Rotate 2nd- and 3rd-order central (or normalized central) moments into the animal's heading frame.

    Transforms image-frame central moments ``mu_{p,q}`` (or normalized central moments ``nu_{p,q}``)
    into the egocentric body frame centered at the mask centroid with ``+x'`` along the animal's
    heading ``bearing_deg`` and ``+y'`` rotated +90 degrees in image coordinates:

    .. code-block:: text

        x' =  cos(theta) * dx + sin(theta) * dy
        y' = -sin(theta) * dx + cos(theta) * dy

    Because ``nu_{p,q} = mu_{p,q} / m00**(1 + (p + q) / 2)`` and ``m00`` is rotation invariant,
    the exact same linear transformation applies identically to both ``mu_*`` and ``nu_*`` arrays.
    Computation is performed in float64 to avoid precision loss before casting back to float32.
    Frames where ``bearing_deg`` or any input moment is ``NaN`` propagate ``NaN`` cleanly.

    Args:
        mu20: Image-frame 2nd-order moment ``mu20`` (or ``nu20``) of shape ``(F,)``.
        mu11: Image-frame 2nd-order moment ``mu11`` (or ``nu11``) of shape ``(F,)``.
        mu02: Image-frame 2nd-order moment ``mu02`` (or ``nu02``) of shape ``(F,)``.
        mu30: Image-frame 3rd-order moment ``mu30`` (or ``nu30``) of shape ``(F,)``.
        mu21: Image-frame 3rd-order moment ``mu21`` (or ``nu21``) of shape ``(F,)``.
        mu12: Image-frame 3rd-order moment ``mu12`` (or ``nu12``) of shape ``(F,)``.
        mu03: Image-frame 3rd-order moment ``mu03`` (or ``nu03``) of shape ``(F,)``.
        bearing_deg: Per-frame animal heading angle in degrees (``atan2(dy, dx)``), shape ``(F,)``.

    Returns:
        Tuple ``(r20, r11, r02, r30, r21, r12, r03)`` of float32 arrays in the animal's
        egocentric heading frame.
    """
    m20 = np.asarray(mu20, dtype=np.float64)
    m11 = np.asarray(mu11, dtype=np.float64)
    m02 = np.asarray(mu02, dtype=np.float64)
    m30 = np.asarray(mu30, dtype=np.float64)
    m21 = np.asarray(mu21, dtype=np.float64)
    m12 = np.asarray(mu12, dtype=np.float64)
    m03 = np.asarray(mu03, dtype=np.float64)

    rad = np.deg2rad(np.asarray(bearing_deg, dtype=np.float64))
    c = np.cos(rad)
    s = np.sin(rad)

    c2 = c * c
    s2 = s * s
    cs = c * s

    r20 = c2 * m20 + 2.0 * cs * m11 + s2 * m02
    r11 = -cs * m20 + (c2 - s2) * m11 + cs * m02
    r02 = s2 * m20 - 2.0 * cs * m11 + c2 * m02

    c3 = c2 * c
    s3 = s2 * s
    c2s = c2 * s
    cs2 = c * s2

    r30 = c3 * m30 + 3.0 * c2s * m21 + 3.0 * cs2 * m12 + s3 * m03
    r21 = -c2s * m30 + (c3 - 2.0 * cs2) * m21 + (2.0 * c2s - s3) * m12 + cs2 * m03
    r12 = cs2 * m30 + (s3 - 2.0 * c2s) * m21 + (c3 - 2.0 * cs2) * m12 + c2s * m03
    r03 = -s3 * m30 + 3.0 * cs2 * m21 - 3.0 * c2s * m12 + c3 * m03

    return (
        r20.astype(np.float32),
        r11.astype(np.float32),
        r02.astype(np.float32),
        r30.astype(np.float32),
        r21.astype(np.float32),
        r12.astype(np.float32),
        r03.astype(np.float32),
    )


class Moments(Feature):
    """Feature for heading-aligned (egocentric) image moments of segmentation contours."""

    _name = "moments"
    _requires = frozenset({PoseCapability.SEGMENTATION})
    # OpenCV exposes m*, mu*, and nu* moments. We keep m00 (mask area) plus the 2nd- and
    # 3rd-order central (mu) and normalized central (nu) moments rotated into the animal's
    # bearing frame so all 15 features are translation- and rotation-invariant across rigs
    # and arena orientations (e.g. DAX2 bottom-hopper vs. DAX3 right-hopper layouts).
    _moments_to_use: typing.ClassVar[list[str]] = [
        "m00",
        *(f"{prefix}{pq}" for prefix in _ROTATED_PREFIXES for pq in _ROTATED_ORDERS),
    ]

    def __init__(self, poses: PoseLike, pixel_scale: float, moment_cache: "MomentInfo"):
        super().__init__(poses, pixel_scale)
        self._moment_cache = moment_cache

    def per_frame(self, identity: int) -> dict[str, npt.NDArray[np.float32]]:
        """Computes per-frame egocentric image moment features for a specific identity.

        For each frame, extracts mask area (``m00``) and rotates the cached central (``mu*``)
        and normalized central (``nu*``) moments into the animal's bearing frame so the
        resulting features are invariant to both arena position and camera/arena rotation.
        On frames where segmentation exists but the animal's bearing is ``NaN``, ``m00``
        remains finite while the bearing-aligned ``mu*`` and ``nu*`` moments are ``NaN``.

        Args:
            identity: The identity index for which to compute moment features.

        Returns:
            Dictionary mapping moment names to float32 per-frame arrays of shape ``(F,)``.
        """
        values = {"m00": self._moment_cache.get_moment_array("m00").astype(np.float32, copy=True)}
        bearings = self._poses.compute_all_bearings(identity)
        for prefix in _ROTATED_PREFIXES:
            keys = [f"{prefix}{pq}" for pq in _ROTATED_ORDERS]
            rotated = rotate_central_moments(
                *(self._moment_cache.get_moment_array(key) for key in keys), bearings
            )
            values.update(zip(keys, rotated, strict=True))
        return values
