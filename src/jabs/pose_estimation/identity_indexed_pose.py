"""Shared pose accessors for pose files whose arrays are indexed by identity.

Pose format v3 introduced the identity-major array layout that every later
format version still uses: ``_points`` has shape
``(#identities, #frames, #keypoints, 2)``, ``_point_mask`` has shape
``(#identities, #frames, #keypoints)``, and ``_identity_mask`` has shape
``(#identities, #frames)``. Reading a value out of those arrays is the same
work no matter which format version filled them in, so
:class:`IdentityIndexedPoseMixin` holds that shared implementation and the
version-specific readers supply only the parsing that differs.

Pose format v2 predates long term identity and stores frame-major arrays for a
single animal, so ``PoseEstimationV2`` implements these accessors itself rather
than using this mixin.
"""

import numpy as np
import numpy.typing as npt


class IdentityIndexedPoseMixin:
    """Read accessors for pose arrays indexed by identity first.

    Mix this in ahead of :class:`~jabs.core.abstract.pose_est.PoseEstimation` to
    satisfy its pose accessor abstract methods. The mixin stores nothing of its
    own; the including class is responsible for populating these attributes
    during initialization:

    Attributes:
        _points: Keypoint coordinates, shape ``(#identities, #frames, #keypoints, 2)``.
        _point_mask: Per-keypoint validity, shape ``(#identities, #frames, #keypoints)``.
        _identity_mask: Per-frame identity presence, shape ``(#identities, #frames)``.
    """

    _points: npt.NDArray
    _point_mask: npt.NDArray
    _identity_mask: npt.NDArray

    def get_points(
        self, frame_index: int, identity: int, scale: float | None = None
    ) -> tuple[npt.NDArray, npt.NDArray] | tuple[None, None]:
        """get points and mask for an identity for a given frame

        Args:
            frame_index: index of frame
            identity: identity that we want the points for
            scale: optional scale factor, set to cm_per_pixel to convert
                poses from pixel coordinates to cm coordinates

        Returns:
            points, mask if identity has data for this frame, otherwise
            ``None, None``
        """
        if not self._identity_mask[identity, frame_index]:
            return None, None

        if scale is not None:
            return (
                self._points[identity, frame_index, ...] * scale,
                self._point_mask[identity, frame_index, :],
            )
        else:
            return (
                self._points[identity, frame_index, ...],
                self._point_mask[identity, frame_index, :],
            )

    def get_identity_poses(
        self, identity: int, scale: float | None = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """return all points and point masks

        Args:
            identity: identity that we want the points for
            scale: optional scale factor, set to cm_per_pixel to convert
                poses from pixel coordinates to cm coordinates

        Returns:
            numpy array of points (#frames, 12, 2), numpy array of point masks
            (#frames, 12)
        """
        if scale is not None:
            return (
                self._points[identity, ...] * scale,
                self._point_mask[identity, ...],
            )
        else:
            return self._points[identity, ...], self._point_mask[identity, ...]

    def identity_mask(self, identity: int) -> npt.NDArray:
        """get the identity mask for a given identity

        Args:
            identity: identity to get the mask for

        Returns:
            array of length #frames, nonzero in each frame where the identity
            is present
        """
        return self._identity_mask[identity, :]

    def get_identity_point_mask(self, identity: int) -> npt.NDArray:
        """get the point mask array for a given identity

        Args:
            identity: identity to return point mask for

        Returns:
            array of point masks (#frames, 12)
        """
        return self._point_mask[identity, :]

    def get_reduced_point_mask(self) -> npt.NDArray:
        """Returns a boolean array of length 12 indicating which keypoints are valid.

        Determines which keypoints are valid for any identity across all frames.

        Returns:
            numpy array of shape (12,) with boolean values indicating validity
            of each keypoint.
        """
        return np.any(self._point_mask, axis=(0, 1))
