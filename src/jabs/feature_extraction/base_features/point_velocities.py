import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

# TODO: merge this with point_speeds to reduce compute
#  since they both use keypoint gradients


class PointVelocityDirs(Feature):
    """Direction each keypoint is moving, measured relative to the animal's bearing.

    One angle per keypoint per frame, plus its sine and cosine. The angle is the
    direction of the keypoint's velocity minus the animal's bearing, so it is
    intended to answer "is this keypoint moving forward, backward or sideways
    relative to the way the animal is facing" independently of the animal's
    orientation in the arena.

    Note:
        The angles this feature emits are offset by 180 degrees from that intended
        convention: a keypoint moving exactly along the animal's bearing reads
        ``-180`` rather than ``0``, and one moving 90 degrees off the bearing reads
        ``-90`` rather than ``90``. The sibling feature ``CentroidVelocityDir``
        computes the same bearing-relative quantity and does not carry the offset -
        see the wrapping expression in ``per_frame()`` below.

        The offset is a constant rotation, so it loses no information and the
        circular window statistics computed over these angles are unaffected in
        spread, but a reported value cannot be read as a bearing-relative heading.
        Correcting it changes cached feature values and so requires bumping
        ``FEATURE_VERSION``; it is left in place here so that change can be made
        deliberately rather than as a side effect.
        ``test_point_velocity_dirs_bearing_relative_convention`` pins the angles
        currently emitted, so correcting the wrap will show up there.
    """

    # subclass must override this
    _name = "point_velocity_dirs"
    _point_index = None
    _use_circular = True

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute per-frame feature values

        Args:
            identity (int): subject identity

        Returns:
            dict[str, np.ndarray]: dictionary of per frame values for this identity,
            keyed by ``"<KEYPOINT> velocity direction"`` and its ``" sine"`` and
            ``" cosine"`` variants. The direction angles are in degrees in the range
            ``[-180, 180)``, carrying the 180 degree offset described in the class
            docstring.
        """
        poses, point_masks = self._poses.get_identity_poses(identity, self._pixel_scale)

        bearings = self._poses.compute_all_bearings(identity)

        features = {}
        xy_deltas = np.gradient(poses, axis=0)
        angles = np.degrees(np.arctan2(xy_deltas[:, :, 1], xy_deltas[:, :, 0]))

        for keypoint in PoseEstimation.KeypointIndex:
            # Wraps into [-180, 180), but offset 180 degrees from a bearing-relative
            # heading: `+ 360` leaves the modulo a no-op where `+ 180` would have
            # centered the range (compare CentroidVelocityDir.per_frame, which uses
            # `+ 180`). Kept as-is because changing it changes cached feature values;
            # see the class docstring.
            features[f"{keypoint.name} velocity direction"] = (
                (angles[:, keypoint.value] - bearings + 360) % 360
            ) - 180

            features[f"{keypoint.name} velocity direction sine"] = np.sin(
                np.deg2rad(features[f"{keypoint.name} velocity direction"])
            )
            features[f"{keypoint.name} velocity direction cosine"] = np.cos(
                np.deg2rad(features[f"{keypoint.name} velocity direction"])
            )

        return features
