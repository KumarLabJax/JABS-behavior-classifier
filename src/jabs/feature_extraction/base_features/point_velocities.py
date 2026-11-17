import numpy as np
import numpy.typing as npt

from jabs.core.utils import wrap_degrees
from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

# TODO: merge this with point_speeds to reduce compute
#  since they both use keypoint gradients


class PointVelocityDirs(Feature):
    """Direction each keypoint is moving, measured relative to the animal's bearing.

    One angle per keypoint per frame, plus its sine and cosine. The angle is the
    direction of the keypoint's velocity minus the animal's bearing, so it
    answers "is this keypoint moving forward, backward or sideways relative to the
    way the animal is facing" independently of the animal's orientation in the arena:
    a keypoint moving along the animal's bearing reads ``0``, one moving 90 degrees
    counterclockwise of it reads ``90``, and one moving directly against it reads
    ``-180``. This is the same convention as ``CentroidVelocityDir``.

    Note:
        Before ``FEATURE_VERSION`` 19 these angles were offset by 180 degrees (motion
        along the bearing read ``-180``), because the wrap added 360 instead of 180.
    """

    # subclass must override this
    _name = "point_velocity_dirs"
    _point_index = None
    _use_circular = True

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> dict[str, npt.NDArray[np.floating]]:
        """Compute per-frame feature values.

        Args:
            identity: Subject identity.

        Returns:
            Dictionary of per frame values for this identity, keyed by ``"<KEYPOINT> velocity direction"`` and its ``" sine"`` and
            ``" cosine"`` variants. The direction angles are in degrees in the range
            ``[-180, 180)``, relative to the animal's bearing.
        """
        poses, point_masks = self._poses.get_identity_poses(identity, self._pixel_scale)

        bearings = self._poses.compute_all_bearings(identity)

        features = {}
        xy_deltas = np.gradient(poses, axis=0)
        angles = np.degrees(np.arctan2(xy_deltas[:, :, 1], xy_deltas[:, :, 0]))

        for keypoint in PoseEstimation.KeypointIndex:
            # subtract the animal's bearing and wrap into [-180, 180), the same way
            # CentroidVelocityDir.per_frame does
            features[f"{keypoint.name} velocity direction"] = wrap_degrees(
                angles[:, keypoint.value] - bearings
            )

            features[f"{keypoint.name} velocity direction sine"] = np.sin(
                np.deg2rad(features[f"{keypoint.name} velocity direction"])
            )
            features[f"{keypoint.name} velocity direction cosine"] = np.cos(
                np.deg2rad(features[f"{keypoint.name} velocity direction"])
            )

        return features
