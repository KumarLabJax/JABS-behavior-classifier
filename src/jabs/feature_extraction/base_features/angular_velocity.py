import numpy as np

from jabs.feature_extraction.feature_base_class import Feature


class AngularVelocity(Feature):
    """Signed rate of change of the animal's bearing, in degrees per second.

    For each frame, the shortest turn from that frame's bearing to the next
    frame's is scaled by the frame rate. A positive value is a turn in the
    direction of increasing bearing. The final frame has no following frame to
    compare against, so it is nan, as is any frame where either bearing is
    missing.
    """

    _name = "angular_velocity"

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity

        Args:
            identity: identity to compute features for

        Returns:
            dict with feature values
        """
        fps = self._poses.fps

        bearings = self._poses.compute_all_bearings(identity)
        velocities = np.full(bearings.shape, np.nan, bearings.dtype)

        for i in range(len(bearings) - 1):
            angle1 = bearings[i]
            angle2 = bearings[i + 1]

            if np.isnan(angle1) or np.isnan(angle2):
                continue

            # `% 360` puts both bearings in [0, 360), so these are the only ways
            # to get from angle1 to angle2: directly, or by wrapping once around
            # the circle in either direction. Exactly one of the three is the
            # shortest turn, so taking the smallest magnitude picks it.
            angle1 = angle1 % 360
            angle2 = angle2 % 360

            turns = (
                angle2 - angle1,
                (360 + angle2) - angle1,
                angle2 - (360 + angle1),
            )

            # min() keeps the first of equal keys, so when two turns tie (an
            # exact half turn) the direct difference wins and the sign follows
            # the direction of travel
            velocities[i] = min(turns, key=abs)

        velocities = velocities * fps

        return {"angular_velocity": velocities}
