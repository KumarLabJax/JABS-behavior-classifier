import typing

import numpy as np

from jabs.feature_extraction.angle_index import AngleIndex
from jabs.feature_extraction.feature_base_class import Feature


class Angles(Feature):
    """this module computes joint angles the result is a dict of features of length #frames rows"""

    _name = "angles"
    _use_circular = True

    # need to override to set the correct range for circular operations
    _circular_range: typing.ClassVar[tuple[float, float]] = (0.0, 360.0)

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """compute the value of the per frame features for a specific identity"""
        values = {}

        poses, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        for named_angle in AngleIndex:
            point, vertex, other_point = AngleIndex.get_angle_indices(named_angle)
            name = f"angle {AngleIndex.get_angle_name(named_angle)}"

            angles = self._compute_angles(poses[:, point], poses[:, vertex], poses[:, other_point])
            radians = np.deg2rad(angles)

            values[name] = angles
            values[f"{name} sine"] = np.sin(radians)
            values[f"{name} cosine"] = np.cos(radians)

        return values

    @staticmethod
    def _compute_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """compute angles for a set of points

        The angle is measured at vertex ``b``: it is the direction of the ray
        ``b -> c`` relative to the ray ``b -> a``, wrapped to ``[0, 360)``.

        Args:
            a: array of point coordinates
            b: array of vertex point coordinates
            c: array of point coordinates

        Returns:
            array containing one angle, in degrees, per row of a, b, and c, with range [0, 360)
        """
        angles = np.degrees(
            np.arctan2(c[:, 1] - b[:, 1], c[:, 0] - b[:, 0])
            - np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0])
        )
        return angles % 360
