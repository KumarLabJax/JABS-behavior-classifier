"""Geometry helpers shared by feature extraction and the GUI.

Note: ``Angles._compute_angles`` in the ``base_features`` angle feature deliberately
does not use :func:`signed_angle_degrees`. It wraps to an unsigned ``[0, 360)`` range
instead, and its output is stored in the feature cache, so switching conventions there
would change computed feature values.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def signed_angle_degrees(
    a: npt.ArrayLike, vertex: npt.ArrayLike, c: npt.ArrayLike
) -> np.floating | npt.NDArray[np.floating]:
    """Compute the signed angle created by three connected points.

    Measures the direction of the ray ``vertex -> c`` relative to the ray
    ``vertex -> a``, so the result is the interior angle at ``vertex`` carrying the
    sign of the turn from ``a`` to ``c``.

    Args:
        a: Point, as ``(x, y)`` or an array of such points.
        vertex: Vertex point the angle is measured at, same shape as ``a``.
        c: Point, same shape as ``a``.

    Returns:
        Angle in degrees wrapped to the range ``[-180, 180)``. Scalar for single
        points, an array of one angle per row for arrays of points. The input
        dtype is preserved: float32 coordinates yield float32 angles.
    """
    a_xy = np.asarray(a)
    vertex_xy = np.asarray(vertex)
    c_xy = np.asarray(c)

    # trailing-axis indexing so a single (x, y) point and an (n, 2) array of points
    # both work
    angle = np.degrees(
        np.arctan2(c_xy[..., 1] - vertex_xy[..., 1], c_xy[..., 0] - vertex_xy[..., 0])
        - np.arctan2(a_xy[..., 1] - vertex_xy[..., 1], a_xy[..., 0] - vertex_xy[..., 0])
    )
    return ((angle + 180) % 360) - 180
