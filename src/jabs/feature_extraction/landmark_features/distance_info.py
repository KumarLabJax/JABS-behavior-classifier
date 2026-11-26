"""Shared lazily-populated cache of distances and bearings to a static object.

A landmark feature group needs the same per-identity distances and bearings in two
places: one feature reports the distances and another the bearings, and they cannot be
merged into a single ``Feature`` because the bearing features use circular window
operations and the distance features do not. The computation is therefore done once in
a helper object that both features hold, and this base class is the half of that helper
which is the same whichever static object is involved - the lazy fill on first access
and the handing out of the results.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from jabs.core.pose_protocol import PoseLike

# Feature column name to its per-frame values, one value per frame.
LandmarkFeatureValues: TypeAlias = dict[str, npt.NDArray[np.floating]]


class LandmarkDistanceInfo(abc.ABC):
    """Caches one video's distances and bearings to a static object, per identity.

    Subclasses implement :meth:`cache_features`, which fills ``_cached_distances`` and
    ``_cached_bearings`` for one identity. Everything else - computing on first access,
    and not recomputing afterwards - lives here so the two subclasses cannot drift
    apart on it.

    Attributes:
        _poses: Pose estimation data the features are derived from.
        _pixel_scale: Scale factor converting pixel distances to cm.
        _cached_distances: Per-identity distance columns, filled by ``cache_features``.
        _cached_bearings: Per-identity bearing columns, filled by ``cache_features``.
    """

    def __init__(self, poses: PoseLike, pixel_scale: float) -> None:
        self._poses = poses
        self._pixel_scale = pixel_scale
        self._cached_distances: dict[int, LandmarkFeatureValues] = {}
        self._cached_bearings: dict[int, LandmarkFeatureValues] = {}

    @abc.abstractmethod
    def cache_features(self, identity: int) -> None:
        """Compute and cache one identity's distances and bearings.

        Implementations return immediately when the identity is already cached, so
        calling this more than once costs nothing.

        Args:
            identity: Identity to compute and cache values for.
        """

    def get_distances(self, identity: int) -> LandmarkFeatureValues:
        """Return the distance columns for an identity, computing them if needed.

        Args:
            identity: Identity to get distances for.

        Returns:
            A new dict mapping each distance column name to its per-frame values. The
            dict is a copy, so a caller adding derived columns to it (as the bearing
            features do) cannot change what the next caller is given; the value arrays
            themselves are shared and must not be modified in place.
        """
        if identity not in self._cached_distances:
            self.cache_features(identity)
        return dict(self._cached_distances[identity])

    def get_bearings(self, identity: int) -> LandmarkFeatureValues:
        """Return the bearing columns for an identity, computing them if needed.

        Args:
            identity: Identity to get bearings for.

        Returns:
            A new dict mapping each bearing column name to its per-frame values, copied
            for the same reason as :meth:`get_distances`.
        """
        if identity not in self._cached_bearings:
            self.cache_features(identity)
        return dict(self._cached_bearings[identity])
