"""Helper computations for social distance, field-of-view, and huddle topology features."""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

from jabs.core.utils import signed_angle_degrees
from jabs.core.utils.pose_util import identity_centroids
from jabs.pose_estimation import PoseEstimation

logger = logging.getLogger(__name__)

# Fraction of an animal's characteristic body convex-hull size (sqrt(area))
# representing the soft flank/fur margin outside the internal skeletal keypoints.
_CONTACT_AREA_SCALE = 0.35
_CONTACT_PERIMETER_SCALE = 0.08

# For any elongated body shape (e.g. a 2:1 rectangle gives 1.5s vs sqrt(area) = 1.41s),
# 0.25 * perimeter is the larger term and serves as the usual body-length estimate;
# it also remains stable (~0.5 * spine_length) when only 3 nearly-collinear spine
# keypoints are visible and polygon area approaches 0. sqrt(area) bounds compact/curled
# postures, while 1.0 px guards against division by zero on degenerate geometries.
_BODY_LENGTH_PERIMETER_SCALE = 0.25
_MIN_BODY_CHAR_LEN_PX = 1.0

# Resolution-invariant segmentation contour contact tolerance: scales with the
# mean characteristic body size of the two animals (~4% of body width) while
# enforcing a 2.0 px floor because non-overlapping integer-pixel instance masks
# of two touching animals are separated by a 1.0 to sqrt(2) (~1.41 px) raster
# grid gap even when their coats are pressed directly together.
_SEG_CONTACT_BODY_SCALE = 0.04
_SEG_CONTACT_MIN_RASTER_PX = 2.0


def _characteristic_body_lengths(
    area: npt.NDArray[np.float64], length: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the characteristic pixel-space body length of each convex hull.

    Uses ``0.25 * perimeter`` as the usual body-length estimate for elongated animal
    hulls (which also remains stable when only 3 nearly-collinear keypoints are visible
    and polygon area collapses toward zero), ``sqrt(area)`` for compact/curled
    postures, and a ``1.0 px`` floor to prevent division by zero.

    Args:
        area: Hull areas (clamped to be non-negative), NaN where there is no hull.
        length: Hull perimeters (clamped to be non-negative), NaN where there is no hull.

    Returns:
        Characteristic body length per hull, NaN where there is no hull.
    """
    return np.maximum(
        np.maximum(_BODY_LENGTH_PERIMETER_SCALE * length, np.sqrt(area)), _MIN_BODY_CHAR_LEN_PX
    )


def _hull_contact_margins(
    area: npt.NDArray[np.float64],
    length: npt.NDArray[np.float64],
    is_segmentation_fallback: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Compute the pixel-space body contact margin of each convex hull.

    Because a skeletal convex hull spans internal keypoints rather than the outer
    fur coat, two huddling mice whose coats are in physical contact have a nonzero
    hull-to-hull gap proportional to their body width. Scaling this margin by the
    square root of hull area (and perimeter for elongated/collinear poses) makes the
    contact criterion resolution-invariant.

    Where ``is_segmentation_fallback`` is True, the hull was built from the outer
    segmentation contour and already wraps the coat, so the margin is ``0.0`` to
    avoid double-counting the fur allowance.

    Args:
        area: Hull areas (clamped to be non-negative), NaN where there is no hull.
        length: Hull perimeters (clamped to be non-negative), NaN where there is no hull.
        is_segmentation_fallback: Whether each hull was constructed from segmentation
            contour vertices rather than skeletal keypoints.

    Returns:
        Pixel-space contact margin per hull.
    """
    margin = np.maximum(_CONTACT_AREA_SCALE * np.sqrt(area), _CONTACT_PERIMETER_SCALE * length)
    return np.where(is_segmentation_fallback, 0.0, margin)


def _seg_contact_tolerances(
    body_len_a: npt.NDArray[np.float64], body_len_b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the resolution-invariant fur contour contact tolerance in pixels.

    Scales with the mean characteristic body size of the two animals so the fur
    contact criterion generalizes across camera resolutions and zoom levels, while
    enforcing a ``2.0 px`` floor for integer-pixel raster mask quantization.
    """
    mean_body_len = 0.5 * (body_len_a + body_len_b)
    return np.maximum(_SEG_CONTACT_MIN_RASTER_PX, _SEG_CONTACT_BODY_SCALE * mean_body_len)


def _extract_contour_geometry(
    frame_seg: npt.NDArray[np.generic],
) -> BaseGeometry | None:
    """Build a closed Shapely contour geometry from one frame's segmentation array.

    Constructs closed ``LineString`` rings (or a ``MultiLineString`` when an animal's
    mask has multiple contour components) over all valid vertices in ``frame_seg``
    without subsampling, so ``geom_a.distance(geom_b)`` measures exact Euclidean
    distance between contour line segments rather than strided vertices.

    Args:
        frame_seg: Array of shape ``(num_contours, num_vertices, 2)`` padded with ``-1``.

    Returns:
        Shapely ``LineString``, ``MultiLineString``, or ``Point`` representing the
        closed outer contour boundary, or ``None`` if no valid vertices exist.
    """
    rings: list[npt.NDArray[np.float32]] = []
    single_pt: npt.NDArray[np.float32] | None = None
    for contour in frame_seg:
        valid_mask = contour[:, 0] != -1
        if not np.any(valid_mask):
            continue
        pts = contour[valid_mask].astype(np.float32, copy=False)
        if len(pts) >= 2:
            closed = np.empty((len(pts) + 1, 2), dtype=np.float32)
            closed[:-1] = pts
            closed[-1] = pts[0]
            rings.append(closed)
        elif len(pts) == 1 and single_pt is None:
            single_pt = pts[0]
    if len(rings) == 1:
        return LineString(rings[0])
    if len(rings) > 1:
        return MultiLineString(rings)
    if single_pt is not None:
        return Point(float(single_pt[0]), float(single_pt[1]))
    return None


def _populate_segmentation_fallback_hulls(
    hulls: list[BaseGeometry | None],
    seg_arr: npt.NDArray[np.generic] | None,
) -> tuple[list[BaseGeometry | None], npt.NDArray[np.bool_]]:
    """Fill missing skeletal convex hulls from segmentation contours when ``< 3`` keypoints are valid.

    During deep multi-animal huddling, a buried animal may have fewer than 3 skeletal
    keypoints above the confidence threshold even though its segmentation mask remains
    visible. Constructing a fallback convex hull from the contour geometry preserves
    social distance and group topology features on those frames.

    Returns:
        Tuple of ``(updated_hulls, is_fallback_mask)`` where ``is_fallback_mask[frame]``
        is True on frames populated from ``seg_arr`` so hull-margin contact tests do not
        double-count the fur coat.
    """
    num_frames = len(hulls)
    is_fallback = np.zeros(num_frames, dtype=bool)
    if seg_arr is None:
        return list(hulls), is_fallback

    updated = list(hulls)
    for frame_idx, hull in enumerate(updated):
        if hull is None and frame_idx < len(seg_arr):
            contour_geom = _extract_contour_geometry(seg_arr[frame_idx])
            if contour_geom is not None:
                cand_hull = contour_geom.convex_hull
                if cand_hull.area > 0.0:
                    updated[frame_idx] = cand_hull
                    is_fallback[frame_idx] = True
    return updated, is_fallback


def _compute_identity_centroid_speeds(
    poses: PoseEstimation,
    identity: int,
) -> npt.NDArray[np.float32]:
    """Compute per-frame convex-hull centroid speed for one identity, in pixels per second.

    Matches ``CentroidVelocityMag`` in ``base_features`` by delegating centroid
    computation to ``identity_centroids(poses, identity)``, ensuring identical
    ``identity_mask`` handling and avoiding centroid jumps on frames that use a
    segmentation fallback hull. Speeds are left unscaled so the cache holding them
    does not depend on ``pixel_scale``.

    Args:
        poses: Pose estimation object for the video.
        identity: Identity index to compute speeds for.

    Returns:
        1-D ``float32`` array of centroid speeds of length ``poses.num_frames``.
    """
    num_frames = poses.num_frames
    if num_frames < 2:
        return np.full(num_frames, np.nan, dtype=np.float32)

    centroid_centers = identity_centroids(poses, identity)
    velocity = np.gradient(centroid_centers, axis=0)
    speed = np.linalg.norm(velocity, axis=-1) * float(poses.fps)
    return speed.astype(np.float32, copy=False)


# Contour geometries hold every vertex of every frame, so they are built and released
# in chunks of frames rather than all at once, bounding peak memory on long videos.
_CONTOUR_CHUNK_FRAMES = 4096


def _contour_geometries(
    seg_arr: npt.NDArray[np.generic] | None,
    present: npt.NDArray[np.bool_],
    start: int,
    stop: int,
) -> npt.NDArray[np.object_]:
    """Build closed contour geometries for one identity over a range of frames.

    Args:
        seg_arr: Segmentation contours for the identity, shape
            ``(num_frames, num_contours, num_vertices, 2)``, or None without segmentation.
        present: Whether the identity has a hull on each frame of the whole video.
        start: First frame of the range.
        stop: End of the range (exclusive).

    Returns:
        Object array of length ``stop - start`` holding a geometry, or None for frames
        where the identity has no hull or no valid contour vertices.
    """
    geoms = np.full(stop - start, None, dtype=object)
    if seg_arr is None:
        return geoms
    for frame in np.flatnonzero(present[start:stop]) + start:
        if frame < len(seg_arr):
            geoms[frame - start] = _extract_contour_geometry(seg_arr[frame])
    return geoms


def _connected_group_sizes(
    adjacency: npt.NDArray[np.bool_], present: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    """Return the size of the connected contact component containing each identity.

    Args:
        adjacency: Symmetric contact adjacency, shape ``(num_frames, k, k)``, only True
            between identities that are both present.
        present: Whether each identity is present, shape ``(num_frames, k)``.

    Returns:
        Component sizes, shape ``(num_frames, k)``; 1 for a present identity with no
        contacts, NaN for an absent identity.
    """
    k = adjacency.shape[-1]
    reach = adjacency.copy()
    diagonal = np.arange(k)
    reach[:, diagonal, diagonal] = present
    # transitive closure by repeated squaring; paths are at most k - 1 edges long
    steps = 1
    while steps < k:
        reach = np.matmul(reach.astype(np.uint8), reach.astype(np.uint8)) > 0
        steps *= 2
    return np.where(present, reach.sum(axis=2), np.nan).astype(np.float32)


@dataclass
class _VideoSocialCache:
    """Video-level pre-computed hulls, distances, and contact topology for all identities.

    ``IdentityFeatures`` builds a new ``SocialFeatureGroup`` (and ``ClosestIdentityInfo``)
    for each identity, but callers reuse the same ``PoseEstimation`` object across
    identities. ``for_pose()`` keys the cache on that object so the per-video work is
    done once rather than once per subject animal.

    Per-identity arrays have one element per frame. Pair arrays are keyed by the sorted
    identity pair and are NaN on frames where either identity has no hull.
    """

    has_seg: bool
    present_by_id: dict[int, npt.NDArray[np.bool_]]
    centroids_px_by_id: dict[int, npt.NDArray[np.float64]]
    body_lengths_px_by_id: dict[int, npt.NDArray[np.float64]]
    speeds_px_by_id: dict[int, npt.NDArray[np.float32]]
    contact_counts_by_id: dict[int, npt.NDArray[np.float32]]
    group_sizes_by_id: dict[int, npt.NDArray[np.float32]]
    hull_dists_px_by_pair: dict[tuple[int, int], npt.NDArray[np.float64]]
    seg_dists_px_by_pair: dict[tuple[int, int], npt.NDArray[np.float32]]

    def hull_distances_px(self, id_a: int, id_b: int) -> npt.NDArray[np.float64]:
        """Return the per-frame hull distance between two identities."""
        return self.hull_dists_px_by_pair[(id_a, id_b) if id_a < id_b else (id_b, id_a)]

    def seg_distances_px(
        self,
        id_a: npt.NDArray[np.integer],
        id_b: npt.NDArray[np.integer],
        frames: npt.NDArray[np.intp],
        hull_dists_px: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return contour distances for identity pairs, falling back to hull distances.

        Args:
            id_a: First identity for each element.
            id_b: Second identity for each element.
            frames: Frame for each element.
            hull_dists_px: Hull distance for each element, used when there is no
                contour distance.

        Returns:
            Contour distance (bounded by hull distance) for each element.
        """
        result = np.array(hull_dists_px, dtype=np.float64, copy=True)
        if not self.has_seg:
            return result
        for (pair_a, pair_b), pair_arr in self.seg_dists_px_by_pair.items():
            sel = ((id_a == pair_a) & (id_b == pair_b)) | ((id_a == pair_b) & (id_b == pair_a))
            vals = pair_arr[frames[sel]]
            result[sel] = np.where(np.isnan(vals), result[sel], vals)
        return result

    def pair_hull_distances_px(
        self,
        id_a: npt.NDArray[np.integer],
        id_b: npt.NDArray[np.integer],
        frames: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        """Return the hull distance for each (identity, identity, frame) element."""
        result = np.full(len(frames), np.nan, dtype=np.float64)
        for (pair_a, pair_b), pair_arr in self.hull_dists_px_by_pair.items():
            sel = ((id_a == pair_a) & (id_b == pair_b)) | ((id_a == pair_b) & (id_b == pair_a))
            result[sel] = pair_arr[frames[sel]]
        return result

    @classmethod
    def for_pose(cls, poses: PoseEstimation) -> _VideoSocialCache:
        """Return the shared cache for ``poses``, building it on first use."""
        cache = _VIDEO_SOCIAL_CACHES.get(poses)
        if cache is None:
            cache = cls.build(poses)
            _VIDEO_SOCIAL_CACHES[poses] = cache
        return cache

    @classmethod
    def build(cls, poses: PoseEstimation) -> _VideoSocialCache:
        """Build shared social geometry and contact caches for all identities in ``poses``."""
        num_frames = poses.num_frames
        identities = list(poses.identities)
        logger.debug(
            "Building social feature cache for %d identities, %d frames",
            len(identities),
            num_frames,
        )
        has_seg = bool(poses.format_major_version >= 6 and poses.has_segmentation)

        hulls_by_id: dict[int, npt.NDArray[np.object_]] = {}
        is_fallback_by_id: dict[int, npt.NDArray[np.bool_]] = {}
        present_by_id: dict[int, npt.NDArray[np.bool_]] = {}
        seg_by_id: dict[int, npt.NDArray[np.generic] | None] = {}
        centroids_px_by_id: dict[int, npt.NDArray[np.float64]] = {}
        body_lengths_px_by_id: dict[int, npt.NDArray[np.float64]] = {}
        margins_px_by_id: dict[int, npt.NDArray[np.float64]] = {}
        for curr_id in identities:
            seg_arr = poses.get_segmentation_data(curr_id) if has_seg else None
            seg_by_id[curr_id] = seg_arr
            raw_hulls: list[BaseGeometry | None] = list(poses.get_identity_convex_hulls(curr_id))
            hull_list, is_fallback_by_id[curr_id] = _populate_segmentation_fallback_hulls(
                raw_hulls, seg_arr
            )
            hulls = np.array(hull_list, dtype=object)
            present = ~shapely.is_missing(hulls)
            hulls_by_id[curr_id] = hulls
            present_by_id[curr_id] = present

            centroids = np.full((num_frames, 2), np.nan, dtype=np.float64)
            centroids[present] = shapely.get_coordinates(shapely.centroid(hulls[present]))
            centroids_px_by_id[curr_id] = centroids

            area = np.maximum(shapely.area(hulls), 0.0)
            length = np.maximum(shapely.length(hulls), 0.0)
            body_lengths_px_by_id[curr_id] = _characteristic_body_lengths(area, length)
            margins_px_by_id[curr_id] = _hull_contact_margins(
                area, length, is_fallback_by_id[curr_id]
            )

        speeds_px_by_id: dict[int, npt.NDArray[np.float32]] = {
            curr_id: _compute_identity_centroid_speeds(poses, curr_id) for curr_id in identities
        }

        k = len(identities)
        pairs = [(a_idx, b_idx) for a_idx in range(k) for b_idx in range(a_idx + 1, k)]
        hull_gaps: dict[tuple[int, int], npt.NDArray[np.float64]] = {}
        for a_idx, b_idx in pairs:
            id_a, id_b = identities[a_idx], identities[b_idx]
            both = np.flatnonzero(present_by_id[id_a] & present_by_id[id_b])
            hull_gap = np.full(num_frames, np.nan, dtype=np.float64)
            hull_gap[both] = shapely.distance(hulls_by_id[id_a][both], hulls_by_id[id_b][both])
            hull_gaps[(a_idx, b_idx)] = hull_gap

        # exact contour distance, bounded by the hull distance, where both animals have
        # contours and their hulls do not already touch
        seg_gaps = {pair: gap.copy() for pair, gap in hull_gaps.items()}
        if has_seg:
            for start in range(0, num_frames, _CONTOUR_CHUNK_FRAMES):
                stop = min(start + _CONTOUR_CHUNK_FRAMES, num_frames)
                chunk_contours = [
                    _contour_geometries(seg_by_id[curr_id], present_by_id[curr_id], start, stop)
                    for curr_id in identities
                ]
                for (a_idx, b_idx), seg_gap in seg_gaps.items():
                    contour_a, contour_b = chunk_contours[a_idx], chunk_contours[b_idx]
                    hull_gap = hull_gaps[(a_idx, b_idx)][start:stop]
                    with np.errstate(invalid="ignore"):
                        need = (
                            (hull_gap > 0.0)
                            & ~shapely.is_missing(contour_a)
                            & ~shapely.is_missing(contour_b)
                        )
                    seg_gap[start:stop][need] = np.minimum(
                        shapely.distance(contour_a[need], contour_b[need]), hull_gap[need]
                    )

        adjacency = np.zeros((num_frames, k, k), dtype=bool)
        hull_dists_px_by_pair: dict[tuple[int, int], npt.NDArray[np.float64]] = {}
        seg_dists_px_by_pair: dict[tuple[int, int], npt.NDArray[np.float32]] = {}
        for a_idx, b_idx in pairs:
            id_a, id_b = identities[a_idx], identities[b_idx]
            pair_key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
            both = np.flatnonzero(present_by_id[id_a] & present_by_id[id_b])
            hull_gap = hull_gaps[(a_idx, b_idx)]
            seg_gap = seg_gaps[(a_idx, b_idx)]

            margin_touch = ~(is_fallback_by_id[id_a][both] & is_fallback_by_id[id_b][both]) & (
                hull_gap[both] <= margins_px_by_id[id_a][both] + margins_px_by_id[id_b][both]
            )
            tolerance = _seg_contact_tolerances(
                body_lengths_px_by_id[id_a][both], body_lengths_px_by_id[id_b][both]
            )
            touching = margin_touch | (seg_gap[both] <= tolerance)
            adjacency[both, a_idx, b_idx] = touching
            adjacency[both, b_idx, a_idx] = touching

            hull_dists_px_by_pair[pair_key] = hull_gap
            # contact is decided on the float64 distance above; the stored copy is
            # float32, as features built from it always have been
            seg_dists_px_by_pair[pair_key] = seg_gap.astype(np.float32)

        present_matrix = np.stack([present_by_id[i] for i in identities], axis=1)
        counts = np.where(present_matrix, adjacency.sum(axis=2), np.nan).astype(np.float32)
        group_sizes = _connected_group_sizes(adjacency, present_matrix)

        return cls(
            has_seg=has_seg,
            present_by_id=present_by_id,
            centroids_px_by_id=centroids_px_by_id,
            body_lengths_px_by_id=body_lengths_px_by_id,
            speeds_px_by_id=speeds_px_by_id,
            contact_counts_by_id={i: counts[:, n] for n, i in enumerate(identities)},
            group_sizes_by_id={i: group_sizes[:, n] for n, i in enumerate(identities)},
            hull_dists_px_by_pair=hull_dists_px_by_pair,
            seg_dists_px_by_pair=seg_dists_px_by_pair,
        )


def _scale_like_python_float(
    values_px: npt.NDArray[np.float64], pixel_scale: float
) -> npt.NDArray[np.floating]:
    """Scale pixel distances that were previously Python floats, one frame at a time.

    ``python_float * pixel_scale`` takes the precision of ``pixel_scale``: float32
    arithmetic for ``np.float32`` (as read from pose files), float64 for a Python float
    or ``np.float64``. This reproduces that so vectorized values are unchanged.

    Args:
        values_px: Distances in pixels.
        pixel_scale: Scale factor to convert pixel distances to real-world units.

    Returns:
        Scaled distances.
    """
    # same type as the original ``python_float * pixel_scale``; it is a float even when
    # pixel_scale is an integer
    return values_px.astype(np.result_type(1.0 * pixel_scale)) * pixel_scale


def _gather_by_identity(
    values_by_id: dict[int, npt.NDArray[np.float32]],
    ids: npt.NDArray[np.integer],
    frames: npt.NDArray[np.intp],
) -> npt.NDArray[np.float32]:
    """Return ``values_by_id[ids[i]][frames[i]]`` for each element."""
    result = np.full(len(frames), np.nan, dtype=np.float32)
    for curr_id, values in values_by_id.items():
        sel = ids == curr_id
        result[sel] = values[frames[sel]]
    return result


# Shared caches keyed weakly on the PoseEstimation object, so a cache is released
# together with the pose data it was built from.
_VIDEO_SOCIAL_CACHES: weakref.WeakKeyDictionary[PoseEstimation, _VideoSocialCache] = (
    weakref.WeakKeyDictionary()
)


class ClosestIdentityInfo:
    """Pre-computed social proximity, field-of-view, and group topology data for one identity.

    This helper computes nearest-neighbor identities, hull and segmentation distances,
    field-of-view angles, multi-animal contact counts, transitive connected group sizes,
    and nearest-neighbor speeds once for a given subject identity, then shares the cached
    arrays across all social feature classes in ``SocialFeatureGroup``.

    Args:
        poses: Pose estimation data for the video.
        identity: Subject identity index.
        pixel_scale: Scale factor to convert pixel distances to real-world units (cm).
    """

    # Taken from the ICY paper where the full field of view is 240 degrees.
    HALF_FOV_DEGREE = 120

    def __init__(self, poses: PoseEstimation, identity: int, pixel_scale: float) -> None:
        idx = PoseEstimation.KeypointIndex

        self._poses = poses
        self._identity = identity
        self._pixel_scale = pixel_scale

        num_frames = poses.num_frames
        identities = list(poses.identities)
        other_identities = [curr_id for curr_id in identities if curr_id != identity]

        self._closest_identities = np.full(num_frames, -1, dtype=np.int16)
        self._second_closest_identities = np.full(num_frames, -1, dtype=np.int16)
        self._closest_fov_identities = np.full(num_frames, -1, dtype=np.int16)
        self._fov_angles = np.full(num_frames, np.nan, dtype=np.float32)

        self._closest_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_fov_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._second_closest_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_segmentation_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._second_closest_segmentation_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_segmentation_distances_body_units = np.full(
            num_frames, np.nan, dtype=np.float32
        )
        self._social_chain_bottleneck_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_neighbor_speeds = np.full(num_frames, np.nan, dtype=np.float32)
        self._second_closest_neighbor_speeds = np.full(num_frames, np.nan, dtype=np.float32)

        cache = _VideoSocialCache.for_pose(poses)

        # Contact count and connected group size come from the video's symmetric
        # per-frame contact graph, so all animals in a huddle cluster agree.
        self._social_contact_counts = cache.contact_counts_by_id[identity].copy()
        self._social_connected_group_sizes = cache.group_sizes_by_id[identity].copy()

        if not other_identities:
            return

        others = np.array(other_identities, dtype=np.int16)
        # hull distance to each other identity, shape (num_frames, num_others); NaN where
        # either animal has no hull (valid keypoints or a segmentation fallback)
        hull_dists = np.stack(
            [cache.hull_distances_px(identity, curr_id) for curr_id in other_identities], axis=1
        )
        has_dist = ~np.isnan(hull_dists)
        num_valid = has_dist.sum(axis=1)

        # rank the other identities by hull distance; a stable sort keeps the earliest
        # identity first on ties
        order = np.argsort(np.where(has_dist, hull_dists, np.inf), axis=1, kind="stable")
        frames_1 = np.flatnonzero(num_valid >= 1)
        frames_2 = np.flatnonzero(num_valid >= 2)
        col_1 = order[frames_1, 0]
        closest_ids = others[col_1]
        closest_dists = hull_dists[frames_1, col_1]
        self._closest_identities[frames_1] = closest_ids
        self._closest_distances[frames_1] = _scale_like_python_float(closest_dists, pixel_scale)
        self._closest_neighbor_speeds[frames_1] = (
            _gather_by_identity(cache.speeds_px_by_id, closest_ids, frames_1) * pixel_scale
        )

        identity_1 = np.full(len(frames_1), identity)
        seg_dist_px = cache.seg_distances_px(identity_1, closest_ids, frames_1, closest_dists)
        self._closest_segmentation_distances[frames_1] = _scale_like_python_float(
            seg_dist_px, pixel_scale
        )
        self._closest_segmentation_distances_body_units[frames_1] = (
            seg_dist_px / cache.body_lengths_px_by_id[identity][frames_1]
        )

        if len(other_identities) >= 2:
            col_2 = order[frames_2, 1]
            second_ids = others[col_2]
            second_dists = hull_dists[frames_2, col_2]
            first_ids = others[order[frames_2, 0]]
            self._second_closest_identities[frames_2] = second_ids
            self._second_closest_distances[frames_2] = _scale_like_python_float(
                second_dists, pixel_scale
            )
            self._second_closest_neighbor_speeds[frames_2] = (
                _gather_by_identity(cache.speeds_px_by_id, second_ids, frames_2) * pixel_scale
            )

            identity_2 = np.full(len(frames_2), identity)
            seg_dist_2_px = cache.seg_distances_px(identity_2, second_ids, frames_2, second_dists)
            self._second_closest_segmentation_distances[frames_2] = _scale_like_python_float(
                seg_dist_2_px, pixel_scale
            )

            # Continuous single-linkage bottleneck distance connecting the triad
            # (identity, closest_id, second_id): the second-smallest edge of the
            # 3-animal triangle is the maximum edge of its minimum spanning tree,
            # independent of whether hull ranking and contour ranking agree.
            hull_12 = cache.pair_hull_distances_px(first_ids, second_ids, frames_2)
            dist_12_px = cache.seg_distances_px(first_ids, second_ids, frames_2, hull_12)
            seg_dist_1_px = seg_dist_px[np.searchsorted(frames_1, frames_2)]
            bottleneck_px = np.sort(np.stack([seg_dist_1_px, seg_dist_2_px, dist_12_px]), axis=0)[
                1
            ]
            self._social_chain_bottleneck_distances[frames_2] = _scale_like_python_float(
                bottleneck_px, pixel_scale
            )

        # Nose and base-neck keypoints define the subject's facing direction for
        # field-of-view (FoV) features, whereas body-hull proximity depends only
        # on valid convex hulls so nose occlusion during huddling does not drop
        # closest-neighbor identity or distance measurements.
        points, mask = poses.get_identity_poses(identity, pixel_scale)
        has_fov_keypoints = (
            np.asarray(poses.identity_mask(identity), dtype=bool)
            & (mask[:, idx.NOSE] == 1)
            & (mask[:, idx.BASE_NECK] == 1)
        )
        view_angles = np.stack(
            [
                signed_angle_degrees(
                    points[:, idx.NOSE, :],
                    points[:, idx.BASE_NECK, :],
                    cache.centroids_px_by_id[curr_id] * self._pixel_scale,
                )
                for curr_id in other_identities
            ],
            axis=1,
        )
        with np.errstate(invalid="ignore"):
            in_fov = (
                has_fov_keypoints[:, np.newaxis]
                & has_dist
                & (np.abs(view_angles) <= self.HALF_FOV_DEGREE)
            )
        fov_frames = np.flatnonzero(in_fov.any(axis=1))
        # first identity on ties, matching the closest-identity ranking
        fov_col = np.argmin(np.where(in_fov, hull_dists, np.inf), axis=1)[fov_frames]
        self._closest_fov_identities[fov_frames] = others[fov_col]
        self._fov_angles[fov_frames] = view_angles[fov_frames, fov_col]
        self._closest_fov_distances[fov_frames] = _scale_like_python_float(
            hull_dists[fov_frames, fov_col], pixel_scale
        )

    @property
    def closest_identities(self) -> npt.NDArray[np.int16]:
        """Return the closest identities for each frame."""
        return self._closest_identities

    @property
    def closest_fov_identities(self) -> npt.NDArray[np.int16]:
        """Return the closest identities in the field of view for each frame."""
        return self._closest_fov_identities

    @property
    def closest_fov_angles(self) -> npt.NDArray[np.float32]:
        """Return the angles of the closest animals in the field of view."""
        return self._fov_angles

    @property
    def closest_distances(self) -> npt.NDArray[np.float32]:
        """Return the pre-computed hull distances to the closest identity per frame."""
        return self._closest_distances

    @property
    def closest_fov_distances(self) -> npt.NDArray[np.float32]:
        """Return the pre-computed hull distances to the closest FoV identity per frame."""
        return self._closest_fov_distances

    @property
    def second_closest_distances(self) -> npt.NDArray[np.float32]:
        """Return the hull distances to the second-closest identity per frame."""
        return self._second_closest_distances

    @property
    def closest_segmentation_distances(self) -> npt.NDArray[np.float32]:
        """Return the segmentation contour distances to the closest identity per frame."""
        return self._closest_segmentation_distances

    @property
    def second_closest_segmentation_distances(self) -> npt.NDArray[np.float32]:
        """Return the segmentation contour distances to the second-closest identity per frame."""
        return self._second_closest_segmentation_distances

    @property
    def closest_segmentation_distances_body_units(self) -> npt.NDArray[np.float32]:
        """Return the closest segmentation distance normalized by characteristic body size."""
        return self._closest_segmentation_distances_body_units

    @property
    def social_contact_counts(self) -> npt.NDArray[np.float32]:
        """Return the number of other identities directly in body contact per frame."""
        return self._social_contact_counts

    @property
    def social_connected_group_sizes(self) -> npt.NDArray[np.float32]:
        """Return the size of the subject's connected social contact cluster per frame."""
        return self._social_connected_group_sizes

    @property
    def social_chain_bottleneck_distances(self) -> npt.NDArray[np.float32]:
        """Return the single-linkage minimax distance connecting the 3 nearest animals per frame."""
        return self._social_chain_bottleneck_distances

    @property
    def closest_neighbor_speeds(self) -> npt.NDArray[np.float32]:
        """Return the centroid speed of the closest identity per frame."""
        return self._closest_neighbor_speeds

    @property
    def second_closest_neighbor_speeds(self) -> npt.NDArray[np.float32]:
        """Return the centroid speed of the second-closest identity per frame."""
        return self._second_closest_neighbor_speeds

    def compute_distances(
        self, closest_identities: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.float32]:
        """Compute frame-wise distances between the subject's hull and the specified identity's hull.

        Args:
            closest_identities: Array of closest identity indices for each frame.

        Returns:
            Array of distances for each frame, with NaN where distance cannot be computed.
        """
        values = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        self_convex_hulls = self._poses.get_identity_convex_hulls(self._identity)

        for frame in range(self._poses.num_frames):
            points, _ = self._poses.get_points(frame, self._identity, self._pixel_scale)
            if points is None:
                continue

            closest_id = int(closest_identities[frame])
            if closest_id != -1:
                self_shape = self_convex_hulls[frame]
                other_shape = self._poses.get_identity_convex_hulls(closest_id)[frame]
                if self_shape is not None and other_shape is not None:
                    values[frame] = float(self_shape.distance(other_shape)) * self._pixel_scale

        return values

    def compute_pairwise_social_distances(
        self,
        social_points: list[PoseEstimation.KeypointIndex],
        closest_identities: npt.NDArray[np.integer],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Compute pairwise distances between specified keypoints of the subject and closest identity.

        Args:
            social_points: List of keypoint indices to use for pairwise distance calculation.
            closest_identities: Array of closest identity indices for each frame.

        Returns:
            Dictionary mapping keypoint pair names to arrays of distances for each frame,
            NaN where there is no closest identity or either animal is absent.
        """
        num_frames = self._poses.num_frames
        closest_identities = np.asarray(closest_identities)
        self_points, _ = self._poses.get_identity_poses(self._identity, self._pixel_scale)

        # keypoints of each frame's closest identity, gathered from every identity
        closest_points = np.full_like(self_points, np.nan)
        closest_present = np.zeros(num_frames, dtype=bool)
        for curr_id in self._poses.identities:
            frames = closest_identities == curr_id
            if not frames.any():
                continue
            points, _ = self._poses.get_identity_poses(curr_id, self._pixel_scale)
            closest_points[frames] = points[frames]
            closest_present[frames] = np.asarray(self._poses.identity_mask(curr_id), dtype=bool)[
                frames
            ]

        valid = np.asarray(self._poses.identity_mask(self._identity), dtype=bool) & closest_present

        return_dict: dict[str, npt.NDArray[np.float32]] = {}
        for p1 in social_points:
            subject_xy = self_points[valid, p1.value].astype(np.float64)
            for p2 in social_points:
                other_xy = closest_points[valid, p2.value].astype(np.float64)
                values = np.full(num_frames, np.nan, dtype=np.float32)
                values[valid] = np.hypot(
                    subject_xy[:, 0] - other_xy[:, 0], subject_xy[:, 1] - other_xy[:, 1]
                )
                return_dict[f"social dist. {p1.name}-{p2.name}"] = values

        return return_dict
