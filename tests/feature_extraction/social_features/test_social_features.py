"""Tests for social distance, field-of-view gating, and multi-animal huddle features."""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from shapely.geometry import Polygon

import jabs.pose_estimation as pose_est
from jabs.feature_extraction.social_features import (
    ClosestDistances,
    ClosestFovAngles,
    ClosestFovDistances,
    HuddleFeatures,
    SocialFeatureGroup,
    social_distance,
)
from jabs.feature_extraction.social_features.social_distance import ClosestIdentityInfo
from jabs.pose_estimation import PoseEstimation


def _square_hull(cx: float, cy: float, size: float = 10.0) -> Polygon:
    """Build an axis-aligned square Polygon centered at ``(cx, cy)``."""
    half = size / 2.0
    return Polygon(
        [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
    )


class _MultiFrameStubPose:
    """Synthetic multi-frame pose stand-in for testing social and huddle features."""

    def __init__(
        self,
        centers_by_id: dict[int, list[tuple[float, float] | None]],
        *,
        hull_size: float = 10.0,
        nose_valid: bool = True,
        fps: int = 30,
        format_major_version: int = 5,
        seg_contours_by_id: dict[int, npt.NDArray[np.int16]] | None = None,
    ) -> None:
        self._centers_by_id = centers_by_id
        self._hull_size = hull_size
        self._nose_valid = nose_valid
        self.fps = fps
        self.format_major_version = format_major_version
        self._seg_contours_by_id = seg_contours_by_id
        self.has_segmentation = seg_contours_by_id is not None
        self.identities = list(centers_by_id.keys())
        self.num_frames = len(next(iter(centers_by_id.values())))

        self._hulls: dict[int, list[Polygon | None]] = {
            ident: [
                None if center is None else _square_hull(center[0], center[1], size=hull_size)
                for center in frame_centers
            ]
            for ident, frame_centers in centers_by_id.items()
        }

    def get_identity_convex_hulls(self, identity: int) -> list[Polygon | None]:
        """Return pre-built square hulls for ``identity``."""
        return self._hulls[identity]

    def get_points(
        self, frame_index: int, identity: int, scale: float | None = None
    ) -> tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.uint8] | None]:
        """Return synthetic keypoints with nose ahead of base neck along +x."""
        center = self._centers_by_id[identity][frame_index]
        if center is None:
            return None, None

        factor = 1.0 if scale is None else scale
        n_kp = len(PoseEstimation.KeypointIndex)
        points = np.zeros((n_kp, 2), dtype=np.float32)
        mask = np.ones(n_kp, dtype=np.uint8)

        cx, cy = center
        points[:, 0] = cx * factor
        points[:, 1] = cy * factor
        points[PoseEstimation.KeypointIndex.NOSE] = ((cx + 5.0) * factor, cy * factor)
        points[PoseEstimation.KeypointIndex.BASE_NECK] = (cx * factor, cy * factor)
        if not self._nose_valid and identity == 0:
            mask[PoseEstimation.KeypointIndex.NOSE] = 0
        return points, mask

    def get_identity_poses(
        self, identity: int, scale: float | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Return synthetic keypoints for every frame; zeros where ``identity`` is absent."""
        n_kp = len(PoseEstimation.KeypointIndex)
        points = np.zeros((self.num_frames, n_kp, 2), dtype=np.float32)
        mask = np.zeros((self.num_frames, n_kp), dtype=np.uint8)
        for frame in range(self.num_frames):
            frame_points, frame_mask = self.get_points(frame, identity, scale)
            if frame_points is not None:
                points[frame], mask[frame] = frame_points, frame_mask
        return points, mask

    def identity_mask(self, identity: int) -> npt.NDArray[np.uint8]:
        """Return 1 on frames where ``identity`` has a center, 0 otherwise."""
        return np.array(
            [0 if c is None else 1 for c in self._centers_by_id[identity]],
            dtype=np.uint8,
        )

    def get_segmentation_data(self, identity: int) -> npt.NDArray[np.int16] | None:
        """Return synthetic segmentation contours when configured."""
        if self._seg_contours_by_id is None:
            return None
        return self._seg_contours_by_id.get(identity)


def test_nose_occlusion_preserves_closest_social_distance_while_masking_fov() -> None:
    """Occluding the nose during huddling keeps hull distance valid while FoV becomes NaN.

    Matches the behavior of ``jabs.ui.player_widget.closest_identity``: body-hull
    proximity depends only on valid convex hulls, whereas field-of-view features
    additionally require valid nose and base-neck keypoints.
    """
    pose = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0)],
            1: [(20.0, 0.0), (20.0, 0.0)],
            2: [(50.0, 0.0), (50.0, 0.0)],
        },
        hull_size=10.0,
        nose_valid=False,
    )

    info = ClosestIdentityInfo(pose, identity=0, pixel_scale=1.0)
    np.testing.assert_array_equal(info.closest_identities, np.array([1, 1], dtype=np.int16))
    np.testing.assert_array_equal(info.closest_fov_identities, np.array([-1, -1], dtype=np.int16))

    closest_feat = ClosestDistances(pose, 1.0, info).per_frame(0)
    fov_dist_feat = ClosestFovDistances(pose, 1.0, info).per_frame(0)
    fov_angle_feat = ClosestFovAngles(pose, 1.0, info).per_frame(0)

    # Hulls of size 10 centered at x=0 and x=20 have gap 20 - (5 + 5) = 10.
    np.testing.assert_allclose(closest_feat["closest social distance"], [10.0, 10.0])
    # Second closest is identity 2 at x=50, gap 50 - (5 + 5) = 40.
    np.testing.assert_allclose(closest_feat["second closest social distance"], [40.0, 40.0])
    assert np.all(np.isnan(fov_dist_feat["closest social distance in FoV"]))
    assert np.all(np.isnan(fov_angle_feat["angle of closest social distance in FoV"]))


def test_compute_distances_and_cached_properties() -> None:
    """closest_distances/closest_fov_distances properties and compute_distances agree."""
    pose = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0)],
            1: [(-20.0, 0.0), (-20.0, 0.0)],  # Behind subject 0 (which faces +x)
            2: [(40.0, 0.0), (40.0, 0.0)],  # In front of subject 0 (in FoV)
        },
        hull_size=10.0,
        nose_valid=True,
    )
    info = ClosestIdentityInfo(pose, identity=0, pixel_scale=2.0)

    # Nearest overall is identity 1 (gap 10 px * scale 2.0 = 20.0).
    np.testing.assert_array_equal(info.closest_identities, [1, 1])
    np.testing.assert_allclose(info.closest_distances, [20.0, 20.0])
    np.testing.assert_allclose(info.compute_distances(info.closest_identities), [20.0, 20.0])

    # Nearest in FoV is identity 2 (gap 30 px * scale 2.0 = 60.0).
    np.testing.assert_array_equal(info.closest_fov_identities, [2, 2])
    np.testing.assert_allclose(info.closest_fov_distances, [60.0, 60.0])
    np.testing.assert_allclose(info.compute_distances(info.closest_fov_identities), [60.0, 60.0])

    # Custom identity array [2, -1] exercises general frame-wise distance calculation.
    custom = np.array([2, -1], dtype=np.int16)
    custom_dists = info.compute_distances(custom)
    assert custom_dists[0] == pytest.approx(60.0)
    assert np.isnan(custom_dists[1])


def test_huddle_features_distinguish_isolated_dyad_triangle_and_chain() -> None:
    """HuddleFeatures separates isolated (1), dyadic (2), triangle (3), and chain (3) groups.

    Frames:
        0 (isolated): 0 at (0,0), 1 at (100,0), 2 at (200,0) -> contact=0, group=1
        1 (dyad 0-1): 0 at (0,0), 1 at (12,0), 2 at (200,0) -> contact=1, group=2
        2 (triangle 0-1-2): 0 at (0,0), 1 at (10,0), 2 at (5,8) -> contact=2, group=3
        3 (chain 0-1-2): 0 at (0,0), 1 at (12,0), 2 at (24,0)
          For subject 0, hull gap to 1 is 2.0 (in contact, margin=3.5+3.5=7.0),
          while hull gap to 2 is 14.0 (> 7.0, not in direct contact). Transitive
          connectivity through 1 yields contact_count=1 and connected_group_size=3.
    """
    pose = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            1: [(100.0, 0.0), (12.0, 0.0), (10.0, 0.0), (12.0, 0.0)],
            2: [(200.0, 0.0), (200.0, 0.0), (5.0, 8.0), (24.0, 0.0)],
        },
        hull_size=10.0,
    )

    info_0 = ClosestIdentityInfo(pose, identity=0, pixel_scale=1.0)
    huddle_0 = HuddleFeatures(pose, 1.0, info_0).per_frame(0)

    np.testing.assert_allclose(huddle_0["social contact count"], [0.0, 1.0, 2.0, 1.0])
    np.testing.assert_allclose(huddle_0["social connected group size"], [1.0, 2.0, 3.0, 3.0])
    # Threshold-free single-linkage bottleneck distance connecting the 3 animals:
    # Frame 0 (0@0, 1@100, 2@200): d01=90, d02=190, d12=90 -> sorted(90, 190, 90)[1] = 90.0
    # Frame 1 (0@0, 1@12, 2@200): d01=2, d02=190, d12=178 -> sorted(2, 190, 178)[1] = 178.0
    # Frame 3 (chain 0@0, 1@12, 2@24): d01=2, d02=14, d12=2 -> sorted(2, 14, 2)[1] = 2.0!
    assert huddle_0["social chain bottleneck distance"][0] == pytest.approx(90.0)
    assert huddle_0["social chain bottleneck distance"][1] == pytest.approx(178.0)
    assert huddle_0["social chain bottleneck distance"][3] == pytest.approx(2.0)

    # Middle mouse (identity 1) on frame 3 directly touches both 0 and 2.
    info_1 = ClosestIdentityInfo(pose, identity=1, pixel_scale=1.0)
    huddle_1 = HuddleFeatures(pose, 1.0, info_1).per_frame(1)
    assert huddle_1["social contact count"][3] == pytest.approx(2.0)
    assert huddle_1["social connected group size"][3] == pytest.approx(3.0)


def test_symmetric_contact_graph_and_order_independent_bottleneck_in_mixed_chain() -> None:
    """All animals in a mixed hull+fur chain agree on group size, and bottleneck is order-independent.

    Tests two edge cases:
    1. Mixed contact chain (Frame 0):
       - 0 at (0, 0), 1 at (12, 0), 2 at (42, 0) with 10x10 skeletal hulls.
       - 0 and 1 touch by skeletal hull margin (hull gap = 2.0 <= 7.0).
       - 1 and 2 have skeletal hull gap = 20.0 (> 7.0), but their outer fur contours
         touch (contour gap = 1.0 px <= 2.0 px).
       - Every subject (0, 1, and 2) must report connected_group_size = 3.0 and
         symmetric contact counts (1.0, 2.0, 1.0).
    2. Order-independent bottleneck when contour order reverses hull order (Frame 1):
       - Subject 0 at (0, 0), identity 1 at (20, 0) (hull gap = 10.0),
         identity 2 at (0, 22) (hull gap = 12.0).
       - Identity 1 is closer by skeletal hull (10.0 < 12.0), but identity 2's outer
         fur contour touches both 0 and 1 at 1.0 px (seg_dist_px = 10.0, seg_dist_2_px = 1.0,
         dist_12_px = 1.0).
       - The true 3-animal bottleneck is sorted((10.0, 1.0, 1.0))[1] == 1.0.
    """
    # Frame 0: 1 and 2 have fur contours touching at x=26..27
    # Frame 1: 2's fur contour extends to touch both 0 (at y=5..6) and 1 (at x=14..15)
    seg_0 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_0[0, 0] = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=np.int16)
    seg_0[1, 0] = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=np.int16)

    seg_1 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_1[0, 0] = np.array([[7, -5], [26, -5], [26, 5], [7, 5]], dtype=np.int16)
    seg_1[1, 0] = np.array([[15, -5], [25, -5], [25, 5], [15, 5]], dtype=np.int16)

    seg_2 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_2[0, 0] = np.array([[27, -5], [47, -5], [47, 5], [27, 5]], dtype=np.int16)
    seg_2[1, 0] = np.array([[-5, 6], [14, 5], [14, 27], [-5, 27]], dtype=np.int16)

    pose = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0)],
            1: [(12.0, 0.0), (20.0, 0.0)],
            2: [(42.0, 0.0), (0.0, 22.0)],
        },
        hull_size=10.0,
        format_major_version=6,
        seg_contours_by_id={0: seg_0, 1: seg_1, 2: seg_2},
    )

    group = SocialFeatureGroup(pose, pixel_scale=1.0)
    h0 = group.per_frame(0)["huddle_features"]
    h1 = group.per_frame(1)["huddle_features"]
    h2 = group.per_frame(2)["huddle_features"]

    # Frame 0: symmetric contact counts (1, 2, 1) and identical group size (3, 3, 3)
    assert h0["social contact count"][0] == pytest.approx(1.0)
    assert h1["social contact count"][0] == pytest.approx(2.0)
    assert h2["social contact count"][0] == pytest.approx(1.0)
    assert h0["social connected group size"][0] == pytest.approx(3.0)
    assert h1["social connected group size"][0] == pytest.approx(3.0)
    assert h2["social connected group size"][0] == pytest.approx(3.0)

    # Frame 1: order-independent bottleneck is 1.0 even though seg_dist_px (10.0) > seg_dist_2_px (1.0)
    assert h0["social chain bottleneck distance"][1] == pytest.approx(1.0)


def test_closest_neighbor_speed_distinguishes_still_partner_from_moving_partner() -> None:
    """closest neighbor speed reflects the velocity of the nearest animal."""
    pose_still = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            1: [(15.0, 0.0), (15.0, 0.0), (15.0, 0.0)],
        },
        fps=30,
    )
    info_still = ClosestIdentityInfo(pose_still, identity=0, pixel_scale=1.0)
    feat_still = HuddleFeatures(pose_still, 1.0, info_still).per_frame(0)
    np.testing.assert_allclose(feat_still["closest neighbor speed"], [0.0, 0.0, 0.0])

    pose_moving = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            1: [(15.0, 0.0), (15.0, 4.0), (15.0, 8.0)],
        },
        fps=30,
    )
    info_moving = ClosestIdentityInfo(pose_moving, identity=0, pixel_scale=0.5)
    feat_moving = HuddleFeatures(pose_moving, 0.5, info_moving).per_frame(0)
    # 4 px/frame * 30 fps * 0.5 cm/px = 60.0 cm/s
    np.testing.assert_allclose(feat_moving["closest neighbor speed"], [60.0, 60.0, 60.0])


def test_closest_segmentation_distance_uses_contours_in_v6_and_falls_back_in_v5() -> None:
    """Segmentation contours extending outside skeletal hulls reduce the coat distance."""
    seg_0 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_0[:, 0, :, :] = np.array([[[-10, -8], [14, -8], [14, 8], [-10, 8]]], dtype=np.int16)

    seg_1 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_1[:, 0, :, :] = np.array([[[15, -8], [40, -8], [40, 8], [15, 8]]], dtype=np.int16)

    pose_v6 = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0)],
            1: [(30.0, 0.0), (30.0, 0.0)],
        },
        hull_size=10.0,
        format_major_version=6,
        seg_contours_by_id={0: seg_0, 1: seg_1},
    )
    info_v6 = ClosestIdentityInfo(pose_v6, identity=0, pixel_scale=1.0)
    dists_v6 = ClosestDistances(pose_v6, 1.0, info_v6).per_frame(0)
    huddle_v6 = HuddleFeatures(pose_v6, 1.0, info_v6).per_frame(0)

    np.testing.assert_allclose(dists_v6["closest social distance"], [20.0, 20.0])
    np.testing.assert_allclose(dists_v6["closest social segmentation distance"], [1.0, 1.0])
    np.testing.assert_allclose(huddle_v6["social contact count"], [1.0, 1.0])

    # Without segmentation (v5), segmentation distance falls back to hull distance (20.0).
    pose_v5 = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0)],
            1: [(30.0, 0.0), (30.0, 0.0)],
        },
        hull_size=10.0,
        format_major_version=5,
    )
    info_v5 = ClosestIdentityInfo(pose_v5, identity=0, pixel_scale=1.0)
    dists_v5 = ClosestDistances(pose_v5, 1.0, info_v5).per_frame(0)
    np.testing.assert_allclose(dists_v5["closest social segmentation distance"], [20.0, 20.0])


def test_social_feature_group_on_real_pose_v6_slice() -> None:
    """SocialFeatureGroup computes all distance and huddle features on real PoseEstimationV6 data."""
    sample_v6 = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"
    with tempfile.TemporaryDirectory() as tmpdir:
        short_v6 = Path(tmpdir) / sample_v6.name
        shutil.copy(sample_v6, short_v6)
        poses = pose_est.open_pose_file(short_v6)

        group = SocialFeatureGroup(poses, pixel_scale=1.0)
        per_frame = group.per_frame(0)

        assert "closest_distances" in per_frame
        assert "huddle_features" in per_frame
        assert {
            "closest social distance",
            "second closest social distance",
            "closest social segmentation distance",
            "second closest social segmentation distance",
            "closest social segmentation distance body units",
        } == set(per_frame["closest_distances"].keys())
        assert {
            "social contact count",
            "social connected group size",
            "social chain bottleneck distance",
            "closest neighbor speed",
            "second closest neighbor speed",
            "torso blob distance",
            "torso blob centroid distance",
            "torso connected group size",
            "torso group blob hull area",
            "torso group blob solidity",
        } == set(per_frame["huddle_features"].keys())

        hull_d = per_frame["closest_distances"]["closest social distance"]
        seg_d = per_frame["closest_distances"]["closest social segmentation distance"]
        both_valid = np.isfinite(hull_d) & np.isfinite(seg_d)
        assert np.any(both_valid)
        assert np.all(seg_d[both_valid] <= hull_d[both_valid] + 1e-5)


def test_segmentation_contour_fallback_hull_when_skeletal_keypoints_drop_out() -> None:
    """When < 3 skeletal keypoints are valid (hull is None), segmentation contours provide a fallback hull."""
    seg_0 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_0[:, 0, :, :] = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.int16)

    seg_1 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_1[:, 0, :, :] = np.array([[[12, 0], [22, 0], [22, 10], [12, 10]]], dtype=np.int16)

    pose_occluded = _MultiFrameStubPose(
        {
            0: [None, (5.0, 5.0)],
            1: [(17.0, 5.0), (17.0, 5.0)],
        },
        hull_size=10.0,
        format_major_version=6,
        seg_contours_by_id={0: seg_0, 1: seg_1},
    )
    info = ClosestIdentityInfo(pose_occluded, identity=0, pixel_scale=1.0)
    dists = ClosestDistances(pose_occluded, 1.0, info).per_frame(0)
    huddle = HuddleFeatures(pose_occluded, 1.0, info).per_frame(0)

    # Frame 0 recovers a valid 2.0 px distance and connected group size 2 from seg_0!
    np.testing.assert_allclose(dists["closest social segmentation distance"], [2.0, 2.0])
    np.testing.assert_allclose(
        dists["closest social segmentation distance body units"], [0.2, 0.2]
    )
    np.testing.assert_allclose(huddle["social connected group size"], [2.0, 2.0])


def test_video_social_cache_is_built_once_per_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every identity on one pose object shares a single video-level social cache.

    ``IdentityFeatures`` builds a new ``SocialFeatureGroup`` per identity, so the
    cache must be keyed on the pose object rather than held by the feature group.
    """
    build_spy = mock.Mock(wraps=social_distance._VideoSocialCache.build)
    monkeypatch.setattr(social_distance._VideoSocialCache, "build", build_spy)

    pose = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0)],
            1: [(12.0, 0.0), (12.0, 0.0)],
            2: [(24.0, 0.0), (24.0, 0.0)],
        }
    )
    for identity in pose.identities:
        SocialFeatureGroup(pose, 1.0)._init_feature_mods(identity)
    assert build_spy.call_count == 1

    other_pose = _MultiFrameStubPose({0: [(0.0, 0.0)], 1: [(12.0, 0.0)]})
    ClosestIdentityInfo(other_pose, identity=0, pixel_scale=1.0)
    assert build_spy.call_count == 2


def test_neighbor_speed_scales_with_pixel_scale_on_shared_cache() -> None:
    """The shared cache stores unscaled speeds, so each pixel_scale gets its own units."""
    pose = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            1: [(20.0, 0.0), (21.0, 0.0), (22.0, 0.0)],
        },
        fps=30,
    )
    speeds_px = ClosestIdentityInfo(pose, 0, 1.0).closest_neighbor_speeds
    speeds_cm = ClosestIdentityInfo(pose, 0, 0.5).closest_neighbor_speeds
    np.testing.assert_allclose(speeds_px, [30.0, 30.0, 30.0])
    np.testing.assert_allclose(speeds_cm, [15.0, 15.0, 15.0])


def test_contour_segment_distance_avoids_vertex_subsampling_overestimate() -> None:
    """Contour distance is measured to closed contour segments rather than strided vertices."""
    # Animal 0 has a 100-vertex contour whose right edge is a single long segment from
    # (10, -40) to (10, 40); any vertex-only or 64-vertex strided check against a neighbor
    # near (11, 0) would overstate the gap (> 40 px to the nearest corner vertex), whereas
    # closed LineString segment distance measures the exact 1.0 px perpendicular gap.
    left_pts = [[0, y] for y in range(-40, 41)]  # 81 vertices along x=0
    ring_0 = np.array([*left_pts, [10, 40], [10, -40]], dtype=np.int16)
    seg_0 = np.full((1, 1, len(ring_0), 2), -1, dtype=np.int16)
    seg_0[0, 0, :, :] = ring_0

    seg_1 = np.full((1, 1, 4, 2), -1, dtype=np.int16)
    seg_1[0, 0, :, :] = np.array([[11, -2], [25, -2], [25, 2], [11, 2]], dtype=np.int16)

    pose = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0)],
            1: [(30.0, 0.0)],
        },
        hull_size=10.0,
        format_major_version=6,
        seg_contours_by_id={0: seg_0, 1: seg_1},
    )
    info = ClosestIdentityInfo(pose, identity=0, pixel_scale=1.0)
    dists = ClosestDistances(pose, 1.0, info).per_frame(0)
    huddle = HuddleFeatures(pose, 1.0, info).per_frame(0)

    np.testing.assert_allclose(dists["closest social segmentation distance"], [1.0])
    np.testing.assert_allclose(huddle["social contact count"], [1.0])


def test_connected_group_sizes_follow_transitive_contact() -> None:
    """A contact chain forms one group; absent identities are NaN, isolated ones are 1."""
    # frame 0: 0-1 and 1-2 touch (chain of 3), 3 alone; frame 1: identity 2 absent
    adjacency = np.zeros((2, 4, 4), dtype=bool)
    for a, b in [(0, 1), (1, 2)]:
        adjacency[0, a, b] = adjacency[0, b, a] = True
    adjacency[1, 0, 1] = adjacency[1, 1, 0] = True
    present = np.array([[True, True, True, True], [True, True, False, True]])

    sizes = social_distance._connected_group_sizes(adjacency, present)

    np.testing.assert_array_equal(sizes[0], [3, 3, 3, 1])
    np.testing.assert_array_equal(sizes[1], [2, 2, np.nan, 1])


def test_equidistant_neighbors_rank_by_identity_order() -> None:
    """On a distance tie the earlier identity is closest, as in the original per-frame loop."""
    # identities 1 and 2 are the same hull distance from 0; both are in 0's field of view
    pose = _MultiFrameStubPose({0: [(0.0, 0.0)], 1: [(30.0, 0.0)], 2: [(0.0, 30.0)]})

    info = social_distance.ClosestIdentityInfo(pose, 0, 1.0)

    np.testing.assert_array_equal(info.closest_identities, [1])
    np.testing.assert_array_equal(info.closest_fov_identities, [1])
    np.testing.assert_array_equal(info.second_closest_distances, info.closest_distances)


@pytest.fixture(scope="module")
def sample_pose_v6() -> Iterator[PoseEstimation]:
    """Real v6 pose data (4 identities, with segmentation) from a temporary copy."""
    sample_v6 = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(sample_v6, Path(tmpdir) / sample_v6.name)
        yield pose_est.open_pose_file(Path(tmpdir) / sample_v6.name)


_INFO_PROPERTIES = [
    name
    for name in dir(ClosestIdentityInfo)
    if isinstance(getattr(ClosestIdentityInfo, name), property)
]


def test_contour_chunking_does_not_change_results(
    sample_pose_v6: PoseEstimation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contour distances computed in many small frame chunks equal a single chunk."""
    social_distance._VIDEO_SOCIAL_CACHES.pop(sample_pose_v6, None)
    expected = ClosestIdentityInfo(sample_pose_v6, 0, 1.0)

    social_distance._VIDEO_SOCIAL_CACHES.pop(sample_pose_v6, None)
    monkeypatch.setattr(social_distance, "_CONTOUR_CHUNK_FRAMES", 7)
    chunked = ClosestIdentityInfo(sample_pose_v6, 0, 1.0)
    social_distance._VIDEO_SOCIAL_CACHES.pop(sample_pose_v6, None)

    for name in _INFO_PROPERTIES:
        np.testing.assert_array_equal(getattr(chunked, name), getattr(expected, name), name)


@pytest.mark.parametrize(
    "pixel_scale",
    [0.07928075, np.float32(0.07928075), np.float64(0.07928075), 1, np.int64(1)],
    ids=["python_float", "numpy_float32", "numpy_float64", "python_int", "numpy_int"],
)
def test_scaled_distances_keep_scalar_precision(
    sample_pose_v6: PoseEstimation, pixel_scale: float
) -> None:
    """Scaled distances match the original per-frame ``python_float * pixel_scale``.

    Pose files store ``cm_per_pixel`` as ``np.float32``, which makes that product a
    float32 computation, while a Python float scale keeps float64 precision.
    """
    info = ClosestIdentityInfo(sample_pose_v6, 0, pixel_scale)
    hulls = {i: sample_pose_v6.get_identity_convex_hulls(i) for i in sample_pose_v6.identities}

    # frames where both skeletal hulls exist (others may use segmentation fallback hulls)
    checked = 0
    for frame, closest_id in enumerate(info.closest_identities):
        self_hull, other_hull = hulls[0][frame], hulls[int(closest_id)][frame]
        if closest_id == -1 or self_hull is None or other_hull is None:
            continue
        expected = np.float32(float(self_hull.distance(other_hull)) * pixel_scale)
        assert info.closest_distances[frame] == expected, frame
        checked += 1
    assert checked > 0


def _reference_pairwise_social_distances(
    poses: PoseEstimation,
    identity: int,
    pixel_scale: float,
    social_points: list[PoseEstimation.KeypointIndex],
    closest_identities: npt.NDArray[np.integer],
) -> dict[str, npt.NDArray[np.float32]]:
    """Original per-frame loop from compute_pairwise_social_distances, kept as a reference."""
    result = {
        f"social dist. {p1.name}-{p2.name}": np.full(poses.num_frames, np.nan, dtype=np.float32)
        for p1 in social_points
        for p2 in social_points
    }
    for frame in range(poses.num_frames):
        closest_id = int(closest_identities[frame])
        points, _ = poses.get_points(frame, identity, pixel_scale)
        if points is None or closest_id == -1:
            continue
        closest_points, _ = poses.get_points(frame, closest_id, pixel_scale)
        if closest_points is None:
            continue
        for p1 in social_points:
            for p2 in social_points:
                key = f"social dist. {p1.name}-{p2.name}"
                result[key][frame] = math.dist(points[p1.value], closest_points[p2.value])
    return result


@pytest.mark.parametrize(
    "pixel_scale",
    [0.07928075, np.float32(0.07928075), np.float64(0.07928075), 1],
    ids=["python_float", "numpy_float32", "numpy_float64", "python_int"],
)
@pytest.mark.parametrize("neighbor", ["closest", "closest_fov"])
def test_pairwise_social_distances_match_reference(
    sample_pose_v6: PoseEstimation, pixel_scale: float, neighbor: str
) -> None:
    """Pairwise social distances equal the original per-frame loop, for every identity."""
    points = [
        PoseEstimation.KeypointIndex.NOSE,
        PoseEstimation.KeypointIndex.BASE_NECK,
        PoseEstimation.KeypointIndex.BASE_TAIL,
    ]
    for identity in sample_pose_v6.identities:
        info = ClosestIdentityInfo(sample_pose_v6, identity, pixel_scale)
        closest = getattr(info, f"{neighbor}_identities")

        actual = info.compute_pairwise_social_distances(points, closest)
        expected = _reference_pairwise_social_distances(
            sample_pose_v6, identity, pixel_scale, points, closest
        )

        assert list(actual) == list(expected)
        for key, values in expected.items():
            np.testing.assert_array_equal(actual[key], values, err_msg=f"{identity} {key}")


def test_pairwise_social_distances_nan_without_neighbor_or_subject() -> None:
    """Frames with no neighbor, or where the subject or neighbor is absent, are NaN."""
    # frame 0: neighbor 1 present; frame 1: subject absent; frame 2: neighbor absent
    pose = _MultiFrameStubPose(
        {
            0: [(0.0, 0.0), None, (0.0, 0.0)],
            1: [(20.0, 0.0), (20.0, 0.0), None],
        }
    )
    info = ClosestIdentityInfo(pose, 0, 1.0)
    closest = np.array([1, 1, 1], dtype=np.int16)

    dists = info.compute_pairwise_social_distances(
        [PoseEstimation.KeypointIndex.BASE_NECK], closest
    )

    np.testing.assert_array_equal(
        dists["social dist. BASE_NECK-BASE_NECK"], [20.0, np.nan, np.nan]
    )


def test_torso_blob_morphological_opening_strips_tail_bridge_and_preserves_torso_contact() -> None:
    """Morphological opening (r = 15 px) removes 12-px tail bridges while preserving torso contact."""
    # Frame 0: Two 80x80 torsos separated by 120 px (x in [0, 80] and [200, 280]), bridged by
    # a 12-px-wide tail from Animal 0 reaching x=200 (raw contour gap = 0 px, opened torso gap ~ 120 px).
    # Frame 1: Two 80x80 torsos directly touching at x=80 (opened torso gap = 0 px, group size = 2).
    tail_bridge_ring = np.array(
        [
            [0, 0],
            [80, 0],
            [80, 34],
            [200, 34],
            [200, 46],
            [80, 46],
            [80, 80],
            [0, 80],
        ],
        dtype=np.int16,
    )
    torso_0_f1 = np.array(
        [[0, 0], [80, 0], [80, 80], [0, 80], [0, 0], [0, 0], [0, 0], [0, 0]],
        dtype=np.int16,
    )
    seg_0 = np.full((2, 1, 8, 2), -1, dtype=np.int16)
    seg_0[0, 0, :, :] = tail_bridge_ring
    seg_0[1, 0, :4, :] = torso_0_f1[:4]

    seg_1 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_1[0, 0, :, :] = np.array([[200, 0], [280, 0], [280, 80], [200, 80]], dtype=np.int16)
    seg_1[1, 0, :, :] = np.array([[80, 0], [160, 0], [160, 80], [80, 80]], dtype=np.int16)

    pose = _MultiFrameStubPose(
        {
            0: [(40.0, 40.0), (40.0, 40.0)],
            1: [(240.0, 40.0), (120.0, 40.0)],
        },
        hull_size=40.0,
        format_major_version=6,
        seg_contours_by_id={0: seg_0, 1: seg_1},
    )
    info = ClosestIdentityInfo(pose, identity=0, pixel_scale=1.0)
    dists = ClosestDistances(pose, 1.0, info).per_frame(0)
    huddle = HuddleFeatures(pose, 1.0, info).per_frame(0)

    # Frame 0 raw contour distance is 0.0 px due to the tail bridge, but opened torso distance > 110 px!
    np.testing.assert_allclose(dists["closest social segmentation distance"], [0.0, 0.0])
    assert float(huddle["torso blob distance"][0]) > 110.0
    assert float(huddle["torso blob distance"][1]) <= 2.5
    np.testing.assert_allclose(huddle["torso connected group size"], [1.0, 2.0])


def test_torso_blob_buried_third_mouse_preserves_effective_group_size_and_area() -> None:
    """When 2 visible masks merge a buried 3rd cagemate (>= 135k px^2), effective group size stays 3."""
    # Frame 0: All 3 mice have visible 240x240 masks (57,600 px^2 each) touching in a 3-mouse pile.
    # Frame 1: Mouse 2 burrows under Mouse 0 and Mouse 1; Mouse 0 and Mouse 1 masks absorb Mouse 2's
    # visible fur (360x240 each = 86,400 px^2, total hull = 720x240 = 172,800 px^2) while Mouse 2 is None.
    seg_0 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_0[0, 0, :, :] = np.array([[0, 0], [240, 0], [240, 240], [0, 240]], dtype=np.int16)
    seg_0[1, 0, :, :] = np.array([[0, 0], [360, 0], [360, 240], [0, 240]], dtype=np.int16)

    seg_1 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_1[0, 0, :, :] = np.array([[240, 0], [480, 0], [480, 240], [240, 240]], dtype=np.int16)
    seg_1[1, 0, :, :] = np.array([[360, 0], [720, 0], [720, 240], [360, 240]], dtype=np.int16)

    seg_2 = np.full((2, 1, 4, 2), -1, dtype=np.int16)
    seg_2[0, 0, :, :] = np.array([[480, 0], [720, 0], [720, 240], [480, 240]], dtype=np.int16)
    # Frame 1 of seg_2 stays -1 (buried/occluded)

    pose = _MultiFrameStubPose(
        {
            0: [(120.0, 120.0), (180.0, 120.0)],
            1: [(360.0, 120.0), (540.0, 120.0)],
            2: [(600.0, 120.0), None],
        },
        hull_size=40.0,
        format_major_version=6,
        seg_contours_by_id={0: seg_0, 1: seg_1, 2: seg_2},
    )
    info = ClosestIdentityInfo(pose, identity=0, pixel_scale=1.0)
    huddle = HuddleFeatures(pose, 1.0, info).per_frame(0)

    np.testing.assert_allclose(huddle["torso connected group size"], [3.0, 3.0])
    np.testing.assert_allclose(
        huddle["torso group blob hull area"][0],
        huddle["torso group blob hull area"][1],
        rtol=0.05,
    )
