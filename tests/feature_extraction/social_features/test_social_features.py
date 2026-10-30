"""Tests for social distance, field-of-view gating, and multi-animal huddle features."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

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
