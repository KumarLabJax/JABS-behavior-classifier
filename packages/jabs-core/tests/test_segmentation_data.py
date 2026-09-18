"""Tests for the SegmentationData dataclass and its use by PoseData."""

import numpy as np
import pytest

from jabs.core.types import PoseData, SegmentationData


def _seg_kwargs(num_idents=1, num_frames=3, num_contours=2, num_vertices=5):
    """Build minimal valid SegmentationData constructor kwargs."""
    return {
        "contours": np.zeros(
            (num_idents, num_frames, num_contours, num_vertices, 2), dtype=np.int32
        ),
        "vertex_counts": np.zeros((num_idents, num_frames, num_contours), dtype=np.uint32),
        "is_external": np.zeros((num_idents, num_frames, num_contours), dtype=bool),
    }


def _pose_kwargs(num_idents=1, num_frames=3, num_kp=12):
    """Build minimal valid PoseData constructor kwargs."""
    return {
        "points": np.zeros((num_idents, num_frames, num_kp, 2), dtype=np.float64),
        "point_mask": np.ones((num_idents, num_frames, num_kp), dtype=bool),
        "identity_mask": np.ones((num_idents, num_frames), dtype=bool),
        "body_parts": [f"kp{i}" for i in range(num_kp)],
        "edges": [],
        "fps": 30,
    }


def test_accepts_matching_shapes():
    """Arrays that agree on (identity, frame, contour) are accepted."""
    seg = SegmentationData(**_seg_kwargs())
    assert seg.contours.shape == (1, 3, 2, 5, 2)
    assert seg.vertex_counts.shape == (1, 3, 2)
    assert seg.is_external.shape == (1, 3, 2)


def test_contours_must_be_five_dimensional():
    """A contour array missing the contour-slot axis is rejected."""
    kw = _seg_kwargs()
    kw["contours"] = np.zeros((1, 3, 5, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="contours must have 5 dimensions"):
        SegmentationData(**kw)


def test_contours_last_axis_must_be_xy():
    """A contour array whose vertices are not 2-D points is rejected."""
    kw = _seg_kwargs()
    kw["contours"] = np.zeros((1, 3, 2, 5, 3), dtype=np.int32)
    with pytest.raises(ValueError, match=r"contours last dimension must be 2"):
        SegmentationData(**kw)


def test_vertex_counts_shape_mismatch_raises():
    """vertex_counts must cover exactly the contour slots that contours holds."""
    kw = _seg_kwargs()
    kw["vertex_counts"] = np.zeros((1, 3, 3), dtype=np.uint32)
    with pytest.raises(ValueError, match="vertex_counts shape"):
        SegmentationData(**kw)


def test_is_external_shape_mismatch_raises():
    """is_external must cover exactly the contour slots that contours holds."""
    kw = _seg_kwargs()
    kw["is_external"] = np.zeros((1, 3, 3), dtype=bool)
    with pytest.raises(ValueError, match="is_external shape"):
        SegmentationData(**kw)


def test_pose_data_segmentation_defaults_to_none():
    """segmentation_data is optional: most pose files carry no contours."""
    assert PoseData(**_pose_kwargs()).segmentation_data is None


def test_pose_data_accepts_matching_segmentation():
    """Segmentation covering the same identities and frames as points is accepted."""
    kw = _pose_kwargs(num_idents=2, num_frames=4)
    kw["segmentation_data"] = SegmentationData(**_seg_kwargs(num_idents=2, num_frames=4))
    pose = PoseData(**kw)
    assert pose.segmentation_data.contours.shape[:2] == (2, 4)


@pytest.mark.parametrize(
    ("seg_idents", "seg_frames"),
    [(3, 4), (2, 5)],
    ids=["identity_mismatch", "frame_mismatch"],
)
def test_pose_data_segmentation_shape_mismatch_raises(seg_idents, seg_frames):
    """Segmentation that does not line up with points is rejected."""
    kw = _pose_kwargs(num_idents=2, num_frames=4)
    kw["segmentation_data"] = SegmentationData(
        **_seg_kwargs(num_idents=seg_idents, num_frames=seg_frames)
    )
    with pytest.raises(ValueError, match="segmentation_data covers"):
        PoseData(**kw)
