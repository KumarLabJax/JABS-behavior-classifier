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


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64], ids=str)
def test_accepts_any_signed_integer_width(dtype):
    """Contours keep whatever signed width the source used; nothing is widened for us."""
    kw = _seg_kwargs()
    kw["contours"] = kw["contours"].astype(dtype)
    assert SegmentationData(**kw).contours.dtype == dtype


@pytest.mark.parametrize("dtype", [np.uint16, np.float32], ids=str)
def test_contours_must_be_signed_integers(dtype):
    """Unsigned and floating contours are rejected: the -1 padding sentinel needs a sign.

    An unsigned array wraps -1 to its maximum value, which reads as a real coordinate
    and silently inflates every vertex count derived from the padding.
    """
    kw = _seg_kwargs()
    kw["contours"] = kw["contours"].astype(dtype)
    with pytest.raises(ValueError, match="contours must be a signed integer array"):
        SegmentationData(**kw)


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


def test_vertex_counts_may_not_exceed_the_vertex_capacity():
    """A count past the slot's capacity describes a contour that cannot exist."""
    kw = _seg_kwargs(num_vertices=5)
    kw["vertex_counts"] = np.full((1, 3, 2), 6, dtype=np.uint32)
    with pytest.raises(ValueError, match="must not exceed 5"):
        SegmentationData(**kw)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint16, np.float64])
def test_vertex_counts_must_be_uint32(dtype):
    """vertex_counts is documented and typed as uint32, so other dtypes are rejected."""
    kw = _seg_kwargs(num_vertices=5)
    kw["vertex_counts"] = np.ones((1, 3, 2), dtype=dtype)
    with pytest.raises(ValueError, match="uint32"):
        SegmentationData(**kw)


def test_vertex_counts_may_fill_the_slot_exactly():
    """A count equal to the vertex capacity is valid."""
    kw = _seg_kwargs(num_vertices=5)
    kw["vertex_counts"] = np.full((1, 3, 2), 5, dtype=np.uint32)
    assert SegmentationData(**kw).vertex_counts.max() == 5


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8, np.float64])
def test_is_external_must_be_boolean(dtype):
    """Integer flags, such as the pose reader's -1 sentinel, are rejected rather than cast."""
    kw = _seg_kwargs()
    kw["is_external"] = (
        np.full((1, 3, 2), -1).astype(dtype)
        if dtype != np.uint8
        else np.ones((1, 3, 2), dtype=dtype)
    )
    with pytest.raises(ValueError, match="boolean array"):
        SegmentationData(**kw)
