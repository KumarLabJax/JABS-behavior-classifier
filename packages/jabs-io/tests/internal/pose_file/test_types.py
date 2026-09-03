"""Tests for the pose file domain types."""

import numpy as np
import pytest

from jabs.io.internal.pose_file.types import Component, Skeleton


def _points(frames=4, slots=1):
    return np.zeros((frames, slots, 12, 2), dtype=np.float32)


def test_component_path_for_jabs_namespace():
    """A jabs component maps its id onto a nested group path."""
    c = Component(
        id="jabs.pose.points",
        axes=("frame", "slot", "keypoint", "coord"),
        data=_points(),
        units="pixel",
        coord_order="xy",
        missing={"policy": "nan"},
    )
    assert c.path == "/jabs/pose/points"
    assert c.dtype == "float32"


def test_component_path_for_reverse_dns_namespace():
    """A foreign component keeps its reverse-DNS root as one path element."""
    c = Component(
        id="org.jax.gait.stride_length",
        axes=("frame", "slot"),
        data=np.zeros((4, 1), dtype=np.float32),
        missing={"policy": "nan"},
    )
    assert c.path == "/org.jax.gait/stride_length"


def test_component_rejects_axes_arity_mismatch():
    """Declared axes must name every dimension of the array."""
    with pytest.raises(ValueError, match="axes"):
        Component(
            id="jabs.pose.points",
            axes=("frame", "slot"),
            data=_points(),
            missing={"policy": "nan"},
        )


def test_component_rejects_bad_id():
    """Ids are lowercase, dot-separated and at least two segments."""
    with pytest.raises(ValueError, match="component id"):
        Component(id="Jabs.Pose", axes=("frame",), data=np.zeros(4), missing={"policy": "none"})


def test_component_rejects_foreign_namespace_without_reverse_dns():
    """A non-jabs root must have at least two segments, so it cannot collide."""
    with pytest.raises(ValueError, match="reverse-DNS"):
        Component(
            id="gait.stride",
            axes=("frame",),
            data=np.zeros(4, dtype=np.float32),
            missing={"policy": "nan"},
        )


def test_component_requires_coord_order_on_coord_axis():
    """Coordinates must declare their order; the legacy format's worst trait."""
    with pytest.raises(ValueError, match="coord_order"):
        Component(
            id="jabs.pose.points",
            axes=("frame", "slot", "keypoint", "coord"),
            data=_points(),
            units="pixel",
            missing={"policy": "nan"},
        )


def test_component_requires_sparse_index_with_sample_axis():
    """A sample axis is meaningless without the index that maps it to frames."""
    with pytest.raises(ValueError, match="sparse"):
        Component(
            id="jabs.dynamic_objects.fecal_boli.counts",
            axes=("sample",),
            data=np.zeros(3, dtype=np.uint32),
            missing={"policy": "none"},
        )


def test_component_rejects_sparse_index_without_sample_axis():
    """And the pairing holds in the other direction too."""
    with pytest.raises(ValueError, match="sparse"):
        Component(
            id="jabs.pose.slot_occupied",
            axes=("frame", "slot"),
            data=np.zeros((4, 1), dtype=bool),
            missing={"policy": "none"},
            sparse_index="jabs.dynamic_objects.fecal_boli.frame_index",
        )


def test_skeleton_rejects_out_of_range_edge():
    """An edge cannot name a keypoint the skeleton does not have."""
    with pytest.raises(ValueError, match="edge"):
        Skeleton(body_parts=("NOSE", "TAIL"), edges=((0, 5),))
