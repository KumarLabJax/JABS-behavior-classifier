"""Tests for the PoseData dataclass, focused on the confidence field."""

import numpy as np
import pytest

from jabs.core.types import PoseData


def _kwargs(num_idents=1, num_frames=3, num_kp=12):
    """Build minimal valid PoseData constructor kwargs (no confidence)."""
    return {
        "points": np.zeros((num_idents, num_frames, num_kp, 2), dtype=np.float64),
        "point_mask": np.ones((num_idents, num_frames, num_kp), dtype=bool),
        "identity_mask": np.ones((num_idents, num_frames), dtype=bool),
        "body_parts": [f"kp{i}" for i in range(num_kp)],
        "edges": [],
        "fps": 30,
    }


def test_confidence_defaults_to_none():
    """confidence is optional and defaults to None."""
    pose = PoseData(**_kwargs())
    assert pose.confidence is None


def test_confidence_accepts_matching_shape():
    """A confidence array matching point_mask's shape is accepted."""
    kw = _kwargs()
    kw["confidence"] = np.full((1, 3, 12), 0.9, dtype=np.float32)
    pose = PoseData(**kw)
    assert pose.confidence.shape == (1, 3, 12)


def test_confidence_wrong_shape_raises():
    """A confidence array whose shape does not match points is rejected."""
    kw = _kwargs()
    kw["confidence"] = np.full((1, 3, 11), 0.9, dtype=np.float32)
    with pytest.raises(ValueError, match="confidence shape"):
        PoseData(**kw)
