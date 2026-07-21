"""Tests for the legacy v2 PoseData HDF5 writer."""

import h5py
import numpy as np
import pytest

from jabs.core.enums import JabsPoseVersion, StorageFormat
from jabs.core.types import PoseData
from jabs.io import save
from jabs.io.internal.pose.hdf5 import PoseHDF5Adapter
from jabs.io.registry import get_adapter


def _make_pose(num_frames=3, num_idents=1):
    """Build a single-identity PoseData with distinct, known coordinates."""
    n_kp = 12
    # distinct (x, y) per (frame, keypoint): x = f*100 + k, y = f*100 + k + 1
    points = np.zeros((num_idents, num_frames, n_kp, 2), dtype=np.float64)
    for f in range(num_frames):
        for k in range(n_kp):
            points[0, f, k] = [f * 100 + k, f * 100 + k + 1]
    return PoseData(
        points=points,
        point_mask=np.ones((num_idents, num_frames, n_kp), dtype=bool),
        identity_mask=np.ones((num_idents, num_frames), dtype=bool),
        body_parts=[f"kp{i}" for i in range(n_kp)],
        edges=[],
        fps=30,
        confidence=np.full((num_idents, num_frames, n_kp), 0.9, dtype=np.float32),
        metadata={"config": "gait-model.yaml", "model": "gait-model.pth"},
    )


@pytest.fixture
def adapter():
    """Return a PoseHDF5Adapter instance."""
    return PoseHDF5Adapter()


def test_writes_v2_layout(adapter, tmp_path):
    """The adapter writes the legacy v2 layout with correct dtypes and axis order."""
    out = tmp_path / "x_pose_est_v2.h5"
    adapter.write(_make_pose(), out, legacy=JabsPoseVersion.V2)

    with h5py.File(out, "r") as h5:
        pose = h5["poseest"]
        assert pose.attrs["version"].tolist() == [2, 0]
        assert pose["points"].dtype == np.uint16
        assert pose["points"].shape == (3, 12, 2)
        assert pose["confidence"].dtype == np.float32
        assert pose["confidence"].shape == (3, 12)
        # canonical (x, y) written as on-disk (y, x): frame 0, kp 0 was (0, 1) -> (1, 0)
        assert pose["points"][0, 0].tolist() == [1, 0]
        assert pose["points"].attrs["config"] == "gait-model.yaml"
        assert pose["points"].attrs["model"] == "gait-model.pth"


def test_save_dispatches_to_pose_adapter(tmp_path):
    """The generic save() routes PoseData + .h5 to the pose adapter."""
    out = tmp_path / "y_pose_est_v2.h5"
    save(_make_pose(), out, legacy=JabsPoseVersion.V2)
    with h5py.File(out, "r") as h5:
        assert "poseest/points" in h5


def test_registry_resolves_pose_hdf5_adapter():
    """The registry resolves (HDF5, PoseData) to PoseHDF5Adapter."""
    resolved = get_adapter(StorageFormat.HDF5, PoseData)
    assert isinstance(resolved, PoseHDF5Adapter)


def test_can_handle_truth_table():
    """can_handle is True only for PoseData."""
    assert PoseHDF5Adapter.can_handle(PoseData) is True
    assert PoseHDF5Adapter.can_handle(dict) is False


def test_unsupported_legacy_version_raises(adapter, tmp_path):
    """A legacy version other than V2 is rejected."""
    with pytest.raises(ValueError, match="Unsupported legacy pose version"):
        adapter.write(_make_pose(), tmp_path / "z.h5", legacy=JabsPoseVersion.V3)


def test_multi_identity_raises(adapter, tmp_path):
    """v2 is single-identity; multi-identity input is rejected."""
    with pytest.raises(ValueError, match="single-identity"):
        adapter.write(_make_pose(num_idents=2), tmp_path / "z.h5")


def test_missing_confidence_raises(adapter, tmp_path):
    """v2 requires confidence; None is rejected."""
    pose = _make_pose()
    object.__setattr__(pose, "confidence", None)  # frozen dataclass
    with pytest.raises(ValueError, match="requires confidence"):
        adapter.write(pose, tmp_path / "z.h5")


def test_read_not_implemented(adapter, tmp_path):
    """read is intentionally not implemented in this increment."""
    out = tmp_path / "x_pose_est_v2.h5"
    adapter.write(_make_pose(), out)
    with pytest.raises(NotImplementedError):
        adapter.read(out)
