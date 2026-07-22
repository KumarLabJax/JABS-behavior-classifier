"""End-to-end axis/format contract: write via jabs-io, read via the monolith v2 reader."""

import numpy as np

from jabs.core.enums import JabsPoseVersion
from jabs.core.types import PoseData
from jabs.io import save
from jabs.pose_estimation.pose_est_v2 import PoseEstimationV2


def _make_pose(num_frames=4):
    """Build a single-identity PoseData with distinct, known (x, y) coordinates."""
    n_kp = 12
    points = np.zeros((1, num_frames, n_kp, 2), dtype=np.float64)
    for f in range(num_frames):
        for k in range(n_kp):
            points[0, f, k] = [f * 10 + k, f * 10 + k + 3]  # distinct (x, y)
    return PoseData(
        points=points,
        point_mask=np.ones((1, num_frames, n_kp), dtype=bool),
        identity_mask=np.ones((1, num_frames), dtype=bool),
        body_parts=[f"kp{i}" for i in range(n_kp)],
        edges=[],
        fps=30,
        confidence=np.full((1, num_frames, n_kp), 0.9, dtype=np.float32),
    )


def test_v2_write_read_roundtrip_preserves_xy(tmp_path):
    """Points written by jabs-io are read back identically by the monolith reader."""
    pose = _make_pose()
    out = tmp_path / "clip_pose_est_v2.h5"
    save(pose, out, legacy=JabsPoseVersion.V2)

    reader = PoseEstimationV2(out, fps=30)
    points, point_mask = reader.get_identity_poses(0)

    # Monolith reads back (x, y) after its (y,x)->(x,y) flip; the double flip is identity.
    np.testing.assert_array_equal(points, pose.points[0])
    # confidence 0.9 > MINIMUM_CONFIDENCE (0.3) => all keypoints marked valid.
    assert point_mask.all()
    assert reader.format_major_version == 2


def test_v2_low_confidence_masked_out(tmp_path):
    """Confidence below the reader threshold results in an all-invalid point mask."""
    pose = _make_pose(num_frames=2)
    # frozen dataclass: replace confidence with a below-threshold array
    object.__setattr__(pose, "confidence", np.full((1, 2, 12), 0.1, dtype=np.float32))
    out = tmp_path / "clip_pose_est_v2.h5"
    save(pose, out, legacy=JabsPoseVersion.V2)

    reader = PoseEstimationV2(out, fps=30)
    _, point_mask = reader.get_identity_poses(0)
    assert not point_mask.any()  # 0.1 < 0.3
