"""Segmentation windows as MomentInfo consumes them.

``MomentInfo`` appends one frame per window row, so ``iter_segmentation`` must
tile ``[0, num_frames)`` with contiguous, in-order windows; and
``MomentInfo.frame_contours`` must keep each contour's external flag aligned
with it when contour slots are skipped.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from jabs.core.types import SegmentationWindow
from jabs.feature_extraction.segmentation_features.moment_cache import MomentInfo
from jabs.pose_estimation.conversion_cache import open_unified
from jabs.pose_estimation.pose_est_v6 import PoseEstimationV6

_SAMPLE_V6 = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"

# does not divide the sample's 3600 frames, so the last window is partial
_BLOCK = 7


@pytest.fixture(scope="module", params=["legacy_v6", "unified"])
def pose(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator:
    """The sample v6 file through the legacy V6 reader and through ``Pose``."""
    root = tmp_path_factory.mktemp(request.param)
    path = root / _SAMPLE_V6.name
    shutil.copyfile(_SAMPLE_V6, path)
    if request.param == "legacy_v6":
        yield PoseEstimationV6(path)
    else:
        with open_unified(path, root / "cache") as unified:
            yield unified


def test_iter_segmentation_tiles_all_frames_in_order(pose) -> None:
    """Windows start where the previous one stopped and end exactly at num_frames."""
    assert pose.num_frames % _BLOCK != 0
    for identity in pose.identities:
        expected_start = 0
        lengths = []
        for window in pose.iter_segmentation(identity, block=_BLOCK):
            n = len(window.contour_count)
            assert window.start == expected_start
            assert window.contours.shape[0] == n
            assert window.contour_length.shape[0] == n
            assert 0 < n <= _BLOCK
            expected_start += n
            lengths.append(n)
        assert expected_start == pose.num_frames
        assert lengths[-1] == pose.num_frames % _BLOCK


def test_frame_contours_keeps_flags_aligned_across_an_empty_contour() -> None:
    """An empty contour slot between valid ones is dropped together with its flag."""
    contours = np.zeros((1, 3, 4, 2), dtype=np.int32)
    contours[0, 0, :4] = [[0, 0], [4, 0], [4, 4], [0, 4]]  # external
    contours[0, 2, :3] = [[1, 1], [2, 1], [2, 2]]  # hole
    window = SegmentationWindow(
        contours=contours,
        contour_count=np.array([3], dtype=np.uint32),
        contour_length=np.array([[4, 0, 3]], dtype=np.uint32),
        external_flag=np.array([[True, True, False]]),
        start=0,
    )

    kept, flags = MomentInfo.frame_contours(window, 0)

    assert len(kept) == 2
    np.testing.assert_array_equal(kept[0], contours[0, 0, :4])
    np.testing.assert_array_equal(kept[1], contours[0, 2, :3])
    np.testing.assert_array_equal(flags, [True, False])
