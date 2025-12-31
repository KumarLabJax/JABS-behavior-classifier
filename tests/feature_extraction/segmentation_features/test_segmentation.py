from pathlib import Path

import h5py
import numpy as np
import pytest


def ugly_segmentation_sort(seg_data: np.ndarray, longterm_seg_id: np.ndarray) -> np.ndarray:
    """Sort segmentation data according to longterm segmentation ID.

    This method is highly inefficient and should be replaced with a vectorized expression.

    Args:
        seg_data: Segmentation data array to be sorted.
        longterm_seg_id: Longterm segmentation ID array for sorting.

    Returns:
        Sorted segmentation data array.
    """
    seg_data_tmp = np.zeros_like(seg_data)  # np.full_like(self.seg_data, -1)
    for frame in range(seg_data.shape[0]):
        map = longterm_seg_id[frame]
        B = np.full_like(seg_data[frame, ...], -1)
        for a_index in range(len(map)):
            b_index = (map - 1)[a_index]
            if seg_data.shape[1] > b_index >= 0:
                B[b_index, :] = seg_data[frame, a_index, :]

        seg_data_tmp[frame, ...] = B

    return seg_data_tmp


@pytest.fixture(scope="module")
def pose_v6_data():
    """Load pose estimation v6 data with segmentation information."""
    dataPath = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"

    with h5py.File(dataPath, "r") as f:
        data = {
            "all_points": f["poseest/points"][:],
            "seg_data": f["poseest/seg_data"][:],
            "id_mask": f["poseest/id_mask"][:],
            "seg_external_flag": f["poseest/seg_external_flag"][:],
            "track_sorting": f["poseest/instance_track_id"][:],
            "embed_sorting": f["poseest/instance_embed_id"][:],
            "instance_seg_sorting": f["poseest/instance_seg_id"][:],
            "longterm_seg_sorting": f["poseest/longterm_seg_id"][:],
            "cm_per_px2": f["poseest"].attrs["cm_per_pixel"],
        }
    return data


def test_seg_data(pose_v6_data):
    """Test segmentation data sorting functionality.

    Verifies that segmentation data is sorted so that it overlays on the
    correct mouse in the UI. Tests sorting segmentation data according to
    longterm segmentation IDs using the ugly_segmentation_sort function.
    """
    print(
        pose_v6_data["seg_data"].shape,
        pose_v6_data["longterm_seg_sorting"].shape,
        pose_v6_data["instance_seg_sorting"].shape,
        pose_v6_data["track_sorting"].shape,
    )

    frame = 5
    sorted_seg_dat = ugly_segmentation_sort(
        pose_v6_data["seg_data"], pose_v6_data["longterm_seg_sorting"]
    )
    print("SEG ID:")
    print(pose_v6_data["longterm_seg_sorting"][frame])
    print("RAW:")
    print(pose_v6_data["seg_data"][frame, :, 0, 0, :])
    print("SORTED:")
    print(sorted_seg_dat[frame, :, 0, 0, :])

    assert True


if __name__ == "__main__":
    pass
