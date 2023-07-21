import unittest 
import h5py
import numpy as np

from pathlib import Path

def segmentation_sort(seg_data: np.ndarray, longterm_seg_id: np.ndarray) -> np.ndarray:
    """
    This method attempts to sort the segmentation data according to the longterm segmentation id.  
    This code is inefficient and ideally should be replaced with a vectorized expression.

    :return: sorted segmentation data
    """
    seg_data_tmp = np.zeros_like(seg_data) # np.full_like(self.seg_data, -1)
    for frame in range(seg_data.shape[0]):
        map = longterm_seg_id[frame]
        B = np.full_like(seg_data[frame, ...], -1)
        for a_index in range(len(map)):
            b_index = (map-1)[a_index]
            if seg_data.shape[1] > b_index >= 0:
                B[b_index, :] = seg_data[frame, a_index, :]

        seg_data_tmp[frame, ...] = B  
    
    return seg_data_tmp


class TestPoseMatchSegmentation(unittest.TestCase):
    """
    Ensure that the segmentation data is sorted properly so that 
    it overlays on the correct mouse in the UI.
    """

    @classmethod
    def setUpClass(cls) -> None:

        dataPath = Path(__file__).parent.parent.parent / "data" / "B6J_MDB0054_pose_est_v6.h5"

        with h5py.File(dataPath, 'r') as f:
            print(f['poseest'].keys())
            cls.all_points = f['poseest/points'][:]
            cls.seg_data = f['poseest/seg_data'][:]
            cls.id_mask = f['poseest/id_mask'][:]
            cls.seg_external_flag = f['poseest/seg_external_flag'][:]
            cls.track_sorting = f['poseest/instance_track_id'][:]
            cls.embed_sorting = f['poseest/instance_embed_id'][:]
            cls.instance_seg_sorting = f['poseest/instance_seg_id'][:]
            cls.longterm_seg_sorting = f['poseest/longterm_seg_id'][:]
            cls.cm_per_px2 = f['poseest'].attrs['cm_per_pixel']
        return super().setUpClass()


    def test_seg_data(self):
        frame = 5
        sorted_seg_dat = segmentation_sort(self.seg_data, self.longterm_seg_sorting)
        print("SEG ID:")
        print(self.longterm_seg_sorting[frame])
        print("RAW:")
        print(self.seg_data[frame,:, 0, 0, :])
        print("SORTED:")
        print(sorted_seg_dat[frame, :, 0, 0, :])
        assert True


if __name__ == "__main__": pass