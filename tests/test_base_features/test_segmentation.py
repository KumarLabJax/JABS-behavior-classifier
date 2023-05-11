import unittest 
import h5py
import numpy as np

from pathlib import Path

def ugly_segmentation_sort(seg_data: np.ndarray, longterm_seg_id: np.ndarray) -> np.ndarray:
    """
    This method attempts to sort the segmentation data according to the longterm segmentation id.  
    This code is highly inefficient and ugly should be replaced with a vectorized expression.

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
        print(self.seg_data.shape, self.longterm_seg_sorting.shape, self.instance_seg_sorting.shape, self.track_sorting.shape)
        #print(self.all_points.shape, self.id_mask.shape, self.embed_sorting.shape)
        #extended_id_mask = np.hstack((self.id_mask, np.zeros((self.id_mask.shape[0],1), dtype=np.bool8)))

        #print(self.longterm_seg_sorting, self.longterm_seg_sorting.dtype)
        
        # Attempt 1
        if False:
            # print(self.longterm_seg_sorting.dtype)
            # self.longterm_seg_sorting[:,self.seg_data.shape[1] - 1] = self.seg_data.shape[1] - 1
            # print(seg_data_tmp.shape, self.seg_data.shape)
            seg_data_tmp = np.zeros_like(self.seg_data)
            for frame in range(self.seg_data.shape[0]):
                seg_data_tmp[frame,...] = self.seg_data[frame, self.longterm_seg_sorting[frame], ...]

            idx = 1
            frame = 950
            # print(seg_data_tmp[frame, :, 0, 0, :])
            print("-"*10, self.seg_data[frame, :, 0, 0, :], self.longterm_seg_sorting[frame], sep="\n")

        # Attempt 2
        # ugly, but should work
        if False:

            seg_data_tmp = np.zeros_like(self.seg_data) # np.full_like(self.seg_data, -1)

            for frame in range(self.seg_data.shape[0]):
                map = self.longterm_seg_sorting[frame]
                A = self.seg_data[frame, ...]
                B = np.full_like(self.seg_data[frame, ...], -1)
                for a_index in range(len(map)):
                    b_index = (map-1)[a_index]
                    if A.shape[0] > b_index >= 0:
                        B[b_index, :] = A[a_index, :]
                seg_data_tmp[frame, ...] = B  


            print(seg_data_tmp[950,:, 0, 0, :])   

        # Attempt 3, put into a function.
        if True:
            frame = 5
            sorted_seg_dat = ugly_segmentation_sort(self.seg_data, self.longterm_seg_sorting)
            print("SEG ID:")
            print(self.longterm_seg_sorting[frame])
            print("RAW:")
            print(self.seg_data[frame,:, 0, 0, :])
            print("SORTED:")
            print(sorted_seg_dat[frame, :, 0, 0, :])

        if False:
            seg_data_tmp[np.where(extended_id_mask == 0)[0],
                self.instance_seg_sorting[extended_id_mask == 0] - 1, :, :] = self.seg_data[extended_id_mask == 0, :, :]
            print(np.unique(seg_data_tmp))
        
        if False:
            points_tmp = np.zeros_like(self.all_points)
            points_tmp[np.where(self.id_mask == 0)[0],
                    self.embed_sorting[self.id_mask == 0] - 1, :, :] = self.all_points[
                                                                self.id_mask == 0, :, :]
    
        # print(np.unique(points_tmp))
        
        assert True


if __name__ == "__main__": pass