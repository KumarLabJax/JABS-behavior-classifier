import unittest 
import h5py
import numpy as np

from pathlib import Path

	
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
        if True:
            print(self.longterm_seg_sorting.dtype)
            # self.longterm_seg_sorting[:,self.seg_data.shape[1] - 1] = self.seg_data.shape[1] - 1
            
            seg_data_tmp = np.zeros_like(self.seg_data)
            for frame in range(self.seg_data.shape[0]):
                #print(self.longterm_seg_sorting[frame])
                seg_data_tmp[frame,...] = self.seg_data[frame, self.longterm_seg_sorting[frame], ...]


            idx = 1
            print(seg_data_tmp.shape, self.seg_data.shape)
            frame = 950
            print(seg_data_tmp[frame, :, 0, 0, :],"-"*10, self.seg_data[frame, :, 0, 0, :], self.longterm_seg_sorting[frame], sep="\n")

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