import numpy as np
import typing
from pathlib import Path
import h5py

from .pose_est_v5 import PoseEstimationV5


class PoseEstimationV6(PoseEstimationV5):
    '''Version 6 of the Pose Estimation class.
    '''
    # Does __cache_file_version need to be updated?

    def __init__(self, file_path: Path,
                 cache_dir: typing.Optional[Path] = None,
                 fps: int = 30):
        """
        :param file_path: Path object representing the location of the pose
        file.
        :param cache_dir: optional cache directory, used to cache convex hulls
        for faster loading
        :param fps: frames per second, used for scaling time series features
        from "per frame" to "per second"
        """
        super().__init__(file_path, cache_dir, fps)

        # v6 properties
        # Image segmentation data read from pose v6 files.
        self._segmentation_dict = {
            'instance_seg_id': None, 'longterm_seg_id': None,
            'seg_external_flag': None, 'seg_data': None
        }

        # open the hdf5 pose file and extract segmentation data.
        with h5py.File(self._path, 'r') as pose_h5:
            for seg_key in set(pose_h5["poseest"].keys()) & set(
                    self._segmentation_dict.keys()):
                self._segmentation_dict[seg_key] = \
                    pose_h5[f"poseest/{seg_key}"][:]
            # transpose seg_data similar to the way the points are transposed.

        # sort the segmentation data
        print("new sorting initiated")
        self._segmentation_dict["seg_data"] = self._segmentation_sort(self._segmentation_dict["seg_data"], self._segmentation_dict["longterm_seg_id"])
    
    def get_seg_id(self, frameIndex: int, identity: int) -> int:
        if self._segmentation_dict["longterm_seg_id"] is None:
            return None
        else:
            return self._segmentation_dict['longterm_seg_id'][frameIndex, identity]
    
    @classmethod
    def _segmentation_sort(cls, seg_data: np.ndarray, longterm_seg_id: np.ndarray) -> np.ndarray:
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
                    B[b_index, :] = seg_data[frame, a_index, :] # B[a_index, :] = seg_data[frame, b_index, :] 

            seg_data_tmp[frame, ...] = B  
        
        return seg_data_tmp

    def get_segmentation_data(self, identity: int) -> np.ndarray:
        ''' Given a particular identity, return the appropriate segmentation
        data.
        :param identity: identity to return segmentation data for.
        :return: the ndarray of segmentation data (if it exists) otherwise the 
            function returns None.
        '''

        if self._segmentation_dict["seg_data"] is None:
            return None
        else:
            return self._segmentation_dict['seg_data'][:, identity, ...]
    
    def get_segmentation_data_per_frame(self, frameIndex, identity: int) -> np.ndarray:
        ''' Given a particular identity, return the appropriate segmentation
        data.
        :param identity: identity to return segmentation data for.
        :return: the ndarray of segmentation data (if it exists) otherwise the 
            function returns None.
        '''

        if self._segmentation_dict["seg_data"] is None:
            return None
        else:
            return self._segmentation_dict['seg_data'][frameIndex, identity, ...]

    @property
    def format_major_version(self) -> int:
        return 6
