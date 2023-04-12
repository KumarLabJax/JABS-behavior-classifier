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

    def get_segmentation_data(self, identity: int) -> np.ndarray:
        ''' Given a particular identity, return the appropriate segmentation
        data.
        :param identity: identity to return segmentation data for.
        :return: the ndarray of segmentation data (if it exists) otherwise the 
            function returns None.
        '''
        #print("point shape:",self._points.shape, self._segmentation_dict["seg_data"].shape)
        if self._segmentation_dict["seg_data"] is None:
            return None
        else:
            return self._segmentation_dict['seg_data'][:, identity, ...]

    @property
    def format_major_version(self) -> int:
        return 6