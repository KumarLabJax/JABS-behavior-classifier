import typing
from pathlib import Path
import numpy as np
import h5py

from .pose_est_v4 import PoseEstimationV4

OBJECTS_STORED_YX = [
    'lixit',
    'food_hopper',
]

class PoseEstimationV5(PoseEstimationV4):
    def __init__(self, file_path: Path,
                 cache_dir: typing.Optional[Path] = None,
                 fps: int = 30):
        """
        :param file_path: Path object representing the location of the pose file
        :param cache_dir: optional cache directory, used to cache convex hulls
        for faster loading
        :param fps: frames per second, used for scaling time series features
        from "per frame" to "per second"
        """
        super().__init__(file_path, cache_dir, fps)

        # V5 files are the same as V4, except they have some additional datasets
        # in addition to the posest data. The pose data is all loaded from
        # calling super().__init__(), so now we just need to load the additional
        # data

        self._static_objects = {}

        # open the hdf5 pose file
        with h5py.File(self._path, 'r') as pose_h5:
            # extract data from the HDF5 file
            for g in pose_h5:
                # skip over the poseest group, since that's already been
                # processed
                if g == 'poseest':
                    continue

                # 'static_objects'. Currently anything else is ignored
                if g == 'static_objects':
                    for d in pose_h5['static_objects']:
                        static_object_data = pose_h5['static_objects'][d][:]
                        if d in OBJECTS_STORED_YX:
                            static_object_data = np.flip(static_object_data, axis=-1)
                        self._static_objects[d] = static_object_data

        # drop "lixit" from the static objects if it is an empty array
        if 'lixit' in self._static_objects and self._static_objects['lixit'].shape[0] == 0:
            del self._static_objects['lixit']

    @property
    def format_major_version(self) -> int:
        return 5
