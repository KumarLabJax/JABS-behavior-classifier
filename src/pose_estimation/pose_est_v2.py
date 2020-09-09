from pathlib import Path
import numpy as np
import h5py

from .pose_est import PoseEstimation


class PoseEstimationV2(PoseEstimation):
    """

    """
    def __init__(self, file_path: Path):
        """

        :param path:
        """

        # we will make this look like the PoseEstimationV3 but with a single
        # identity so the main program won't care which type it is
        self._identities = [0]
        self._max_instances = 1

        self._path = file_path

        # open the hdf5 pose file
        with h5py.File(self._path, 'r') as pose_h5:
            # extract data from the HDF5 file
            vid_grp = pose_h5['poseest']

            # load contents
            self._points = vid_grp['points'][:]
            self._point_mask = np.zeros(self._points.shape[:-1], dtype="uint16")
            self._point_mask[:] = vid_grp['confidence'][:] > 0.3

        self._num_frames = self._points.shape[0]

        # build an array that indicates if the identity exists for a each frame
        # require at least a body point
        init_func = np.vectorize(
            lambda x: 0 if np.sum(self._point_mask[x][:-2]) == 0 else 1,
            otypes=["uint8"])
        self._identity_mask = np.fromfunction(init_func, (self._num_frames,),
                                              dtype="int")

    def get_points(self, frame_index, identity):
        if identity not in self.identities:
            raise ValueError("Invalid identity")

        if not self._identity_mask[frame_index]:
            return None, None

        return self._points[frame_index], self._point_mask[frame_index]

    def get_identity_poses(self, identity):
        if identity not in self.identities:
            raise ValueError("Invalid identity")
        return self._points, self._point_mask

    def identity_mask(self, identity):
        if identity not in self.identities:
            raise ValueError("Invalid identity")
        return self._identity_mask
