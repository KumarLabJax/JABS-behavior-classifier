from pathlib import Path
import typing

import h5py
import numpy as np

from .pose_est import PoseEstimation


class PoseEstimationV2(PoseEstimation):
    """
    read in pose_est_v2.h5 file
    """

    def __init__(self, file_path: Path, cache_dir: typing.Optional[Path]=None):
        """
        initialize new object from h5 file
        :param file_path: path to pose_est_v2.h5 file
        """

        super().__init__(file_path, cache_dir)

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
            self._point_mask = np.zeros(self._points.shape[:-1], dtype=np.uint16)
            self._point_mask[:] = vid_grp['confidence'][:] > 0.3

        self._num_frames = self._points.shape[0]

        # build an array that indicates if the identity exists for a each frame
        # require at least one body point, not just tail
        init_func = np.vectorize(
            lambda x: 0 if np.sum(self._point_mask[x][:-2]) == 0 else 1,
            otypes=[np.uint8])
        self._identity_mask = np.fromfunction(init_func, (self._num_frames,),
                                              dtype=np.int_)

    @property
    def identity_to_track(self):
        identity_to_track = np.full((1, self._num_frames), -1,
                                    dtype=np.int32)
        identity_to_track[0, self._identity_mask == 1] = 0
        return identity_to_track

    @property
    def format_major_version(self):
        return 2

    def get_points(self, frame_index, identity):
        """
        return points and point masks for an individual frame
        :param frame_index: frame index of points and masks to be returned
        :param identity: included for compatibility with pose_est_v3. Should
        always be zero.
        :return: numpy array of points (12,2), numpy array of point masks (12,)
        """
        if identity not in self.identities:
            raise ValueError("Invalid identity")

        if not self._identity_mask[frame_index]:
            return None, None

        return self._points[frame_index], self._point_mask[frame_index]

    def get_identity_poses(self, identity):
        """
        return all points and point masks
        :param identity: included for compatibility with pose_est_v3. Should
        always be zero.
        :return: numpy array of points (#frames, 12, 2), numpy array of point
        masks (#frames, 12)
        """
        if identity not in self.identities:
            raise ValueError("Invalid identity")
        return self._points, self._point_mask

    def identity_mask(self, identity):
        """
        get the identity mask (indicates if specified identity is present in
        each frame)
        :param identity: included for compatibility with pose_est_v3. Should
        always be zero.
        :return: numpy array of size (#frames,)
        """
        if identity not in self.identities:
            raise ValueError("Invalid identity")
        return self._identity_mask

    def get_identity_point_mask(self, identity):
        """
        get the point mask array for a given identity
        :param identity: identity to return point mask for
        :return: array of point masks (#frames, 12)
        """
        return self._point_mask

    @classmethod
    def instance_count_from_file(cls, path: Path) -> int:
        return 1
