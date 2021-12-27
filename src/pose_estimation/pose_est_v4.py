import typing
from pathlib import Path

import h5py
import numpy as np

from .pose_est import PoseEstimation, PoseHashException


class _CacheFileVersion(Exception):
    pass


class PoseEstimationV4(PoseEstimation):
    """
    class for opening and parsing version 4 of the pose estimation HDF5 file
    """

    __CACHE_FILE_VERSION = 2

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

        # these are not relevant for v4 pose files, but are included
        self._identity_to_track = None
        self._identity_map = None

        use_cache = False
        if cache_dir is not None:
            try:
                self._load_from_cache()
                use_cache = True
            except (IOError, KeyError, _CacheFileVersion, PoseHashException):
                # if load_from_cache() raises an exception, we'll read from
                # the source pose file below because use_cache will still be
                # set to false, just ignore the exceptions here
                pass

        if not use_cache:
            # open the hdf5 pose file
            with h5py.File(self._path, 'r') as pose_h5:
                # extract data from the HDF5 file
                pose_grp = pose_h5['poseest']
                major_version = pose_grp.attrs['version'][0]

                # get pixel size
                self._cm_per_pixel = pose_grp.attrs.get('cm_per_pixel')

                # ensure the major version matches what we expect
                # TODO temporarily removed while v4 files under development
                #assert major_version == 4

                # load contents
                all_points = pose_grp['points'][:]
                all_confidence = pose_grp['confidence'][:]
                id_mask = pose_grp['id_mask'][:]
                instance_embed_id = pose_grp['instance_embed_id'][:]

            self._num_frames = len(all_points)
            self._num_identities = np.max(np.ma.array(instance_embed_id[...], mask=id_mask[...]))

            # generate list of identities based on the max number of instances in
            # the pose file
            self._identities = [*range(self._num_identities)]

            points_by_id_tmp = np.zeros_like(all_points)
            points_by_id_tmp[np.where(id_mask == 0)[0], instance_embed_id[id_mask == 0] - 1, :, :] = all_points[id_mask == 0, :, :]
            self._points = np.transpose(points_by_id_tmp, [1, 0, 2, 3])

            confidence_by_id_tmp = np.zeros_like(all_confidence)
            confidence_by_id_tmp[np.where(id_mask == 0)[0], instance_embed_id[id_mask == 0] - 1, :] = all_confidence[id_mask == 0, :]
            confidence_by_id = np.transpose(confidence_by_id_tmp, [1, 0, 2])

            self._point_mask = confidence_by_id > 0

            # build a mask for each identity that indicates if it exists or not
            # in the frame
            init_func = np.vectorize(
                lambda x, y: 0 if np.sum(self._point_mask[x][y][:-2]) == 0 else 1,
                otypes=[np.uint8])
            self._identity_mask = np.fromfunction(
                init_func, (self._num_identities, self._num_frames),
                dtype=np.int_)
            # cache pose data
            if cache_dir:
                self._cache_poses()

    @property
    def identity_to_track(self):
        return None

    @property
    def format_major_version(self):
        return 4

    def get_points(self, frame_index: int, identity: int,
                   scale: typing.Optional[float] = None):
        """
        get points and mask for an identity for a given frame
        :param frame_index: index of frame
        :param identity: identity that we want the points for
        :param scale: optional scale factor, set to cm_per_pixel to convert
        poses from pixel coordinates to cm coordinates
        :param fps: video frames per second
        :return: points, mask if identity has data for this frame
        """

        if not self._identity_mask[identity, frame_index]:
            return None, None

        if scale is not None:
            return (
                self._points[identity, frame_index, ...] * scale,
                self._point_mask[identity, frame_index, :]
            )
        else:
            return (
                self._points[identity, frame_index, ...],
                self._point_mask[identity, frame_index, :]
            )

    def get_identity_poses(self, identity: int,
                           scale: typing.Optional[float] = None):
        """
        return all points and point masks
        :param identity: included for compatibility with pose_est_v3. Should
        always be zero.
        :param scale: optional scale factor, set to cm_per_pixel to convert
        poses from pixel coordinates to cm coordinates
        :return: numpy array of points (#frames, 12, 2), numpy array of point
        masks (#frames, 12)
        """
        if scale is not None:
            return (
                self._points[identity, ...] * scale,
                self._point_mask[identity, ...]
            )
        else:
            return self._points[identity, ...], self._point_mask[identity, ...]

    def identity_mask(self, identity):
        return self._identity_mask[identity, :]

    def get_identity_point_mask(self, identity):
        """
        get the point mask array for a given identity
        :param identity: identity to return point mask for
        :return: array of point masks (#frames, 12)
        """
        return self._point_mask[identity, :]

    def _load_from_cache(self):
        """
        :return: None
        :raises: IOError, KeyError, _CacheFileVersion, PoseHashException
        """
        filename = self._path.name.replace('.h5', '_cache.h5')
        cache_file_path = self._cache_dir / filename

        with h5py.File(cache_file_path, 'r') as cache_h5:
            if cache_h5.attrs['version'] != self.__CACHE_FILE_VERSION:
                # cache file version is not what we expect, raise
                # exception so we will revert to reading source pose
                # file
                raise _CacheFileVersion

            if cache_h5.attrs['source_pose_hash'] != self._hash:
                raise PoseHashException

            pose_grp = cache_h5['poseest']
            self._points = pose_grp['points'][:]
            self._point_mask = pose_grp['point_mask'][:]
            self._identity_mask = pose_grp['identity_mask'][:]
            self._num_identities =self._identity_mask.shape[0]
            self._num_frames = self._points.shape[1]
            self._identities = [*range(self._num_identities)]

            # get pixel size
            self._cm_per_pixel = pose_grp.attrs.get('cm_per_pixel')

    def _cache_poses(self):
        """
        cache the pose data in an h5 file in the project cache directory
        :return: None
        """
        filename = self._path.name.replace('.h5', '_cache.h5')
        cache_file_path = self._cache_dir / filename

        with h5py.File(cache_file_path, 'w') as cache_h5:
            cache_h5.attrs['version'] = self.__CACHE_FILE_VERSION
            cache_h5.attrs['source_pose_hash'] = self.hash
            group = cache_h5.create_group('poseest')
            if self._cm_per_pixel is not None:
                group.attrs['cm_per_pixel'] = self._cm_per_pixel
            group.create_dataset('points', data=self._points)
            group.create_dataset('point_mask', data=self._point_mask)
            group.create_dataset('identity_mask', data=self._identity_mask)
