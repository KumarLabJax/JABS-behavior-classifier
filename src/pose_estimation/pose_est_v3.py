import heapq
import typing
from pathlib import Path

import h5py
import numpy as np

from .pose_est import PoseEstimation, PoseHashException


class _CacheFileVersion(Exception):
    pass


class PoseEstimationV3(PoseEstimation):
    """
    class for opening and parsing version 3 of the pose estimation HDF5 file
    """

    __CACHE_FILE_VERSION = 2

    def __init__(self, file_path: Path, cache_dir: typing.Optional[Path]=None):
        """
        :param file_path: Path object representing the location of the pose file
        """
        super().__init__(file_path, cache_dir)

        self._identity_to_track = None
        self._identity_map = None

        # reading the v3 pose files is somewhat expensive due to the
        # creation of the identities and reordering data by assigned identity
        # to speedup reopening the pose file later, we'll cache the transformed
        # pose file in the project dir
        if cache_dir is not None:
            filename = self._path.name.replace('.h5', '_cache.h5')
            cache_file_path = self._cache_dir / filename
            use_cache = True

            try:
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
                    self._identity_to_track = pose_grp['identity_to_track'][:]
                    self._max_instances = self._points.shape[0]
                    self._num_frames = self._points.shape[1]
                    self._identities = [*range(self._max_instances)]

                    # get pixel size
                    self._cm_per_pixel = pose_grp.attrs.get('cm_per_pixel')

            except (IOError, KeyError, _CacheFileVersion, PoseHashException):
                # unable to open or read pose cache file, revert to source pose
                # file
                use_cache = False
        else:
            use_cache = False
            cache_file_path = None

        if not use_cache:
            # open the hdf5 pose file
            with h5py.File(self._path, 'r') as pose_h5:
                # extract data from the HDF5 file
                pose_grp = pose_h5['poseest']
                major_version = pose_grp.attrs['version'][0]

                # ensure the major version matches what we expect
                assert major_version == 3

                # load contents
                all_points = pose_grp['points'][:]
                all_confidence = pose_grp['confidence'][:]
                all_instance_count = pose_grp['instance_count'][:]
                all_track_id = pose_grp['instance_track_id'][:]

                # get pixel size
                self._cm_per_pixel = pose_grp.attrs.get('cm_per_pixel')

            self._num_frames = len(all_points)
            self._max_instances = len(all_points[0])

            # generate list of identities based on the max number of instances
            # in the pose file
            self._identities = [*range(self._max_instances)]

            # maps track instances to identities
            # populate identity_map and identity_to_instance
            self._identity_map = self._build_identity_map(
                all_instance_count, all_track_id)

            self._points = np.zeros(
                (self._max_instances, self.num_frames, len(self.KeypointIndex), 2),
                dtype=np.uint16)
            self._point_mask = np.zeros(self._points.shape[:-1], dtype=np.uint16)

            # build numpy arrays of points and point masks organized by identity
            self._track_dict = self._build_track_dict(
                    all_points, all_confidence, all_instance_count, all_track_id)

            for track_id, track in self._track_dict.items():
                self._points[
                    self._identity_map[track_id],
                    track['start_frame']:track['stop_frame_exclu'],
                    :] = track['points']
                self._point_mask[
                    self._identity_map[track_id],
                    track['start_frame']:track['stop_frame_exclu'],
                    :] = track['point_masks']

            # build a mask for each identity that indicates if it exists or not
            # in the frame
            init_func = np.vectorize(
                    lambda x, y: 0 if np.sum(self._point_mask[x][y][:-2]) == 0 else 1,
                    otypes=[np.uint8])
            self._identity_mask = np.fromfunction(
                init_func, (self._max_instances, self._num_frames), dtype=np.int_)

            if self._cache_dir is not None:
                with h5py.File(cache_file_path, 'w') as cache_h5:
                    cache_h5.attrs['version'] = self.__CACHE_FILE_VERSION
                    cache_h5.attrs['source_pose_hash'] = self.hash
                    group = cache_h5.create_group('poseest')
                    if self._cm_per_pixel is not None:
                        group.attrs['cm_per_pixel'] = self._cm_per_pixel
                    group.create_dataset('points', data=self._points)
                    group.create_dataset('point_mask', data=self._point_mask)
                    group.create_dataset('identity_mask', data=self._identity_mask)
                    group.create_dataset('identity_to_track',
                                         data=self.identity_to_track)

    @property
    def identity_to_track(self):
        if self._identity_to_track is None:
            self._identity_to_track = np.full(
                (self._max_instances, self._num_frames), -1, dtype=np.int32)
            for track in self._track_dict.values():
                identity = self._identity_map[track['track_id']]
                self._identity_to_track[identity, track['start_frame']:track['stop_frame_exclu']] = track['track_id']
        return self._identity_to_track

    @property
    def format_major_version(self):
        return 3

    def get_points(self, frame_index: int, identity: int,
                   scale: typing.Optional[float] = None):
        """
        get points and mask for an identity for a given frame
        :param frame_index: index of frame
        :param identity: identity that we want the points for
        :param scale: optional scale factor, set to cm_per_pixel to convert
        poses from pixel coordinates to cm coordinates
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

    def _build_track_dict(self, all_points, all_confidence, all_instance_count,
                          all_track_id):
        """ iterate through frames and build track dict """
        all_points_mask = all_confidence > 0
        track_dict = {}

        for frame_index in range(self.num_frames):
            curr_instance_count = all_instance_count[frame_index]
            curr_track_ids = all_track_id[frame_index, :curr_instance_count]
            for i, curr_track_id in enumerate(curr_track_ids):
                curr_track_points = all_points[frame_index, i, ...]
                curr_track_points_mask = all_points_mask[frame_index, i, :]

                if curr_track_id in track_dict:
                    # we've seen this track before, append points
                    track_dict[curr_track_id]['points'].append(
                        curr_track_points)
                    track_dict[curr_track_id]['point_masks'].append(
                        curr_track_points_mask)
                else:
                    # this is the first frame the track appears, create a
                    # new dict
                    track_dict[curr_track_id] = {
                        'track_id': curr_track_id,
                        'start_frame': frame_index,
                        'points': [curr_track_points],
                        'point_masks': [curr_track_points_mask],
                    }

        for track in track_dict.values():
            track['points'] = np.stack(track['points'])
            track['point_masks'] = np.stack(track['point_masks'])
            track_length = len(track['points'])
            track['length'] = track_length
            track['stop_frame_exclu'] = track_length + track['start_frame']

        return track_dict

    def _build_identity_map(self, all_instance_count, all_track_id):
        """ map individual tracks to identities """
        free_identities = []
        identity_track_count = {}
        identity_map = {}

        for i in self._identities:
            heapq.heappush(free_identities, i)
            identity_track_count[i] = 0

        last_tracks = []
        for frame_index in range(self._num_frames):
            current_tracks = all_track_id[frame_index][
                                 :all_instance_count[frame_index]]

            # add identities back to the pool for any tracks that terminated
            for track in last_tracks:
                if track not in current_tracks:
                    heapq.heappush(free_identities, identity_map[track])

            # if this is the first time we see the track grab a new identity
            for i in range(len(current_tracks)):
                if current_tracks[i] not in identity_map:
                    identity = heapq.heappop(free_identities)
                    identity_map[current_tracks[i]] = identity
                    identity_track_count[identity] += 1

            last_tracks = current_tracks[:]

        # prune the identities if some end up not being used
        identities = []
        for ident, count in identity_track_count.items():
            if count != 0:
                identities.append(ident)

        # the only identities that get pruned should be at the end of the
        # identities list
        assert(identities == self._identities[:len(identities)])

        self._identities = identities
        self._max_instances = len(identities)
        return identity_map
