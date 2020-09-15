import heapq
from pathlib import Path

import h5py
import numpy as np

from .pose_est import PoseEstimation


class PoseEstimationV3(PoseEstimation):
    """
    class for opening and parsing version 3 of the pose estimation HDF5 file
    """

    def __init__(self, file_path: Path):
        """
        :param file_path: Path object representing the location of the pose file
        """
        super().__init__()

        self._path = file_path

        # open the hdf5 pose file
        with h5py.File(self._path, 'r') as pose_h5:
            # extract data from the HDF5 file
            vid_grp = pose_h5['poseest']
            major_version = vid_grp.attrs['version'][0]

            # ensure the major version matches what we expect
            assert major_version == 3

            # load contents
            all_points = vid_grp['points'][:]
            all_confidence = vid_grp['confidence'][:]
            all_instance_count = vid_grp['instance_count'][:]
            all_track_id = vid_grp['instance_track_id'][:]

        self._num_frames = len(all_points)
        self._max_instances = len(all_points[0])

        # map track instances to identities
        self._identities = [*range(self._max_instances)]

        # maps track instances to identities
        self._identity_map = {}

        # populate identity_map and identity_to_instance
        self._build_identity_map(all_instance_count, all_track_id)

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

    def get_points(self, frame_index, identity):
        """
        get points and mask for an identity for a given frame
        :param frame_index: index of frame
        :param identity: identity that we want the points for
        :return: points, mask if identity has data for this frame
        """

        if not self._identity_mask[identity, frame_index]:
            return None, None

        return (
            self._points[identity, frame_index, ...],
            self._point_mask[identity, frame_index, :]
        )

    def get_identity_poses(self, identity):
        return self._points[identity, ...], self._point_mask[identity, ...]

    def identity_mask(self, identity):
        return self._identity_mask[identity,:]

    @property
    def identity_to_track(self):
        identity_to_track = np.full((self._max_instances, self._num_frames), -1,
                                    dtype=np.int32)
        for track in self._track_dict.values():
            identity = self._identity_map[track['track_id']]
            identity_to_track[identity, track['start_frame']:track['stop_frame_exclu']] = track['track_id']
        return identity_to_track

    @classmethod
    def instance_count_from_file(cls, path: Path) -> int:
        """
        peek into a pose_est file to get the number of instances in the file
        :param path: path to pose_est h5 file
        :return: integer count
        """
        # open the hdf5 pose file
        with h5py.File(path, 'r') as pose_h5:
            # extract data from the HDF5 file
            vid_grp = pose_h5['poseest']
            major_version = vid_grp.attrs['version'][0]

            # ensure the major version matches what we expect
            assert major_version == 3

            return len(vid_grp['points'][0])

    def _build_track_dict(self, all_points, all_confidence, all_instance_count, all_track_id):
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
        for i in self._identities:
            heapq.heappush(free_identities, i)

        last_tracks = []
        for frame_index in range(self._num_frames):
            current_tracks = all_track_id[frame_index][
                                 :all_instance_count[frame_index]]

            # add identities back to the pool for any tracks that terminated
            for track in last_tracks:
                if track not in current_tracks:
                    heapq.heappush(free_identities, self._identity_map[track])

            # if this is the first time we see the track grab a new identity
            for i in range(len(current_tracks)):
                if current_tracks[i] not in self._identity_map:
                    identity = heapq.heappop(free_identities)
                    self._identity_map[current_tracks[i]] = identity

            last_tracks = current_tracks[:]
