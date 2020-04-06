import enum
import heapq
import h5py
import numpy as np


class PoseEstimationV3:
    """
    class for opening and parsing version 3 of the pose estimation HDF5 file
    """

    class KeypointIndex(enum.IntEnum):
        """ enum defining the 12 keypoint indexes """
        NOSE = 0
        LEFT_EAR = 1
        RIGHT_EAR = 2
        BASE_NECK = 3
        LEFT_FRONT_PAW = 4
        RIGHT_FRONT_PAW = 5
        CENTER_SPINE = 6
        LEFT_REAR_PAW = 7
        RIGHT_REAR_PAW = 8
        BASE_TAIL = 9
        MID_TAIL = 10
        TIP_TAIL = 11

    def __init__(self, file_path):

        # Allow the path to the avi file to be passed rather than the path to
        # the .h5 file. Since we use a standardized naming convention we can
        # generate the path to the .h5 from the .avi path
        if file_path.endswith('.avi'):
            self._path = file_path.replace('.avi', '_pose_est_v3.h5')
        else:
            self._path = file_path

        # open the hdf5 pose file
        with h5py.File(self._path, 'r') as pose_h5:
            # extract data from the HDF5 file
            vid_grp = pose_h5['poseest']
            major_version = vid_grp.attrs['version'][0]

            # ensure the major version matches what we expect
            assert major_version == 3

            # load contents
            self._all_points = vid_grp['points'][:]
            self._all_confidence = vid_grp['confidence'][:]
            self._all_instance_count = vid_grp['instance_count'][:]
            self._all_track_id = vid_grp['instance_track_id'][:]

            self._max_instances = len(self._all_points[0])

            # build instance tracks from the HDF5 matrixes and
            self._track_dict = {}
            self._build_track_dict()

            # map track instances to identities
            self._identities = [*range(self._max_instances)]

            # maps track instances to identities
            self._identity_map = {}
            # for every frame, this array maps track instances to an identity
            self._identity_to_instance = np.full(
                (len(self._all_instance_count), self._max_instances), -1,
                dtype=np.int16)

            # populate identity_ma and identity_to_instance
            self._build_identity_map()

    @property
    def identities(self):
        """ return list of identities generated from file """
        return self._identities

    def get_points(self, frame_index, identity):
        """
        get points and mask for an identity for a given frame
        :param frame_index: index of frame
        :param identity: identity that we want the points for
        :return: points, mask if identity has data for this frame
        """
        instance = self._identity_to_instance[frame_index][identity]

        # return None for points and mask if identity doesn't exist for frame
        if instance == -1:
            return None, None

        # identity exists for this frame, get the track that corresponds to
        # this identity at this frame
        track = self._track_dict[instance]

        # find this frame's index in the track and return the points and mask
        track_index = frame_index - track['start_frame']
        return track['points'][track_index], track['point_masks'][track_index]

    def _build_track_dict(self):
        """ iterate through frames and build track dict """
        all_points_mask = self._all_confidence > 0

        frame_count = len(self._all_instance_count)
        for frame_index in range(frame_count):
            curr_instance_count = self._all_instance_count[frame_index]
            curr_track_ids = self._all_track_id[frame_index, :curr_instance_count]
            for i, curr_track_id in enumerate(curr_track_ids):
                curr_track_points = self._all_points[frame_index, i, ...]
                curr_track_points_mask = all_points_mask[frame_index, i, :]

                if curr_track_id in self._track_dict:
                    # we've seen this track before, append points
                    self._track_dict[curr_track_id]['points'].append(
                        curr_track_points)
                    self._track_dict[curr_track_id]['point_masks'].append(
                        curr_track_points_mask)
                else:
                    # this is the first frame the track appears, create a
                    # new dict
                    self._track_dict[curr_track_id] = {
                        'track_id': curr_track_id,
                        'start_frame': frame_index,
                        'points': [curr_track_points],
                        'point_masks': [curr_track_points_mask],
                    }

        for track in self._track_dict.values():
            track['points'] = np.stack(track['points'])
            track['point_masks'] = np.stack(track['point_masks'])
            track_length = len(track['points'])
            track['length'] = track_length
            track['stop_frame_exclu'] = track_length + track['start_frame']

    def _build_identity_map(self):
        free_identities = []
        for i in self._identities:
            heapq.heappush(free_identities, i)

        last_tracks = []
        for frame_index in range(len(self._all_instance_count)):
            current_tracks = self._all_track_id[frame_index][
                             :self._all_instance_count[frame_index]]

            # add identities back to the pool for any tracks that terminated
            for track in last_tracks:
                if track not in current_tracks:
                    identity = self._identity_map[track]
                    heapq.heappush(free_identities, identity)

            # if this is the first time we see the track grab a new identity
            for i in range(len(current_tracks)):
                if current_tracks[i] not in self._identity_map:
                    identity = heapq.heappop(free_identities)
                    self._identity_map[current_tracks[i]] = identity
                    self._identity_to_instance[frame_index][identity] = current_tracks[i]
                else:
                    self._identity_to_instance[frame_index][self._identity_map[current_tracks[i]]] = current_tracks[i]

            last_tracks = current_tracks[:]

        self._identity_to_instance = np.array(self._identity_to_instance)
