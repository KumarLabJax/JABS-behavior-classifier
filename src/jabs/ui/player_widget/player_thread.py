import time

import numpy as np
from PySide6 import QtCore, QtGui

from jabs.feature_extraction.social_features.social_distance import ClosestIdentityInfo
from jabs.pose_estimation import PoseEstimation, PoseEstimationV3
from jabs.video_reader import VideoReader, label_all_identities, label_identity, overlay_landmarks, \
    overlay_segmentation, overlay_pose, draw_track


class PlayerThread(QtCore.QThread):
    """
    thread used to grab frames (numpy arrays) from a video stream and convert
    them to a QImage for display by the frame widget

    handles timing to get correct playback speed
    """

    _CLOSEST_LABEL_COLOR = (255, 0, 0)
    _CLOSEST_FOV_LABEL_COLOR = (0, 255, 0)

    # signals used to update the UI components from the thread
    newImage = QtCore.Signal(dict)
    updatePosition = QtCore.Signal(dict)
    endOfFile = QtCore.Signal()

    def __init__(self, video_reader, pose_est, identity, show_track=False,
                 overlay_pose_flag=False, identities=None, overlay_landmarks_flag=False,
                 overlay_segmentation_flag=False):
        super().__init__()
        self._video_reader = video_reader
        self._pose_est = pose_est
        self._identity = identity
        self._label_closest = False
        self._show_track = show_track
        self._overlay_pose = overlay_pose_flag
        self._overlay_segmentation = overlay_segmentation_flag
        self._overlay_landmarks = overlay_landmarks_flag
        self._identities = identities if identities is not None else []
        self._lock = QtCore.QMutex()

    def stop_playback(self):
        """
        tell run thread to stop playback
        """
        self.requestInterruption()

    def set_identity(self, identity):
        """
        set the active identity
        :param identity: new selected identity
        :return: None
        """
        self._identity = identity

    def set_identities(self, identities):
        self._identities = identities

    def label_closest(self, new_val: bool):
        self._label_closest = new_val

    def set_show_track(self, new_val: bool):
        self._show_track = new_val

    def set_overlay_pose(self, new_val: bool):
        self._overlay_pose = new_val

    def set_overlay_segmentation(self, new_val: bool):
        self._overlay_segmentation = new_val

    def set_overlay_landmarks(self, new_val: bool):
        self._overlay_landmarks = new_val

    def _read_and_emit_frame(self):
        frame = self._video_reader.load_next_frame()
        image = self._prepare_image(frame)
        self.newImage.emit({'image': image, 'source': self._video_reader.filename})
        self.updatePosition.emit({'index': frame['index'], 'source': self._video_reader.filename})

    def _prepare_image(self, frame: dict) -> QtGui.QImage | None:
        if frame['data'] is None:
            return None

        if self._identity is not None:

            if self._show_track:
                draw_track(frame['data'], self._pose_est,
                           self._identity, frame['index'])

            if self._overlay_pose:
                overlay_pose(
                    frame['data'],
                    *self._pose_est.get_points(frame['index'], self._identity)
                )
            if self._overlay_segmentation:
                overlay_segmentation(
                    frame['data'],
                    self._pose_est,
                    identity=self._identity,
                    frameIndex=frame['index'],
                    identities=self._identities
                )
            if self._overlay_landmarks:
                overlay_landmarks(frame['data'], self._pose_est)

            if self._label_closest:
                closest_fov_id = self._get_closest_animal_id(frame['index'], ClosestIdentityInfo.HALF_FOV_DEGREE)
                if closest_fov_id is not None:
                    label_identity(frame['data'], self._pose_est,
                                   closest_fov_id, frame['index'],
                                   color=self._CLOSEST_FOV_LABEL_COLOR)

                closest_id = self._get_closest_animal_id(frame['index'])
                if closest_id is not None and closest_id != closest_fov_id:
                    label_identity(frame['data'], self._pose_est,
                                   closest_id, frame['index'],
                                   color=self._CLOSEST_LABEL_COLOR)

        # label all identities
        label_all_identities(frame['data'],
                             self._pose_est, self._identities,
                             frame['index'], subject=self._identity)

        # convert OpenCV image (numpy array) to QImage
        image = QtGui.QImage(frame['data'], frame['data'].shape[1],
                             frame['data'].shape[0],
                             QtGui.QImage.Format_RGB888).rgbSwapped()

        return image

    def seek(self, position: int):
        if not self.isRunning():
            self._video_reader.seek(position)
            self._read_and_emit_frame()

    def load_new_video(self, video_reader: VideoReader, pose_est: PoseEstimation, identity: int, identities: list):
        if not self.isRunning():
            self._video_reader = video_reader
            self._pose_est = pose_est
            self._identity = identity
            self._identities = identities

    def run(self):
        """
        method to be run as a thread during playback
        handles grabbing the next frame from the buffer, converting to a QImage,
        and sending to the UI component for display.
        """

        end_of_file = False
        next_timestamp = 0
        start_time = 0

        # iterate until we've been told to stop (user clicks pause button)
        # or we reach end of file
        while not self.isInterruptionRequested() and not end_of_file:
            now = time.perf_counter()
            frame = self._video_reader.load_next_frame()
            image = self._prepare_image(frame)

            if image:
                # don't update frame until we've shown the last one for the
                # required duration
                if start_time > 0:
                    # sleep difference between next_timestamp and amount of
                    # actual clock time since we started playback
                    time.sleep(max(0, next_timestamp - (now - start_time)))
                else:
                    # first frame, save the start time
                    start_time = now

                # send the new frame and the frame index to the UI components
                # unless playback was stopped while we were sleeping
                if not self.isInterruptionRequested():
                    self.newImage.emit({'image': image, 'source': self._video_reader.filename})
                    self.updatePosition.emit({'index': frame['index'], 'source': self._video_reader.filename})

                # update timestamp for when should the next frame be shown
                next_timestamp += frame['duration']

            else:
                # if the video stream reached the end of file let the UI know
                self.endOfFile.emit()
                # and terminate the loop
                end_of_file = True

    def _get_closest_animal_id(self, frame_index, half_fov_deg=None):

        idx = PoseEstimationV3.KeypointIndex
        closest_id = None
        closest_dist = None
        ref_shape = self._pose_est.get_identity_convex_hulls(self._identity)[frame_index]
        if ref_shape is not None:
            for curr_id in self._pose_est.identities:
                if curr_id != self._identity:
                    other_shape = self._pose_est.get_identity_convex_hulls(curr_id)[frame_index]

                    if other_shape is not None:

                        curr_dist = ref_shape.distance(other_shape)
                        if half_fov_deg is None or half_fov_deg >= 180:

                            # we can ignore FoV angle and just worry about distance
                            if closest_dist is None or curr_dist < closest_dist:
                                closest_id = curr_id
                                closest_dist = curr_dist
                        else:
                            # we need to account for FoV angle
                            points, mask = self._pose_est.get_points(frame_index, self._identity)

                            # we need nose and base neck to figure out view angle
                            if mask[idx.NOSE] == 1 and mask[idx.BASE_NECK] == 1:
                                ref_base_neck_point = points[idx.BASE_NECK, :]
                                ref_nose_point = points[idx.NOSE, :]
                                other_centroid = np.array((other_shape.centroid.x, other_shape.centroid.y))

                                view_angle = ClosestIdentityInfo.compute_angle(
                                    ref_nose_point,
                                    ref_base_neck_point,
                                    other_centroid)

                                # for FoV we want the range of view angle to be [180, -180)
                                if view_angle > 180:
                                    view_angle -= 360

                                if abs(view_angle) <= half_fov_deg:
                                    # other animal is in FoV
                                    if closest_dist is None or curr_dist < closest_dist:
                                        closest_id = curr_id
                                        closest_dist = curr_dist

        return closest_id

