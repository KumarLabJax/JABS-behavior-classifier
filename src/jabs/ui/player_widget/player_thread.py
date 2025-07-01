import time

import numpy as np
from PySide6 import QtCore, QtGui

from jabs.feature_extraction.social_features.social_distance import ClosestIdentityInfo
from jabs.pose_estimation import PoseEstimation
from jabs.video_reader import (
    VideoReader,
    draw_track,
    mark_identity,
    overlay_landmarks,
    overlay_segmentation,
)


class PlayerThread(QtCore.QThread):
    """Thread for grabbing frames from a video stream and converting them to QImage.

    Handles timing to achieve correct playback speed and emits signals to update UI components.

    Args:
        video_reader (VideoReader): The video reader instance.
        pose_est (PoseEstimation): The pose estimation object.
        identity (int): The active identity to track.
        show_track (bool, optional): Whether to show the track overlay. Defaults to False.
        identities (list[str], optional): List of all identities. Defaults to None.
        overlay_landmarks_flag (bool, optional): Whether to overlay landmarks. Defaults to False.
        overlay_segmentation_flag (bool, optional): Whether to overlay segmentation. Defaults to False.
        label_closest (bool, optional): Whether to label the closest animal. Defaults to False.

    Signals:
        newImage (QImage): Emitted with a new QImage for the PlayerWidget to display.
        updatePosition (int): Emitted with the current frame index
        endOfFile: Emitted when the end of the video is reached.
    """

    _CLOSEST_LABEL_COLOR = (255, 0, 0)
    _CLOSEST_FOV_LABEL_COLOR = (0, 255, 0)

    # signals used to update the UI components from the thread
    newImage = QtCore.Signal(QtGui.QImage)
    updatePosition = QtCore.Signal(int)
    endOfFile = QtCore.Signal()

    # signals used to update the properties of PlayerThread in a thread-safe manner
    setLabelClosest = QtCore.Signal(bool)
    setShowTrack = QtCore.Signal(bool)
    setOverlaySegmentation = QtCore.Signal(bool)
    setOverlayLandmarks = QtCore.Signal(bool)
    setActiveIdentity = QtCore.Signal(int)
    setPlaybackSpeed = QtCore.Signal(float)

    def __init__(
        self,
        video_reader: VideoReader,
        pose_est: PoseEstimation,
        identity: int,
        show_track: bool = False,
        identities: list[str] | None = None,
        overlay_landmarks_flag: bool = False,
        overlay_segmentation_flag: bool = False,
        label_closest: bool = False,
        playback_speed: float = 1.0,
    ):
        super().__init__()

        self._video_reader = video_reader
        self._pose_est = pose_est
        self._identity = identity
        self._label_closest = False
        self._show_track = show_track
        self._overlay_landmarks = overlay_landmarks_flag
        self._overlay_segmentation = overlay_segmentation_flag
        self._label_closest = label_closest
        self._identities = identities if identities is not None else []
        self._playback_speed = playback_speed

        self.setLabelClosest.connect(self._set_label_closest)
        self.setShowTrack.connect(self._set_show_track)
        self.setOverlaySegmentation.connect(self._set_overlay_segmentation)
        self.setOverlayLandmarks.connect(self._set_overlay_landmarks)
        self.setActiveIdentity.connect(self._set_identity)
        self.setPlaybackSpeed.connect(self._set_playback_speed)

    def stop_playback(self):
        """tell run thread to stop playback"""
        self.requestInterruption()

    @QtCore.Slot(int)
    def _set_identity(self, identity: int):
        """set the active identity"""
        self._identity = identity

    @QtCore.Slot(bool)
    def _set_label_closest(self, value: bool):
        self._label_closest = value

    @QtCore.Slot(bool)
    def _set_show_track(self, value: bool):
        self._show_track = value

    @QtCore.Slot(bool)
    def _set_overlay_segmentation(self, new_val: bool):
        """set the overlay segmentation property"""
        self._overlay_segmentation = new_val

    @QtCore.Slot(bool)
    def _set_overlay_landmarks(self, new_val: bool):
        """set the overlay landmarks property"""
        self._overlay_landmarks = new_val

    @QtCore.Slot(float)
    def _set_playback_speed(self, playback_speed: float):
        self._playback_speed = playback_speed

    def _read_and_emit_frame(self):
        frame = self._video_reader.load_next_frame()
        image = self._prepare_image(frame)
        self.updatePosition.emit(frame["index"])
        self.newImage.emit(image)

    def _prepare_image(self, frame: dict) -> QtGui.QImage | None:
        if frame["data"] is None:
            return None

        if self._identity is not None:
            if self._show_track:
                draw_track(frame["data"], self._pose_est, self._identity, frame["index"])

            if self._overlay_segmentation:
                overlay_segmentation(
                    frame["data"],
                    self._pose_est,
                    identity=self._identity,
                    frame_index=frame["index"],
                )

            if self._label_closest:
                closest_fov_id = self._get_closest_animal_id(
                    frame["index"], ClosestIdentityInfo.HALF_FOV_DEGREE
                )
                if closest_fov_id is not None:
                    mark_identity(
                        frame["data"],
                        self._pose_est,
                        closest_fov_id,
                        frame["index"],
                        color=self._CLOSEST_FOV_LABEL_COLOR,
                    )

                closest_id = self._get_closest_animal_id(frame["index"])
                if closest_id is not None and closest_id != closest_fov_id:
                    mark_identity(
                        frame["data"],
                        self._pose_est,
                        closest_id,
                        frame["index"],
                        color=self._CLOSEST_LABEL_COLOR,
                    )

        if self._overlay_landmarks:
            overlay_landmarks(frame["data"], self._pose_est)

        # using numpy slicing to convert from OpenCV BGR to Qt RGB format is more efficient
        # than using QImage.rgbSwapped() because QImage.rgbSwapped() creates a a QImage in BGR
        # first and then makes a copy with the channels swapped.
        img = frame["data"]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.shape[2] == 3:  # Assume BGR from OpenCV
            img_rgb = np.ascontiguousarray(img[..., ::-1])  # BGR to RGB
            height, width, channels = img_rgb.shape
            bytes_per_line = channels * width
            image = QtGui.QImage(
                img_rgb.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format.Format_RGB888,
            )
            return image
        else:
            raise ValueError("Unsupported image format: expected 3 channels (BGR/RGB)")

    def seek(self, position: int):
        """Seek to a specific frame position if the thread is not running.

        Updates the video reader to the given frame position and emits the corresponding frame
        and position to the UI.

        Args:
            position (int): The frame index to seek to.
        """
        if not self.isRunning():
            self._video_reader.seek(position)
            self._read_and_emit_frame()

    def run(self):
        """method to be run as a thread during playback

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
                    self.newImage.emit(image)
                    self.updatePosition.emit(frame["index"])

                # update timestamp for when should the next frame be shown
                next_timestamp += frame["duration"] / self._playback_speed

            else:
                # if the video stream reached the end of file let the UI know
                self.endOfFile.emit()
                # and terminate the loop
                end_of_file = True

    def _get_closest_animal_id(self, frame_index, half_fov_deg=None):
        idx = PoseEstimation.KeypointIndex
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
                                other_centroid = np.array(
                                    (other_shape.centroid.x, other_shape.centroid.y)
                                )

                                view_angle = ClosestIdentityInfo.compute_angle(
                                    ref_nose_point, ref_base_neck_point, other_centroid
                                )

                                # for FoV we want the range of view angle to be [180, -180)
                                if view_angle > 180:
                                    view_angle -= 360

                                if abs(view_angle) <= half_fov_deg and (
                                    closest_dist is None or curr_dist < closest_dist
                                ):
                                    # other animal is in FoV
                                    closest_id = curr_id
                                    closest_dist = curr_dist

        return closest_id
