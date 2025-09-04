import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from jabs.pose_estimation import PoseEstimation
from jabs.project import VideoLabels
from jabs.video_reader import VideoReader

from .frame_with_overlays import FrameWithOverlaysWidget
from .player_thread import PlayerThread

_SPEED_VALUES = [0.5, 1, 2, 4]


@dataclass(frozen=True)
class PlaybackRange:
    """Dataclass to represent a playback range in the video.

    Attributes:
        start_frame (int): The starting frame number of the playback range.
        end_frame (int): The ending frame number of the playback range. (inclusive)
    """

    start_frame: int
    end_frame: int


class PlayerWidget(QtWidgets.QWidget):
    """Video Player Widget.

    Consists of a QLabel to display a frame image, and
    basic player controls below the frame (play/pause button, position slider,
    previous/next frame buttons).

    Signals:
        update_frame_number (int): Emitted when the current frame number changes.
        update_identities (list): Emitted when the list of identities is updated.
        playback_finished (): Emitted when playback of a range finishes.
        eof_reached (): Emitted when the end of the video file is reached.

    The position slider can be dragged while video is paused or is playing and the
    position will be updated. If video was playing when the slider is dragged
    the playback will resume after the slider is released.
    """

    class LabelOverlayMode(enum.IntEnum):
        """Enum for label overlay options."""

        NONE = 0
        LABEL = 1
        PREDICTION = 2

    update_frame_number = QtCore.Signal(int)
    update_identities = QtCore.Signal(list)
    playback_finished = QtCore.Signal()
    eof_reached = QtCore.Signal()
    id_label_clicked = QtCore.Signal(int)

    PoseOverlayMode = FrameWithOverlaysWidget.PoseOverlayMode
    IdentityOverlayMode = FrameWithOverlaysWidget.IdentityOverlayMode

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # make sure the player thread is stopped when quitting the application
        QtCore.QCoreApplication.instance().aboutToQuit.connect(self._cleanup_player_thread)  # type: ignore

        # keep track of the current state
        self._playing = False
        self._resume_playing = False
        self._video_stream = None
        self._pose_est = None
        self._playback_range: PlaybackRange | None = None

        # properties to control video overlays managed by PlayerThread
        self._label_closest = False
        self._show_track = False
        self._overlay_segmentation = False
        self._overlay_landmarks = False
        self._identities = []

        # currently selected identity
        self._active_identity = 0

        # player thread to read and prepare frames for display
        self._player_thread = None

        #  - setup Widget UI components

        # custom widget for displaying a resizable image
        self._frame_widget = FrameWithOverlaysWidget()
        self._frame_widget.playback_speed_changed.connect(self._on_playback_speed_changed)
        self._frame_widget.id_label_clicked.connect(self.id_label_clicked)

        #  -- player controls

        # current time and frame display
        font = QtGui.QFont("Courier New", 12)
        self._time_label = QtWidgets.QLabel("00:00:00")
        self._frame_label = QtWidgets.QLabel("0")
        self._frame_label.setFont(font)
        self._time_label.setFont(font)
        self._time_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._frame_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        time_layout = QtWidgets.QHBoxLayout()
        time_layout.addWidget(self._time_label)
        time_layout.addStretch()
        time_layout.addWidget(self._frame_label)

        # play/pause button
        self._play_button = QtWidgets.QPushButton()
        self._play_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._play_button.setCheckable(True)
        self._play_button.setEnabled(False)
        self._play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self._play_button.clicked.connect(self.toggle_play)

        # previous frame button
        self._previous_frame_button = QtWidgets.QToolButton()
        self._previous_frame_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._previous_frame_button.setText("◀")
        self._previous_frame_button.setMaximumWidth(20)
        self._previous_frame_button.setMaximumHeight(20)
        self._previous_frame_button.setAutoRepeat(True)

        # next frame button
        self._next_frame_button = QtWidgets.QToolButton()
        self._next_frame_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._next_frame_button.setText("▶")
        self._next_frame_button.setMaximumWidth(20)
        self._next_frame_button.setMaximumHeight(20)
        self._next_frame_button.setAutoRepeat(True)

        # Connect the previous/next frame clicked signals to their respective functions.
        # We need to wrap the functions because they take an optional parameter and Qt always
        # passes a boolean "checked" argument to the click signal handler which is interpreted as
        # the optional "frames" parameter for these functions.
        self._previous_frame_button.clicked.connect(lambda _: self.previous_frame())
        self._next_frame_button.clicked.connect(lambda _: self.next_frame())

        # previous/next button layout
        frame_button_layout = QtWidgets.QHBoxLayout()
        frame_button_layout.setSpacing(2)
        frame_button_layout.addWidget(self._previous_frame_button)
        frame_button_layout.addWidget(self._next_frame_button)
        # prev/next frame buttons are disabled until a video is loaded
        self._disable_frame_buttons()

        # position slider
        self._position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._position_slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._position_slider.sliderMoved.connect(
            self._position_slider_moved, QtCore.Qt.ConnectionType.QueuedConnection
        )
        self._position_slider.sliderPressed.connect(self._position_slider_clicked)
        self._position_slider.sliderReleased.connect(self._position_slider_release)
        self._position_slider.valueChanged.connect(self._on_slider_value_changed)
        self._position_slider.setEnabled(False)

        # -- set up the layout of the components

        # main player control layout
        player_control_layout = QtWidgets.QHBoxLayout()
        player_control_layout.setContentsMargins(2, 0, 2, 0)
        player_control_layout.setSpacing(2)
        player_control_layout.addWidget(self._play_button)
        player_control_layout.addWidget(self._position_slider)
        player_control_layout.addLayout(frame_button_layout)

        # widget layout
        player_layout = QtWidgets.QVBoxLayout()
        player_layout.addWidget(self._frame_widget)
        player_layout.addLayout(time_layout)
        player_layout.addLayout(player_control_layout)

        self.setLayout(player_layout)

    @property
    def pose_overlay_mode(self) -> PoseOverlayMode:
        """return the current pose overlay mode from the frame widget"""
        return self._frame_widget.pose_overlay_mode

    @pose_overlay_mode.setter
    def pose_overlay_mode(self, mode: PoseOverlayMode) -> None:
        """set the pose overlay mode in the frame widget"""
        self._frame_widget.pose_overlay_mode = mode

    @property
    def id_overlay_mode(self) -> IdentityOverlayMode:
        """return the current identity overlay mode from the frame widget"""
        return self._frame_widget.identity_overlay_mode

    @id_overlay_mode.setter
    def id_overlay_mode(self, mode: IdentityOverlayMode) -> None:
        """set the identity overlay mode in the frame widget"""
        self._frame_widget.identity_overlay_mode = mode

    @property
    def overlay_annotations_enabled(self) -> bool:
        """return the current overlay annotations state from the frame widget"""
        return self._frame_widget.overlay_annotations_enabled

    @overlay_annotations_enabled.setter
    def overlay_annotations_enabled(self, enabled: bool) -> None:
        """set the overlay annotations in the frame widget"""
        self._frame_widget.overlay_annotations_enabled = enabled

    def _cleanup_player_thread(self) -> None:
        """cleanup function to stop the player thread if it is running"""
        if self._player_thread is not None:
            self._player_thread.stop_playback()
            self._player_thread.wait()
            try:
                self._player_thread.newImage.disconnect()
                self._player_thread.updatePosition.disconnect()
                self._player_thread.endOfFile.disconnect()
            except TypeError:
                # Already disconnected
                pass
            self._player_thread.deleteLater()
            self._player_thread = None
            # Process pending events to flush any queued signals
            QtWidgets.QApplication.processEvents()

    def reset(self) -> None:
        """reset video player before loading a new video"""
        self._video_stream = None
        self._identities = None
        self._pose_est = None
        self._active_identity = None
        self._position_slider.setValue(0)
        self._position_slider.setEnabled(False)
        self._play_button.setEnabled(False)
        self._disable_frame_buttons()
        self._frame_label.setText("")
        self._time_label.setText("")
        self._frame_widget.reset()

    def load_video(self, path: Path, pose_est: PoseEstimation, video_labels: VideoLabels) -> None:
        """load a new video source

        Args:
            path: path to video file
            pose_est: pose file for this video
            video_labels: video labels (behavior and interval annotations) for this video
        """
        # cleanup the old player thread if it exists
        self._cleanup_player_thread()

        # reset the button state to not playing
        self.stop()
        self.reset()

        self._video_stream = VideoReader(path)
        self._pose_est = pose_est
        self._identities = pose_est.identities
        self._frame_widget.set_pose(pose_est)
        self._frame_widget.annotations = video_labels.timeline_annotations

        self._player_thread = PlayerThread(
            self._video_stream,
            self._pose_est,
            self._active_identity,
            self._show_track,
            self._identities,
            self._overlay_landmarks,
            self._overlay_segmentation,
            playback_speed=self._frame_widget.playback_speed,
        )
        self._player_thread.newImage.connect(self._display_image)
        self._player_thread.updatePosition.connect(self._on_frame_number_changed)
        self._player_thread.endOfFile.connect(self._on_eof)

        # set up the position slider
        self._seek(0)
        self._position_slider.setMaximum(self._video_stream.num_frames - 1)
        self._position_slider.setEnabled(True)

        # enable the play button and next/previous frame buttons
        self._play_button.setEnabled(True)
        self._enable_frame_buttons()

    def stop(self) -> None:
        """stop playing and reset the play button to its initial state"""
        # if we have an active player thread, terminate
        if self._player_thread:
            self._player_thread.stop_playback()
            self._player_thread.wait()

        # change the icon to play
        self._play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )

        # switch the button state to off
        self._play_button.setChecked(False)

        self._enable_frame_buttons()
        self._playing = False
        self._playback_range = None

    def play_range(self, start: int, end: int) -> None:
        """play a range of frames in the video

        Args:
            start: starting frame number
            end: ending frame number
        """
        if self._video_stream is None or self._playing:
            return

        # ensure the range is valid
        if start < 0 or end >= self.num_frames or start >= end:
            return

        self.stop()
        self._playback_range = PlaybackRange(start_frame=start, end_frame=end)

        # set the position slider to the start frame
        self._seek(self._playback_range.start_frame)

        # start playback, pause for 250 ms first to let things settle after moving
        # the position to the start of the range.
        QtCore.QTimer.singleShot(250, self._start_player_thread)

    @property
    def current_frame(self) -> int:
        """return the current frame"""
        return self._position_slider.value()

    @property
    def num_frames(self) -> int:
        """get total number of frames in the loaded video"""
        if self._video_stream is None:
            return 0
        return self._video_stream.num_frames

    @property
    def stream_fps(self) -> int:
        """get frames per second from loaded video"""
        assert self._video_stream is not None
        return self._video_stream.fps

    def _set_overlay_attr(
        self, attr: str, signal: QtCore.Signal | None, enabled: bool | None
    ) -> None:
        """Toggle or set an overlay attribute and emit the corresponding signal.

        Args:
            attr: Name of the attribute to toggle or set.
            signal: Signal to emit with the new value.
            enabled: If provided, sets the attribute to this value; if None, toggles the current value.

        This method also forces a redraw of the current frame with updated overlay settings if playback is paused in
        order to force the current frame to be redrawn with the new overlay settings.
        """
        current = getattr(self, attr)
        new_value = not current if enabled is None else enabled
        setattr(self, attr, new_value)
        if signal:
            # noinspection PyUnresolvedReferences
            signal.emit(new_value)
            self.reload_frame()

    def show_closest(self, enabled: bool | None = None) -> None:
        """Toggle or set the 'show closest' overlay state."""
        self._set_overlay_attr(
            "_label_closest",
            self._player_thread.setLabelClosest if self._player_thread else None,
            enabled,
        )

    def show_track(self, enabled: bool | None = None) -> None:
        """Toggle or set the 'show track' overlay state."""
        self._set_overlay_attr(
            "_show_track",
            self._player_thread.setShowTrack if self._player_thread else None,
            enabled,
        )

    def overlay_segmentation(self, enabled: bool | None = None) -> None:
        """Toggle or set the 'overlay segmentation' overlay state."""
        self._set_overlay_attr(
            "_overlay_segmentation",
            self._player_thread.setOverlaySegmentation if self._player_thread else None,
            enabled,
        )

    def overlay_landmarks(self, enabled: bool | None = None) -> None:
        """Toggle or set the 'overlay segmentation' overlay state."""
        self._set_overlay_attr(
            "_overlay_landmarks",
            self._player_thread.setOverlayLandmarks if self._player_thread else None,
            enabled,
        )

    def reload_frame(self) -> None:
        """reload the current frame in the player thread.

        This is useful when video overlays have changed but the video
        is paused and we want to update the current frame display.
        """
        if not self._playing and self._player_thread is not None:
            self._player_thread.seek(self._position_slider.value())

    def _seek(self, position: int) -> None:
        self._player_thread.seek(position)
        self.update_frame_number.emit(position)
        self._update_time_display(position)

    def _position_slider_clicked(self) -> None:
        """Click event for position slider.

        Seek to the new position of the slider. If the video is playing we will temporarily stop playback and
        playback will resume when slider is released. New frame is displayed with the updated position.
        """
        # should we resume playing after the slider is released?
        self._resume_playing = self._playing
        if self._playing:
            self._player_thread.stop_playback()
        self._seek(self._position_slider.value())

    def _position_slider_moved(self) -> None:
        """position slider move event.

        seeks the video to the new frame and displays it
        """
        self._seek(self._position_slider.value())

    def _position_slider_release(self) -> None:
        """release event for the position slider.

        The new position gets updated for the final time. If we were playing
        when we clicked the slider then we resume playing after releasing.
        """
        self._seek(self._position_slider.value())
        if self._resume_playing:
            self._start_player_thread()
            self._resume_playing = False

    def _on_slider_value_changed(self, value):
        """handle slider value change event, this lets us seek by scrolling or dragging the slider"""
        if self._player_thread is not None and not self._playing:
            self._seek(value)

    def toggle_play(self) -> None:
        """handle clicking on the play/pause button

        don't do anything if a video hasn't been loaded, or if we are seeking
        (user clicks space bar while dagging slider)
        """
        if self._player_thread is None or self._position_slider.isSliderDown():
            return

        if self._playing:
            # if we are playing, stop
            self.stop()

        else:
            # we weren't already playing so start
            self._play_button.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
            )
            self._disable_frame_buttons()
            self._start_player_thread()

    def next_frame(self, frames: int = 1) -> None:
        """advance to the next frame and display it

        Args:
            frames: optional, number of frames to advance
        """
        self.seek_to_frame(self._position_slider.value() + frames)

    def previous_frame(self, frames: int = 1) -> None:
        """go back to the previous frame and display it

        Args:
            frames: optional number of frames to move back
        """
        self.seek_to_frame(self._position_slider.value() - frames)

    def seek_to_frame(self, frame_number: int):
        """seek to a specific frame number and display it

        Args:
            frame_number: frame number to seek to
        """
        # don't do anything if a video isn't loaded or if the video is playing
        if self._video_stream is None or self._playing:
            return

        # the frame number should be bounded by the number of frames in the video
        num_frames = self._video_stream.num_frames
        if num_frames == 0:
            return
        new_frame = max(0, min(frame_number, num_frames - 1))

        if new_frame != self._position_slider.value():
            self._position_slider.setValue(new_frame)
            self._player_thread.seek(new_frame)

    def set_active_identity(self, identity: int) -> None:
        """set an active identity, which will be labeled in the video"""
        # don't do anything if a video isn't loaded
        if self._player_thread is None:
            return

        self._active_identity = identity
        self._frame_widget.set_active_identity(identity)
        self._player_thread.setActiveIdentity.emit(identity)
        self.reload_frame()

    def set_labels(self, labels: list[np.ndarray] | None) -> None:
        """set labels used for overlay in the frame widget

        Args:
            labels: list of numpy arrays with behavior/not behavior/no label for each identity.
              Must match the sorted order of identities or be None.
        """
        self._frame_widget.set_label_overlay(labels)
        self.reload_frame()

    @property
    def pixmap_clicked(self) -> QtCore.SignalInstance:
        """return the pixmap_clicked signal from the frame widget"""
        return self._frame_widget.pixmap_clicked

    def _enable_frame_buttons(self) -> None:
        """enable the previous/next frame buttons"""
        self._next_frame_button.setEnabled(True)
        self._previous_frame_button.setEnabled(True)

    def _disable_frame_buttons(self) -> None:
        """disable the previous/next frame buttons"""
        self._next_frame_button.setEnabled(False)
        self._previous_frame_button.setEnabled(False)

    def _update_time_display(self, frame_number: int) -> None:
        """update the time and current frame labels with current frame

        Args:
            frame_number: current frame number
        """
        self._frame_label.setText(f"{frame_number}:{self._video_stream.num_frames - 1}")
        self._time_label.setText(self._video_stream.get_frame_time(frame_number))

    @QtCore.Slot(QtGui.QImage)
    def _display_image(self, image: QtGui.QImage) -> None:
        """display a new frame sent from the player thread

        Args:
            image (QImage): frame ready for display as emitted by player thread
        """
        self._frame_widget.update_frame(image, self.current_frame)

    @QtCore.Slot(int)
    def _on_frame_number_changed(self, frame_number: int) -> None:
        """Handle frame number changes from the player thread.

        Args:
            frame_number (int): The new frame number to display.
        """
        # update the position slider and time display
        self._set_position(frame_number)

        # If we have a playback range, check if we reached the end
        if self._playback_range and frame_number >= self._playback_range.end_frame:
            self.stop()
            self.playback_finished.emit()

    @QtCore.Slot()
    def _on_eof(self) -> None:
        """Handle end of file signal from the player thread."""
        self.stop()
        self.eof_reached.emit()

    def _set_position(self, frame_number: int) -> None:
        """update the position slider during playback

        Args:
            frame_number (int): frame_number emitted by player thread
        """
        # don't update the slider position if it is being dragged
        if self._position_slider.isSliderDown():
            return
        self._position_slider.setValue(frame_number)
        self._update_time_display(frame_number)
        self.update_frame_number.emit(frame_number)

    def _start_player_thread(self) -> None:
        """start video playback in player thread"""
        self._player_thread.start()
        self._playing = True

    def _on_playback_speed_changed(self, speed: float) -> None:
        """Handle playback speed changes."""
        if self._player_thread is not None:
            self._player_thread.setPlaybackSpeed.emit(speed)
