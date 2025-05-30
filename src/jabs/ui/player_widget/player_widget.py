from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QImage, QPaintEvent

from jabs.video_reader import VideoReader

from .player_thread import PlayerThread


class _FrameWidget(QtWidgets.QLabel):
    """widget that implements a resizable pixmap label

    Used for displaying the current frame of the video. If necessary, the pixmap size is scaled down to fit the
    available area.
    """

    pixmap_clicked = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self.firstFrame = True
        self._scaled_pix_x = 0
        self._scaled_pix_y = 0
        self._scaled_pix_width = 0
        self._scaled_pix_height = 0
        self.setMinimumSize(400, 400)

    def mousePressEvent(self, event):
        pix_x, pix_y = self._frame_xy_to_pixmap_xy(event.x(), event.y())
        self.pixmap_clicked.emit({"x": pix_x, "y": pix_y})

        QtWidgets.QLabel.mousePressEvent(self, event)

    def _frame_xy_to_pixmap_xy(self, x, y):
        """Convert the given x, y coordinates from FrameWidget coordinates to pixmap coordinates.

        Ie which pixel did the user click on? We account for image scaling and translation
        """
        pixmap = self.pixmap()
        if (
            pixmap is not None
            and (
                self._scaled_pix_height != pixmap.height()
                or self._scaled_pix_width != pixmap.width()
                or self._scaled_pix_x != 0
                or self._scaled_pix_y != 0
            )
            and self._scaled_pix_width >= 1
            and self._scaled_pix_height >= 1
        ):
            # we've done all the checks and it's safe to transform
            # the x, y point
            x -= self._scaled_pix_x
            y -= self._scaled_pix_y
            x *= pixmap.width() / self._scaled_pix_width
            y *= pixmap.height() / self._scaled_pix_height

        return x, y

    def sizeHint(self):
        """Override QLabel.sizeHint to give an initial starting size."""
        return QtCore.QSize(1024, 1024)

    def reset(self):
        """reset state of frame widget"""
        self.clear()

    def paintEvent(self, event: QPaintEvent):
        """override paintEvent handler to scale the image if the widget is resized.

        Don't enable resizing until after  the first frame has been drawn so
        that the widget will be expanded to fit the actual size of the frame.
        """
        # only draw if we have an image to show
        if self.pixmap() is not None and not self.pixmap().isNull():
            # current size of the widget
            size = self.size()

            painter = QtGui.QPainter(self)
            point = QtCore.QPoint(0, 0)

            # scale the image to the current size of the widget. First frame,
            # the widget will be expanded to fit the full size image.
            pix = self.pixmap().scaled(
                size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

            # because we are maintaining aspect ratio, the scaled frame might
            # not be the same dimensions as the area we are painting it.
            # adjust the start point to center the image in the widget
            point.setX((size.width() - pix.width()) // 2)
            point.setY((size.height() - pix.height()) // 2)

            # draw the pixmap starting at the new calculated offset
            painter.drawPixmap(point, pix)

            self._scaled_pix_x = point.x()
            self._scaled_pix_y = point.y()
            self._scaled_pix_width = pix.width()
            self._scaled_pix_height = pix.height()

        else:
            # if we don't have a pixmap to display just call the original QLabel
            # paintEvent
            super().paintEvent(event)


class PlayerWidget(QtWidgets.QWidget):
    """Video Player Widget.

    Consists of a QLabel to display a frame image, and
    basic player controls below the frame (play/pause button, position slider,
    previous/next frame buttons.

    position slider can be dragged while video is paused or is playing and the
    position will be updated. If video was playing when the slider is dragged
    the playback will resume after the slider is released.
    """

    # signal to allow other UI components to observe current frame number
    updateFrameNumber = QtCore.Signal(int)

    # let the main window UI know what the list of identities should be
    updateIdentities = QtCore.Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # make sure the player thread is stopped when quitting the application
        QtCore.QCoreApplication.instance().aboutToQuit.connect(self._cleanup_player_thread)

        # keep track of the current state
        self._playing = False
        self._seeking = False

        # VideoStream object, will be initialized when video is loaded
        self._video_stream = None

        self._pose_est = None

        self._label_closest = False
        self._show_track = False
        self._overlay_pose = False
        self._overlay_segmentation = False
        self._overlay_landmarks = False
        self._identities = []

        # currently selected identity -- if set will be labeled in the video
        self._active_identity = None

        # player thread to read and prepare frames for display
        self._player_thread = None

        #  - setup Widget UI components

        # custom widget for displaying a resizable image
        self._frame_widget = _FrameWidget()

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
        self._previous_frame_button.clicked.connect(self._previous_frame_clicked)
        self._previous_frame_button.setAutoRepeat(True)

        # next frame button
        self._next_frame_button = QtWidgets.QToolButton()
        self._next_frame_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._next_frame_button.setText("▶")
        self._next_frame_button.setMaximumWidth(20)
        self._next_frame_button.setMaximumHeight(20)
        self._next_frame_button.clicked.connect(self._next_frame_clicked)
        self._next_frame_button.setAutoRepeat(True)

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
        self._position_slider.setEnabled(False)

        # -- set up the layout of the components

        # player control layout
        player_control_layout = QtWidgets.QHBoxLayout()
        player_control_layout.setContentsMargins(2, 0, 2, 0)
        player_control_layout.addWidget(self._play_button)
        player_control_layout.addWidget(self._position_slider)
        player_control_layout.addLayout(frame_button_layout)

        # main widget layout
        player_layout = QtWidgets.QVBoxLayout()
        player_layout.addWidget(self._frame_widget)
        player_layout.addLayout(time_layout)
        player_layout.addLayout(player_control_layout)

        self.setLayout(player_layout)

    def _cleanup_player_thread(self):
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

    def current_frame(self):
        """return the current frame"""
        assert self._video_stream is not None
        return self._position_slider.value()

    def num_frames(self):
        """get total number of frames in the loaded video"""
        assert self._video_stream is not None
        return self._video_stream.num_frames

    def reset(self):
        """reset video player"""
        self._video_stream = None
        self._identities = None
        self._pose_est = None
        self._active_identity = 0
        self._position_slider.setValue(0)
        self._position_slider.setEnabled(False)
        self._play_button.setEnabled(False)
        self._disable_frame_buttons()
        self._frame_label.setText("")
        self._time_label.setText("")
        self._frame_widget.reset()

    def stream_fps(self):
        """get frames per second from loaded video"""
        assert self._video_stream is not None
        return self._video_stream.fps

    def show_closest(self, enabled: bool | None = None):
        """change "show closest" state. Accepts a new boolean value, or toggles"""
        if enabled is None:
            self._label_closest = not self._label_closest
        else:
            self._label_closest = enabled

        if self._player_thread:
            self._player_thread.setLabelClosest.emit(self._label_closest)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show/hide the overlay
                self._seek(self._position_slider.value())

    def show_track(self, enabled: bool | None = None):
        """change "show track" state.

        Accepts a new boolean value, or toggles current state if no value given.
        """
        if enabled is None:
            self._show_track = not self._show_track
        else:
            self._show_track = enabled

        if self._player_thread:
            self._player_thread.setShowTrack.emit(self._show_track)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show the overlay
                self._seek(self._position_slider.value())

    def overlay_pose(self, enabled: bool | None = None):
        """change "overlay pose" state.

        Accepts a new boolean value, or toggles current state if no value given.
        """
        if enabled is None:
            self._overlay_pose = not self._overlay_pose
        else:
            self._overlay_pose = enabled

        if self._player_thread:
            self._player_thread.setOverlayPose.emit(self._overlay_pose)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show the overlay
                self._seek(self._position_slider.value())

    def overlay_segmentation(self, enabled: bool | None = None):
        """change "overlay segmentation" state.

        Accepts a new boolean value, or toggles current state if no value given.
        """
        if enabled is None:
            self._overlay_segmentation = not self._overlay_segmentation
        else:
            self._overlay_segmentation = enabled

        if self._player_thread:
            self._player_thread.setOverlaySegmentation.emit(self._overlay_segmentation)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show the overlay
                self._seek(self._position_slider.value())

    def overlay_landmarks(self, enabled: bool | None = None):
        """change "overlay landmarks" state.

        Accepts a new boolean value, or toggles current state if no value given.
        """
        if enabled is None:
            self._overlay_landmarks = not self._overlay_landmarks
        else:
            self._overlay_landmarks = enabled

        if self._player_thread:
            self._player_thread.setOverlayLandmarks.emit(self._overlay_landmarks)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show the overlay
                self._seek(self._position_slider.value())

    def load_video(self, path: Path, pose_est):
        """load a new video source

        Args:
            path: path to video file
            pose_est: pose file for this video
        """
        # cleanup the old player thread if it exists
        self._cleanup_player_thread()

        # reset the button state to not playing
        self.stop()
        self.reset()

        self._video_stream = VideoReader(path)
        self._pose_est = pose_est
        self._identities = pose_est.identities

        self._player_thread = PlayerThread(
            self._video_stream,
            self._pose_est,
            self._active_identity,
            self._show_track,
            self._overlay_pose,
            self._identities,
            self._overlay_landmarks,
            self._overlay_segmentation,
        )

        self._player_thread.newImage.connect(self._display_image)
        self._player_thread.updatePosition.connect(self._set_position)
        self._player_thread.endOfFile.connect(self.stop)

        # set up the position slider
        self._seek(0)
        self._position_slider.setMaximum(self._video_stream.num_frames - 1)
        self._position_slider.setEnabled(True)

        # enable the play button and next/previous frame buttons
        self._play_button.setEnabled(True)
        self._enable_frame_buttons()

    def stop(self):
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

    def _seek(self, position: int):
        self._player_thread.seek(position)
        self.updateFrameNumber.emit(position)
        self._update_time_display(position)

    def _position_slider_clicked(self):
        """Click event for position slider.

        Seek to the new position of the slider. If the video is playing we will temporarily stop playback and
        playback will resume when slider is released. New frame is displayed with the updated position.
        """
        # this prevents the player thread from updating the position of the
        # slider after we start seeking
        self._seeking = True
        pos = self._position_slider.value()

        # stop playback while the slider is being manipulated
        self._player_thread.stop_playback()

        # seek to the slider position and update the displayed frame
        self._seek(pos)

    def _position_slider_moved(self):
        """position slider move event.

        seeks the video to the new frame and displays it
        """
        self._seek(self._position_slider.value())

    def _position_slider_release(self):
        """release event for the position slider.

        The new position gets updated for the final time. If we were playing
        when we clicked the slider then we resume playing after releasing.
        """
        self._seek(self._position_slider.value())
        self._seeking = False

        # resume playing
        if self._playing:
            self._start_player_thread()

    def _next_frame_clicked(self):
        """handle "clicked" signal for the next frame button.

        Need to wrap self.next_frame because it takes an optional parameter and Qt always
        passes a boolean argument to the click signal handler that is meaningless unless
        the button is "checkable"
        """
        self.next_frame()

    def _previous_frame_clicked(self):
        """handle "clicked" signal for the previous frame button.

        Need to wrap self.previous_frame because it takes an optional parameter and Qt always
        passes a boolean argument to the click signal handler that is meaningless unless the
        button is "checkable"
        """
        self.previous_frame()

    def toggle_play(self):
        """handle clicking on the play/pause button

        don't do anything if a video hasn't been loaded, or if we are seeking
        (user clicks space bar while dagging slider)
        """
        if self._player_thread is None or self._seeking:
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

    def next_frame(self, frames=1):
        """advance to the next frame and display it

        Args:
            frames: optional, number of frames to advance
        """
        self.seek_to_frame(self._position_slider.value() + frames)

    def previous_frame(self, frames=1):
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

        # if new_frame == current value of the position slider we are at the
        # beginning of the video, don't do anything. Otherwise, seek to the
        # new frame and display it.
        if new_frame != self._position_slider.value():
            self._position_slider.setValue(new_frame)
            self._player_thread.seek(new_frame)

    def set_active_identity(self, identity: int):
        """set an active identity, which will be labeled in the video"""
        # don't do anything if a video isn't loaded
        if self._player_thread is None:
            return

        self._active_identity = identity
        self._player_thread.setActiveIdentity.emit(identity)
        if not self._playing:
            self._player_thread.seek(self._position_slider.value())

    @property
    def pixmap_clicked(self) -> QtCore.SignalInstance:
        """return the pixmap_clicked signal from the frame widget"""
        return self._frame_widget.pixmap_clicked

    def _enable_frame_buttons(self):
        """enable the previous/next frame buttons"""
        self._next_frame_button.setEnabled(True)
        self._previous_frame_button.setEnabled(True)

    def _disable_frame_buttons(self):
        """disable the previous/next frame buttons"""
        self._next_frame_button.setEnabled(False)
        self._previous_frame_button.setEnabled(False)

    def _update_time_display(self, frame_number):
        """update the time and current frame labels with current frame

        Args:
            frame_number: current frame number
        """
        self._frame_label.setText(f"{frame_number}:{self._video_stream.num_frames - 1}")
        self._time_label.setText(self._video_stream.get_frame_time(frame_number))

    @QtCore.Slot(QImage)
    def _display_image(self, image: QImage):
        """display a new frame sent from the player thread

        Args:
            image (QImage): frame ready for display as emitted by player thread
        """
        self._frame_widget.setPixmap(QtGui.QPixmap.fromImage(image))

    @QtCore.Slot(int)
    def _set_position(self, frame_number: int):
        """update the position slider during playback

        Args:
            frame_number (int): frame_number emitted by player thread
        """
        # don't update the slider value if user is seeking, since that can interfere.
        if self._seeking:
            return

        self._position_slider.setValue(frame_number)
        self._update_time_display(frame_number)
        self.updateFrameNumber.emit(frame_number)

    def _start_player_thread(self):
        """start video playback in player thread"""
        self._player_thread.start()
        self._playing = True
