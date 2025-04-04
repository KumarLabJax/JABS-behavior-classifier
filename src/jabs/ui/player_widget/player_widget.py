from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QPaintEvent

from .player_thread import PlayerThread
from jabs.video_reader import VideoReader


class _FrameWidget(QtWidgets.QLabel):
    """
    widget that implements a resizable pixmap label, will initialize to the
    full size of the video frame, but then will be resizable after that
    """

    pixmap_clicked = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()

        # initially we want the _FrameWidget to expand to the true size of the
        # image
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.firstFrame = True
        self._scaled_pix_x = 0
        self._scaled_pix_y = 0
        self._scaled_pix_width = 0
        self._scaled_pix_height = 0

    def mousePressEvent(self, event):
        pix_x, pix_y = self._frame_xy_to_pixmap_xy(event.x(), event.y())
        self.pixmap_clicked.emit({'x': pix_x, 'y': pix_y})

        QtWidgets.QLabel.mousePressEvent(self, event)

    def _frame_xy_to_pixmap_xy(self, x, y):
        """
        Convert the given x, y coordinates from _FrameWidget coordinates
        to pixmap coordinates. Ie which pixel did the user click on?
        We account for image scaling and translation
        """
        pixmap = self.pixmap()
        if pixmap is not None:

            if (self._scaled_pix_height != pixmap.height()
                    or self._scaled_pix_width != pixmap.width()
                    or self._scaled_pix_x != 0
                    or self._scaled_pix_y != 0):

                if self._scaled_pix_width >= 1 and self._scaled_pix_height >= 1:
                    # we've done all the checks and it's safe to transform
                    # the x, y point
                    x -= self._scaled_pix_x
                    y -= self._scaled_pix_y
                    x *= pixmap.width() / self._scaled_pix_width
                    y *= pixmap.height() / self._scaled_pix_height

        return x, y

    def sizeHint(self):
        """
        Override QLabel.sizeHint to give an initial starting size.
        """
        return QtCore.QSize(800, 800)

    def reset(self):
        """ reset state of frame widget  """
        self.clear()
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.firstFrame = True

    def paintEvent(self, event: QPaintEvent):
        """
        override the paintEvent() handler to scale the image if the widget is
        resized.
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
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
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

            # after we let the first frame expand the widget
            # switch to the Ignored size policy and we will resize the image to
            # fit the widget
            if self.firstFrame:
                self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                   QtWidgets.QSizePolicy.Ignored)
                self.firstFrame = False
        else:
            # if we don't have a pixmap to display just call the original QLabel
            # paintEvent
            super().paintEvent(event)


class PlayerWidget(QtWidgets.QWidget):
    """
    Video Player Widget. Consists of a QLabel to display a frame image, and
    basic player controls below the frame (play/pause button, position slider,
    previous/next frame buttons.

    position slider can be dragged while video is paused or is playing and the
    position will be updated. If video was playing when the slider is dragged
    the playback will resume after the slider is released.
    """

    # signal to allow parent UI component to observe current frame number
    updateFrameNumber = QtCore.Signal(int)

    # let the main window UI know what the list of identities should be
    updateIdentities = QtCore.Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self._time_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                       QtWidgets.QSizePolicy.Fixed)
        self._frame_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        time_layout = QtWidgets.QHBoxLayout()
        time_layout.addWidget(self._time_label)
        time_layout.addStretch()
        time_layout.addWidget(self._frame_label)

        # play/pause button
        self._play_button = QtWidgets.QPushButton()
        self._play_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self._play_button.setCheckable(True)
        self._play_button.setEnabled(False)
        self._play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self._play_button.clicked.connect(self.toggle_play)

        # previous frame button
        self._previous_frame_button = QtWidgets.QToolButton()
        self._previous_frame_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self._previous_frame_button.setText("◀")
        self._previous_frame_button.setMaximumWidth(20)
        self._previous_frame_button.setMaximumHeight(20)
        self._previous_frame_button.clicked.connect(
            self._previous_frame_clicked)
        self._previous_frame_button.setAutoRepeat(True)

        # next frame button
        self._next_frame_button = QtWidgets.QToolButton()
        self._next_frame_button.setFocusPolicy(QtCore.Qt.NoFocus)
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
        self._position_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self._position_slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self._position_slider.sliderMoved.connect(self._position_slider_moved,
                                                  QtCore.Qt.QueuedConnection)
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

    def __del__(self):
        # make sure we terminate the player thread if it is still active
        # during destruction
        if self._player_thread:
            self._player_thread.stop_playback()
            self._player_thread.wait()
            self._player_thread = None

    def current_frame(self):
        """ return the current frame """
        assert self._video_stream is not None
        return self._position_slider.value()

    def num_frames(self):
        """ get total number of frames in the loaded video  """
        assert self._video_stream is not None
        return self._video_stream.num_frames

    def reset(self):
        """ reset video player """
        self._video_stream = None
        self._identities = None
        self._pose_est = None
        self._active_identity = 0
        self._position_slider.setValue(0)
        self._position_slider.setEnabled(False)
        self._play_button.setEnabled(False)
        self._disable_frame_buttons()
        self._frame_label.setText('')
        self._time_label.setText('')
        self._frame_widget.reset()

    def stream_fps(self):
        """ get frames per second from loaded video """
        assert self._video_stream is not None
        return self._video_stream.fps

    def show_closest(self, enabled: bool | None = None):
        """
        change "show closest" state. Accepts a new boolean value, or toggles
        """
        if enabled is None:
            self._label_closest = not self._label_closest
        else:
            self._label_closest = enabled

        if self._player_thread:
            self._player_thread.label_closest(self._label_closest)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show/hide the overlay
                self._seek(self._position_slider.value())

    def show_track(self, new_val: bool | None = None):
        """
        change "show track" state. Accepts a new boolean value, or toggles
        current state if no value given.
        """
        if new_val is None:
            self._show_track = not self._show_track
        else:
            self._show_track = new_val

        if self._player_thread:
            self._player_thread.set_show_track(self._show_track)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show the overlay
                self._seek(self._position_slider.value())

    def overlay_pose(self, new_val: bool | None = None):
        """
        change "overlay pose" state. Accepts a new boolean value, or toggles
        current state if no value given.
        """
        if new_val is None:
            self._overlay_pose = not self._overlay_pose
        else:
            self._overlay_pose = new_val

        # don't do anything else if a video isn't loaded
        if self._video_stream is None:
            return

        if self._player_thread:
            self._player_thread.set_overlay_pose(self._overlay_pose)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show the overlay
                self._seek(self._position_slider.value())

    def overlay_segmentation(self, new_val: bool | None = none):
        """
        change "overlay segmentation" state. Accepts a new boolean value, or toggles
        current state if no value given.
        """
        if new_val is None:
            self._overlay_segmentation = not self._overlay_segmentation
        else:
            self._overlay_segmentation = new_val

        if self._player_thread:
            self._player_thread.set_overlay_segmentation(self._overlay_segmentation)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show the overlay
                self._seek(self._position_slider.value())

    def overlay_landmarks(self, new_val: bool | None = None):
        """
        change "overlay landmarks" state. Accepts a new boolean value, or
        toggles current state if no value given.
        """
        if new_val is None:
            self._overlay_landmarks = not self._overlay_landmarks
        else:
            self._overlay_landmarks = new_val

        if self._player_thread:
            self._player_thread.set_overlay_landmarks(self._overlay_landmarks)
            if not self._playing:
                # if we are not playing, we need to update the current frame
                # to show the overlay
                self._seek(self._position_slider.value())

    def set_identities(self, identities):
        self._identities = identities
        if self._player_thread:
            self._player_thread.set_identities(self._identities)

    def load_video(self, path: Path, pose_est):
        """
        load a new video source
        :param path: path to video file
        :param pose_est: pose file for this video
        """

        # if we already have a video loaded make sure it is stopped
        if self._playing:
            self.stop()
            self._player_thread.wait()
        self.reset()

        # load the video and pose file
        self._video_stream = VideoReader(path)

        self._pose_est = pose_est

        # tell main window to populate the identity selection drop down
        # this will cause set_active_identity() to be called
        self.updateIdentities.emit(self._pose_est.identities)
        self.set_identities(self._pose_est.identities)

        # set up the position slider
        self._position_slider.setMaximum(self._video_stream.num_frames - 1)
        self._position_slider.setEnabled(True)

        if self._player_thread is None:
            self._player_thread = PlayerThread(
                self._video_stream, self._pose_est, self._active_identity,
                self._show_track, self._overlay_pose, self._identities,
                self._overlay_landmarks, self._overlay_segmentation)
            self._player_thread.newImage.connect(self._display_image)
            self._player_thread.updatePosition.connect(self._set_position)
            self._player_thread.endOfFile.connect(self.stop)
            self._player_thread.label_closest(self._label_closest)
        else:
            self._player_thread.load_new_video(self._video_stream, self._pose_est, self._active_identity, self._identities)

        self._seek(0)

        # enable the play button and next/previous frame buttons
        self._play_button.setEnabled(True)
        self._enable_frame_buttons()

    def stop(self):
        """
        stop playing and reset the play button to its initial state
        """
        # if we have an active player thread, terminate
        if self._player_thread:
            self._player_thread.stop_playback()

        # change the icon to play
        self._play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        # switch the button state to off
        self._play_button.setChecked(False)

        self._enable_frame_buttons()
        self._playing = False

    def _seek(self, position: int):
        self._player_thread.seek(position)
        self.updateFrameNumber.emit(position)
        self._update_time_display(position)

    def _position_slider_clicked(self):
        """
        Click event for position slider.
        Seek to the new position of the slider. If the video is playing we will temporarily stop playback and
        playback will resume when slider is released.
        New frame is displayed with the updated position.
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
        """
        position slider move event. seek the video to the new frame and display it
        """
        self._seek(self._position_slider.value())

    def _position_slider_release(self):
        """
        release event for the position slider.
        The new position gets updated for the final time. If we were playing
        when we clicked the slider then we resume playing after releasing.
        """
        self._seek(self._position_slider.value())
        self._seeking = False

        # resume playing
        if self._playing:
            self._start_player_thread()

    def _next_frame_clicked(self):
        """
        handle "clicked" signal for the next frame button. Need to wrap
        self.next_frame because it takes an optional parameter and Qt always
        passes a boolean argument to the click signal handler that is
        meaningless unless the button is "checkable"
        """
        self.next_frame()

    def _previous_frame_clicked(self):
        """
        handle "clicked" signal for the previous frame button. Need to wrap
        self.previous_frame because it takes an optional parameter and Qt always
        passes a boolean argument to the click signal handler that is
        meaningless unless the button is "checkable"
        """
        self.previous_frame()

    def toggle_play(self):
        """
        handle clicking on the play/pause button

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
                self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            self._disable_frame_buttons()
            self._start_player_thread()

    def next_frame(self, frames=1):
        """
        advance to the next frame and display it
        :param frames: optional, number of frames to advance
        """

        # don't do anything if a video hasn't been loaded or if the video is playing
        if self._player_thread is None or self._playing:
            return

        new_frame = min(self._position_slider.value() + frames,
                        self._position_slider.maximum())

        # if new_frame == the current value of the position slider we are at
        # the end of the video, don't do anything. Otherwise, show the next
        # frame and advance the slider.
        if new_frame != self._position_slider.value():
            self._position_slider.setValue(new_frame)
            self._player_thread.seek(new_frame)

    def previous_frame(self, frames=1):
        """
        go back to the previous frame and display it
        :param frames: optional number of frames to move back
        """

        # don't do anything if a video hasn't been loaded or if the video is playing
        if self._video_stream is None or self._playing:
            return

        new_frame = max(self._position_slider.value() - frames, 0)

        # if new_frame == current value of the position slider we are at the
        # beginning of the video, don't do anything. Otherwise, seek to the
        # new frame and display it.
        if new_frame != self._position_slider.value():
            self._position_slider.setValue(new_frame)
            self._player_thread.seek(new_frame)

    def set_active_identity(self, identity):
        """ set an active identity, which will be labeled in the video """

        # don't do anything if a video isn't loaded
        if self._player_thread is None:
            return

        self._active_identity = identity
        self._player_thread.set_identity(identity)
        if not self._playing:
            self._player_thread.seek(self._position_slider.value())

    @property
    def pixmap_clicked(self):
        return self._frame_widget.pixmap_clicked

    def get_identity_mask(self):
        """
        get an array that indicates if an identity is present in a given
        frame or not for the currently selected identity
        :return: numpy uint8 array of length num_frames
        """
        return self._pose_est.identity_mask(self._active_identity)

    def _enable_frame_buttons(self):
        """
        enable the previous/next frame buttons
        """
        self._next_frame_button.setEnabled(True)
        self._previous_frame_button.setEnabled(True)

    def _disable_frame_buttons(self):
        """
        disable the previous/next frame buttons
        """
        self._next_frame_button.setEnabled(False)
        self._previous_frame_button.setEnabled(False)

    def _update_time_display(self, frame_number):
        """
        update the time and current frame labels with current frame
        :param frame_number: current frame number
        """

        self._frame_label.setText(
            f"{frame_number}:{self._video_stream.num_frames - 1}")
        self._time_label.setText(
            self._video_stream.get_frame_time(frame_number))

    @QtCore.Slot(dict)
    def _display_image(self, data: dict):
        """
        display a new QImage sent from the player thread
        :param data: dict emitted by player thread

        data dict includes QImage to display as next frame as well as the source video
        """

        # When switching videos, it's possible us to receive frames emitted by the player thread for the
        # previous video. We need to ignore those frames.
        if data['source'] != self._video_stream.filename:
            return

        self._frame_widget.setPixmap(QtGui.QPixmap.fromImage(data['image']))

    @QtCore.Slot(int)
    def _set_position(self, data: dict):
        """
        update the value of the position slider to the frame number sent from
        the player thread
        :param frame_number: new value for the progress slider
        """

        # ignore events from previous videos that haven't been processed yet
        if data["source"] != self._video_stream.filename:
            return

        # don't update the slider value if user is seeking, since that can interfere.
        if self._seeking:
            return

        frame_number = data["index"]

        self._position_slider.setValue(frame_number)
        self.updateFrameNumber.emit(frame_number)
        self._update_time_display(frame_number)

    def _start_player_thread(self):
        """
        start video playback in player thread
        """
        self._player_thread.start()
        self._playing = True
