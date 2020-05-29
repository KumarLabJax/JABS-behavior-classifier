import time
from PyQt5 import QtCore, QtGui, QtWidgets

from src.video_stream import VideoStream, label_identity, label_all_identities
from src.pose_estimation import PoseEstimationV3


class _PlayerThread(QtCore.QThread):
    """
    thread used to grab frames (numpy arrays) from a video stream and convert
    them to a QImage for display by the frame widget

    handles timing to get correct playback speed
    """

    # signals used to update the UI components from the thread
    newImage = QtCore.pyqtSignal('QImage')
    updatePosition = QtCore.pyqtSignal(int)
    endOfFile = QtCore.pyqtSignal()

    def __init__(self, video_stream, pose_est, identity=None):
        QtCore.QThread.__init__(self)
        self._stream = video_stream
        self._pose_est = pose_est
        self._identity = identity
        self._label_identities = False
        self._identities = []

    def terminate(self):
        """
        tell run thread to stop playback
        """
        self._stream.stop()

    def set_identity(self, identity):
        """
        set the active identity -- we will draw a dot to indicate the position
        of the selected identity
        :param identity: new selected identity
        :return: None
        """
        self._identity = identity

    def set_label_identities(self, val):
        self._label_identities = val

    def set_labels(self, identities):
        self._identities = identities

    def run(self):
        """
        method to be run as a thread during playback
        handles grabbing the next frame from the buffer, converting to a QImage,
        and sending to the UI component for display.
        """

        # flag used to terminate loop after we've displayed the last frame
        end_of_file = False

        # tell stream to start buffering frames
        self._stream.start()

        next_timestamp = 0
        start_time = 0

        # iterate until we've been told to stop (user clicks pause button)
        # or we reach end of file
        while not self._stream.stopped and not end_of_file:
            now = time.perf_counter()

            # grab next frame from stream buffer
            frame = self._stream.read()

            if frame['data'] is not None:

                if self._label_identities and self._identities:
                    # label all identities mode
                    label_all_identities(frame['data'], self._pose_est, self._identities, frame['index'])

                elif self._identity is not None:
                    # if active identity set, label it on the frame
                    label_identity(frame['data'],
                                   *self._pose_est.get_points(frame['index'],
                                                              self._identity))

                # convert OpenCV image (numpy array) to QImage
                image = QtGui.QImage(frame['data'], frame['data'].shape[1],
                                     frame['data'].shape[0],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()

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
                if not self._stream.stopped:
                    self.newImage.emit(image)
                    self.updatePosition.emit(frame['index'])

                # update timestamp for when should the next frame be shown
                next_timestamp += frame['duration']

            else:
                # if the video stream reached the end of file let the UI know
                self.endOfFile.emit()
                # and terminate the loop
                end_of_file = True


class _FrameWidget(QtWidgets.QLabel):
    """
    widget that implements a resizable pixmap label, will initialize to the
    full size of the video frame, but then will be resizable after that
    """
    def __init__(self):
        super(_FrameWidget, self).__init__()

        # initially we want the _FrameWidget to expand to the true size of the
        # image
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.firstFrame = True

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

    def paintEvent(self, event):
        """
        override the paintEvent() handler to scale the image if the widget is
        resized.
        Don't enable resizing until after  the first frame has been drawn so
        that the widget will be expanded to fit the actual size of the frame.
        """

        # only draw if we have an image to show
        if self.pixmap() is not None:
            # current size of the widget
            size = self.size()

            painter = QtGui.QPainter(self)
            point = QtCore.QPoint(0, 0)

            # scale the image to the current size of the widget. First frame,
            # the widget will be expanded to fit the full size image.
            pix = self.pixmap().scaled(
                size,
                QtCore.Qt.KeepAspectRatio,
                transformMode=QtCore.Qt.SmoothTransformation
            )

            # because we are maintaining aspect ratio, the scaled frame might
            # not be the same dimensions as the area we are painting it.
            # adjust the start point to center the image in the widget
            point.setX((size.width() - pix.width()) / 2)
            point.setY((size.height() - pix.height()) / 2)

            # draw the pixmap starting at the new calculated offset
            painter.drawPixmap(point, pix)

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
    updateFrameNumber = QtCore.pyqtSignal(int)

    # let the main window UI know what the list of identities should be
    updateIdentities = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(PlayerWidget, self).__init__(*args, **kwargs)

        # keep track of the current state
        self._playing = False
        self._seeking = False

        # VideoStream object, will be initialized when video is loaded
        self._video_stream = None

        # track annotation
        self._tracks = None

        self._label_all_identities = False
        self._identities = []

        # currently selected identity -- if set will be labeled in the video
        self._active_identity = None

        # player thread spawned during playback
        self._player_thread = None

        #  - setup Widget UI components

        # custom widget for displaying a resizable image
        self._frame_widget = _FrameWidget()

        #  -- player controls

        # current time and frame display
        font = QtGui.QFont("Courier New", 14)
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
        self._play_button.setCheckable(True)
        self._play_button.setEnabled(False)
        self._play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self._play_button.clicked.connect(self.toggle_play)

        # previous frame button
        self._previous_frame_button = QtWidgets.QPushButton("◀")
        self._previous_frame_button.setMaximumWidth(35)
        self._previous_frame_button.clicked.connect(
            self._previous_frame_clicked)
        self._previous_frame_button.setAutoRepeat(True)

        # next frame button
        self._next_frame_button = QtWidgets.QPushButton("▶")
        self._next_frame_button.setMaximumWidth(35)
        self._next_frame_button.clicked.connect(self._next_frame_clicked)
        self._next_frame_button.setAutoRepeat(True)

        # previous/next button layout
        frame_button_layout = QtWidgets.QHBoxLayout()
        frame_button_layout.setSpacing(0)
        frame_button_layout.addWidget(self._previous_frame_button)
        frame_button_layout.addWidget(self._next_frame_button)
        # prev/next frame buttons are disabled until a video is loaded
        self._disable_frame_buttons()

        # position slider
        self._position_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self._position_slider.sliderMoved.connect(self._position_slider_moved,
                                                  QtCore.Qt.QueuedConnection)
        self._position_slider.sliderPressed.connect(
            self._position_slider_clicked)
        self._position_slider.sliderReleased.connect(
            self._position_slider_release)
        self._position_slider.setEnabled(False)

        # -- setup the layout of the components

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
            self._player_thread.terminate()

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
        self._tracks = None
        self._position_slider.setValue(0)
        self._position_slider.setEnabled(False)
        self.updateIdentities.emit([])
        self._play_button.setEnabled(False)
        self._disable_frame_buttons()
        self._frame_label.setText('')
        self._time_label.setText('')
        self._frame_widget.reset()

    def stream_fps(self):
        """ get frames per second from loaded video """
        assert self._video_stream is not None
        return self._video_stream.fps

    def set_identity_label_mode(self, enabled):
        self._label_all_identities = enabled

        # don't do anything else if a video isn't loaded
        if self._video_stream is None:
            return

        if self._player_thread:
            if enabled:
                self._player_thread.set_labels(self._identities)
            self._player_thread.set_label_identities(enabled)
        else:
            self._video_stream.seek(self._position_slider.value())
            self._video_stream.load_next_frame()
            self._update_frame(self._video_stream.read())

    def set_identity_labels(self, identities):
        self._identities = identities

    def load_video(self, path):
        """
        load a new video source
        :param path: path to video file
        """

        # if we already have a video loaded make sure it is stopped
        self._stop()
        self.reset()

        # load the video and pose file
        self._video_stream = VideoStream(path)
        self._tracks = PoseEstimationV3(path)

        # setup the position slider
        self._position_slider.setMaximum(self._video_stream.num_frames - 1)
        self._position_slider.setEnabled(True)

        # tell main window to populate the identity selection drop down
        # this will cause _set_active_identity() to be called, which will load
        # and display the current frame
        self.updateIdentities.emit(self._tracks.identities)

        # enable the play button and next/previous frame buttons
        self._play_button.setEnabled(True)
        self._enable_frame_buttons()

    def _stop(self):
        """
        stop playing and reset the play button to its initial state
        """
        # don't do anything if a video hasn't been loaded
        if self._video_stream is None:
            return

        # if we have an active player thread, terminate
        if self._player_thread:
            self._player_thread.terminate()
            self._player_thread.wait()
            self._player_thread = None

        # seek to the current position of the slider -- it's possible a
        # player thread had read a frame or two beyond that but the frames
        # were discarded when they arrived at the UI thread after playback
        # stopped. This makes sure we don't skip over those frames when
        # playback resumes.
        self._video_stream.seek(self._position_slider.value())
        self._update_frame(self._video_stream.read())

        # change the icon to play
        self._play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        # switch the button state to off
        self._play_button.setChecked(False)

        self._enable_frame_buttons()
        self._playing = False

    def _position_slider_clicked(self):
        """
        Click event for position slider.
        Seek to the new position of the slider. If the video is playing, the
        current player thread is terminated, and we seek to the new position.
        New frame is displayed.
        """

        # this prevents the player thread from updating the position of the
        # slider after we start seeking
        self._seeking = True
        pos = self._position_slider.value()

        # if the video is playing, we stop the player thread and wait for it
        # to terminate
        if self._player_thread:
            self._player_thread.terminate()
            self._player_thread.wait()
            self._player_thread = None

        # seek to the slider position and update the displayed frame
        self._video_stream.seek(pos)
        self._update_frame(self._video_stream.read())

    def _position_slider_moved(self):
        """
        position slider move event. seek the video to the new frame and display
        it
        """
        self._video_stream.seek(self._position_slider.value())
        self._update_frame(self._video_stream.read())

    def _position_slider_release(self):
        """
        release event for the position slider.
        The new position gets updated for the final time. If we were playing
        when we clicked the slider then we resume playing after releasing.
        :return:
        """
        self._video_stream.seek(self._position_slider.value())
        self._update_frame(self._video_stream.read())
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
        """
        # don't do anything if a video hasn't been loaded, or if we are seeking
        if self._video_stream is None or self._seeking:
            return

        if self._playing:
            # if we are playing, stop
            self._stop()

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

        # don't do anything if a video hasn't been loaded or if the video is
        # playing
        if self._video_stream is None or self._playing:
            return

        new_frame = min(self._position_slider.value() + frames,
                        self._position_slider.maximum())

        # if new_frame == the current value of the position slider we are at
        # the end of the video, don't do anything. Otherwise, show the next
        # frame and advance the slider.
        if new_frame != self._position_slider.value():
            self._position_slider.setValue(new_frame)

            if frames == 1:
                # make sure the buffer isn't empty
                self._video_stream.load_next_frame()
            else:
                # skipping ahead by more than one frame, can rely on the
                # basic read ahead buffering to get the frame
                self._video_stream.seek(new_frame)

            frame = self._video_stream.read()
            self._update_frame(frame)

    def previous_frame(self, frames=1):
        """
        go back to the previous frame and display it
        :param frames: optional number of frames to move back
        """

        # don't do anything if a video hasn't been loaded or if the video is
        # playing
        if self._video_stream is None or self._playing:
            return

        new_frame = max(self._position_slider.value() - frames, 0)

        # if new_frame == current value of the position slider we are at the
        # beginning of the video, don't do anything. Otherwise, seek to the
        # new frame and display it.
        if new_frame != self._position_slider.value():
            self._position_slider.setValue(new_frame)
            self._video_stream.seek(new_frame)
            frame = self._video_stream.read()
            self._update_frame(frame)

    def _set_active_identity(self, identity):
        """ set an active identity, which will be labeled in the video """

        # don't do anything if a video isn't loaded
        if self._video_stream is None:
            return

        self._active_identity = identity
        if self._player_thread:
            self._player_thread.set_identity(identity)
        else:
            self._video_stream.seek(self._position_slider.value())
            self._video_stream.load_next_frame()
            self._update_frame(self._video_stream.read())

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

    def _update_frame(self, frame):
        """
        update the displayed frame
        :param frame: dict returned by video_stream.read()
        """
        if frame['index'] != -1:
            if self._label_all_identities:
                label_all_identities(frame['data'], self._tracks,
                                     self._identities, frame['index'])
            else:
                label_identity(frame['data'],
                               *self._tracks.get_points(frame['index'],
                                                        self._active_identity))

            image = QtGui.QImage(frame['data'], frame['data'].shape[1],
                                 frame['data'].shape[0],
                                 QtGui.QImage.Format_RGB888).rgbSwapped()
            self._display_image(image)
            self.updateFrameNumber.emit(frame['index'])
            self._update_time_display(frame['index'])

    def _update_time_display(self, frame_number):
        """
        update the time and current frame labels with current frame
        :param frame_number: current frame number
        """

        self._frame_label.setText(
            f"{frame_number}:{self._video_stream.num_frames - 1}")
        self._time_label.setText(
            self._video_stream.get_frame_time(frame_number))

    @QtCore.pyqtSlot(QtGui.QImage)
    def _display_image(self, image):
        """
        display a new QImage sent from the player thread
        :param image: QImage to display as next frame
        """
        # make sure the video stream hasn't been closed since this signal
        # was sent
        if self._video_stream is None:
            return
        self._frame_widget.setPixmap(QtGui.QPixmap.fromImage(image))

    @QtCore.pyqtSlot(int)
    def _set_position(self, frame_number):
        """
        update the value of the position slider to the frame number sent from
        the player thread
        :param frame_number: new value for the progress slider
        """
        # don't update the slider value if user is seeking, since that can
        # interfere.
        # Also make sure the video stream hasn't been closed since the signal
        # was sent.
        if not self._playing or self._seeking or self._video_stream is None:
            return

        self._position_slider.setValue(frame_number)
        self.updateFrameNumber.emit(frame_number)
        self._update_time_display(frame_number)

    def _start_player_thread(self):
        """
        start a new player thread and connect it to the UI components
        """
        self._player_thread = _PlayerThread(self._video_stream,
                                            self._tracks,
                                            self._active_identity)
        self._player_thread.newImage.connect(self._display_image)
        self._player_thread.updatePosition.connect(self._set_position)
        self._player_thread.endOfFile.connect(self._stop)
        self._player_thread.start()
        self._playing = True
