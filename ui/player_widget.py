import time
from PyQt5 import QtCore, QtGui, QtWidgets

from video_stream import VideoStream


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

    def __init__(self, video_stream):
        QtCore.QThread.__init__(self)
        self.stream = video_stream

    def terminate(self):
        """
        tell run thread to stop playback
        """
        self.stream.stop()

    def run(self):
        """
        method to be run as a thread during playback
        handles grabbing the next frame from the buffer, converting to a QImage,
        and sending to the UI component for display.
        """

        # flag used to terminate loop after we've displayed the last frame
        end_of_file = False

        # tell stream to start buffering frames
        self.stream.start()

        # no delay before showing the first frame
        delay = 0

        # iterate until we've been told to stop (user clicks pause button)
        # or we reach end of file
        while not self.stream.stopped and not end_of_file:
            # time iteration to account for it in delay between frame refresh
            iteration_start = time.perf_counter()

            # grab next frame from stream buffer
            frame = self.stream.read()

            if frame['data'] is not None:
                # convert OpenCV image (numpy array) to QImage
                image = QtGui.QImage(frame['data'], frame['data'].shape[1],
                                     frame['data'].shape[0],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
                # don't update frame until we've shown the last one for the
                # required duration
                time.sleep(delay)

                # send the new frame and the frame index to the UI components
                self.newImage.emit(image)
                self.updatePosition.emit(frame['index'])

                # 'iteration time' is how long this loop took minus the time we
                # slept
                iteration_time = time.perf_counter() - iteration_start - delay

                # next iteration we sleep the frame duration minus how long
                # this loop took.
                delay = max(frame['duration'] - iteration_time, 0)
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
        super().__init__()

        # initially we want the _FrameWidget to expand to the true size of the
        # image
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.firstFrame = True

    def paintEvent(self, event):
        """
        override the paintEvent() handler to scale the image if the widget is
        resized. We don't allow resizing the first frame so the size of the
        widget will be expanded to fit the actual size of the frame.
        """

        # only draw if we have an image to show
        if self.pixmap() is not None:
            # current size of the widget
            size = self.size()

            painter = QtGui.QPainter(self)
            point = QtCore.QPoint(0, 0)

            # scale the image to the current size of the widget. With the
            # initial Expanding size policy, the widget will have already been
            # expanded to the size of the frame
            # once we change the size policy, this will resize the image to
            # fit the current size of the widget
            pix = self.pixmap().scaled(
                size,
                QtCore.Qt.KeepAspectRatio,
                transformMode=QtCore.Qt.FastTransformation
            )

            # because we are maintaining aspect ratio, the scaled frame might
            # not be the same dimensions as the area we are painting it.
            # adjust the start point to center the image in the widget
            point.setX((size.width() - pix.width()) / 2)
            point.setY((size.height() - pix.height()) / 2)
            painter.drawPixmap(point, pix)

            # after we let the first frame expand the widget (and it's parent)
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

    def __init__(self, *args, **kwargs):
        super(PlayerWidget, self).__init__(*args, **kwargs)

        # keep track of the current state
        self._playing = False
        self._seeking = False

        # VideoStream object, will be initialized when video is loaded
        self._video_stream = None

        # player thread spawned during playback
        self._player_thread = None

        # setup Widget UI components

        # custom widget for displaying a resizable image
        self._frame_widget = _FrameWidget()

        # the player controls

        # play/pause button
        self._play_button = QtWidgets.QPushButton()
        self._play_button.setCheckable(True)
        self._play_button.setEnabled(False)
        self._play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self._play_button.clicked.connect(self.play)

        # previous frame button
        self._previous_frame_button = QtWidgets.QPushButton("◀")
        self._previous_frame_button.setMaximumWidth(20)
        self._previous_frame_button.setMaximumHeight(20)
        self._previous_frame_button.clicked.connect(self.previous_frame)
        self._previous_frame_button.setAutoRepeat(True)

        # next frame button
        self._next_frame_button = QtWidgets.QPushButton("▶")
        self._next_frame_button.setMaximumWidth(20)
        self._next_frame_button.setMaximumHeight(20)
        self._next_frame_button.clicked.connect(self.next_frame)
        self._next_frame_button.setAutoRepeat(True)

        # prev/next frame buttons are disabled until a video is loaded
        self._disable_frame_buttons()

        # position slider
        self._position_slider = QtWidgets.QSlider(
            minimum=0, maximum=0, orientation=QtCore.Qt.Horizontal)
        self._position_slider.sliderMoved.connect(self._position_slider_moved)
        self._position_slider.sliderPressed.connect(
            self._position_slider_clicked)
        self._position_slider.sliderReleased.connect(
            self._position_slider_release)
        self._position_slider.setEnabled(False)

        # setup the layout of the components

        # player control layout
        player_control_layout = QtWidgets.QHBoxLayout()
        player_control_layout.setContentsMargins(0, 0, 0, 0)
        player_control_layout.addWidget(self._play_button)
        player_control_layout.addWidget(self._position_slider)
        player_control_layout.addWidget(self._previous_frame_button)
        player_control_layout.addWidget(self._next_frame_button)

        # main widget layout
        player_layout = QtWidgets.QVBoxLayout()
        player_layout.addWidget(self._frame_widget)
        player_layout.addLayout(player_control_layout)

        self.setLayout(player_layout)

    def __del__(self):
        if self._player_thread:
            self._player_thread.terminate()

    def load_video(self, path):
        """
        load a new video source
        :param path: path to video file
        """
        self._video_stream = VideoStream(path)
        self._frame_widget.firstFrame = True
        self._update_frame(self._video_stream.read())
        self._position_slider.setValue(0)
        self._position_slider.setMaximum(self._video_stream.num_frames - 1)
        self._play_button.setEnabled(True)
        self._enable_frame_buttons()
        self._position_slider.setEnabled(True)

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

    def play(self):
        """
        handle clicking on the play button
        """
        # don't do anything if a video hasn't been loaded
        if self._video_stream is None:
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

    def stop(self):
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

        # change the icon to play
        self._play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        # switch the button state to off
        self._play_button.setChecked(False)

        self._enable_frame_buttons()
        self._playing = False

    def next_frame(self):
        """
        advance to the next frame and display it
        """

        # don't do anything if a video hasn't been loaded
        if self._video_stream is None:
            return

        new_frame = min(self._position_slider.value() + 1,
                        self._position_slider.maximum())

        # if new_frame == the current value of the position slider we are at
        # the end of the video, don't do anything. Otherwise, show the next
        # frame and advance the slider.
        if new_frame != self._position_slider.value():
            self._position_slider.setValue(new_frame)
            # make sure the buffer isn't empty before calling video_stream.read
            self._video_stream.load_next_frame()
            self._update_frame(self._video_stream.read())

    def previous_frame(self):
        """
        go back to the previous frame and display it
        """

        # don't do anything if a video hasn't been loaded
        if self._video_stream is None:
            return

        new_frame = max(self._position_slider.value() - 1, 0)

        # if new_frame == current value of the position slider we are at the
        # beginning of the video, don't do anything. Otherwise, seek to the
        # new frame and display it.
        if new_frame != self._position_slider.value():
            self._position_slider.setValue(new_frame)
            self._video_stream.seek(new_frame)
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
            image = QtGui.QImage(frame['data'], frame['data'].shape[1],
                                 frame['data'].shape[0],
                                 QtGui.QImage.Format_RGB888).rgbSwapped()
            self._display_image(image)
            self.updateFrameNumber.emit(frame['index'])

    @QtCore.pyqtSlot(QtGui.QImage)
    def _display_image(self, image):
        """
        display a new QImage sent from the player thread
        :param image: QImage to display as next frame
        """
        self._frame_widget.setPixmap(QtGui.QPixmap.fromImage(image))

    @QtCore.pyqtSlot(int)
    def _set_position(self, frame_number):
        """
        update the value of the position slider to the frame number sent from
        the player thread
        :param frame_number: new value for the progress slider
        """
        # don't update the slider value if user is seeking, since that can
        # interfere
        if not self._seeking:
            self._position_slider.setValue(frame_number)
            self.updateFrameNumber.emit(frame_number)

    def _start_player_thread(self):
        """
        start a new player thread and connect it to the UI components
        """
        self._player_thread = _PlayerThread(self._video_stream)
        self._player_thread.newImage.connect(self._display_image)
        self._player_thread.updatePosition.connect(self._set_position)
        self._player_thread.endOfFile.connect(self.stop)
        self._player_thread.start()
        self._playing = True
