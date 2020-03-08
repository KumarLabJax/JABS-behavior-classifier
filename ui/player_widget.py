import time
from PyQt5 import QtCore, QtGui, QtWidgets

from video_stream import VideoStream


class _PlayerThread(QtCore.QThread):
    newImage = QtCore.pyqtSignal('QImage')
    updatePosition = QtCore.pyqtSignal(int)
    endOfFile = QtCore.pyqtSignal()

    def __init__(self, video_stream):
        QtCore.QThread.__init__(self)
        self.stream = video_stream

    def terminate(self):
        self.stream.stop()

    def run(self):
        self.stream.start()
        next_sleep = 0
        while not self.stream.stopped:
            iteration_start = time.perf_counter()
            frame = self.stream.read()
            frame_data = frame['data']
            if frame_data is not None:
                image = QtGui.QImage(frame_data, frame_data.shape[1],
                                     frame_data.shape[0],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
                time.sleep(next_sleep)
                self.newImage.emit(image)
                self.updatePosition.emit(frame['index'])

                # 'iteration time' is how long the loop took minus the time we
                # slept
                iteration_time = time.perf_counter() - iteration_start - next_sleep

                # next iteration we sleep the frame duration minus how long
                # this loop took. over time the video should appear to be
                # playing back in real time.
                next_sleep = max(frame['duration'] - iteration_time, 0)
            else:
                self.endOfFile.emit()
                return


class _FrameWidget(QtWidgets.QLabel):
    """
    widget that implements a resizable pixmap label, will initialize to the
    full size of the video frame, but then will be resizable after that
    """
    def __init__(self):
        super().__init__()
        # initially we want the _FrameWidget to expand to the size of the frame
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.installEventFilter(self)
        self.firstFrame = True

    def paintEvent(self, event):
        if self.pixmap() is not None:
            size = self.size()
            painter = QtGui.QPainter(self)
            point = QtCore.QPoint(0, 0)

            pix = self.pixmap().scaled(size, QtCore.Qt.KeepAspectRatio, transformMode = QtCore.Qt.FastTransformation)
            # start painting the label from left upper corner
            point.setX((size.width() - pix.width()) / 2)
            point.setY((size.height() - pix.height()) / 2)
            painter.drawPixmap(point, pix)

            # after we let the first frame epand the widget (and it's parent)
            # switch to the tgnored size policy and handle resizing ourself.
            if self.firstFrame:
                self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                   QtWidgets.QSizePolicy.Ignored)
                self.firstFrame = False
        else:
            super().paintEvent(event)


class PlayerWidget(QtWidgets.QWidget):

    updateFrameNumber = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(PlayerWidget, self).__init__(*args, **kwargs)

        self.playing = False
        self.seeking = False
        self.video_stream = None
        self.player_thread = None

        # setup the player controls
        self.frame_widget = _FrameWidget()
        self.play_button = QtWidgets.QPushButton()
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play)

        self.previous_frame_button = QtWidgets.QPushButton("◀")
        self.previous_frame_button.setMaximumWidth(20)
        self.previous_frame_button.setMaximumHeight(20)
        self.previous_frame_button.clicked.connect(self.previous_frame)
        self.previous_frame_button.setAutoRepeat(True)

        self.next_frame_button = QtWidgets.QPushButton("▶")
        self.next_frame_button.setMaximumWidth(20)
        self.next_frame_button.setMaximumHeight(20)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.next_frame_button.setAutoRepeat(True)

        self.disable_frame_buttons()

        self.position_slider = QtWidgets.QSlider(minimum=0, maximum=0,
                                                 orientation=QtCore.Qt.Horizontal)
        self.position_slider.sliderMoved.connect(self.position_slider_moved)
        self.position_slider.sliderPressed.connect(self.position_slider_clicked)
        self.position_slider.sliderReleased.connect(
            self.position_slider_release)

        player_control_layout = QtWidgets.QHBoxLayout()
        player_control_layout.setContentsMargins(0, 0, 0, 0)
        player_control_layout.addWidget(self.play_button)
        player_control_layout.addWidget(self.position_slider)
        player_control_layout.addWidget(self.previous_frame_button)
        player_control_layout.addWidget(self.next_frame_button)

        player_layout = QtWidgets.QVBoxLayout()
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.addWidget(self.frame_widget)
        player_layout.addLayout(player_control_layout)

        self.setLayout(player_layout)

    def __del__(self):
        if self.player_thread:
            self.player_thread.terminate()

    def load_video(self, path):
        """
        load a new video source
        :param path: path to video file
        """
        self.video_stream = VideoStream(path)
        self.frame_widget.firstFrame = True
        self.update_frame(self.video_stream.read())
        self.position_slider.setValue(0)
        self.position_slider.setMaximum(self.video_stream.num_frames - 1)
        self.play_button.setEnabled(True)
        self.enable_frame_buttons()

    def position_slider_clicked(self):
        """
        Click event for position slider.
        Seek to the new position of the slider. If the video is playing, the
        current player thread is terminated, and we seek to the new position.
        New frame is displayed.
        """

        # this prevents the player thread from updating the position of the
        # slider after we start seeking
        self.seeking = True
        pos = self.position_slider.value()

        # if the video is playing, we stop the player thread and wait for it
        # to terminate
        if self.player_thread:
            self.player_thread.terminate()
            self.player_thread.wait()
            self.player_thread = None

        # seek to the slider position and update the displayed frame
        self.video_stream.seek(pos)
        self.update_frame(self.video_stream.read())

    def position_slider_moved(self):
        """
        position slider move event. seek the video to the new frame and display
        it
        """
        self.video_stream.seek(self.position_slider.value())
        self.update_frame(self.video_stream.read())

    def position_slider_release(self):
        """
        release event for the position slider.
        The new position gets updated for the final time. If we were playing
        when we clicked the slider then we resume playing after releasing.
        :return:
        """
        self.video_stream.seek(self.position_slider.value())
        self.update_frame(self.video_stream.read())
        self.seeking = False

        # resume playing
        if self.playing:
            self.start_player_thread()

    def play(self):
        """
        handle clicking on the play button
        """

        if self.playing:
            # if we are playing, stop
            self.stop()

        else:
            # we weren't already playing so start
            self.play_button.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            self.disable_frame_buttons()
            self.start_player_thread()

    def stop(self):
        """
        stop playing and rest the play button to its initial state
        """

        # if we have an active player thread, terminate
        if self.player_thread:
            self.player_thread.terminate()
            self.player_thread.wait()
            self.player_thread = None

        # change the icon to play
        self.play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        # switch the button state to off
        self.play_button.setChecked(False)

        self.enable_frame_buttons()
        self.playing = False

    def next_frame(self):
        """
        advance to the next frame and display it
        """
        new_frame = min(self.position_slider.value() + 1, self.video_stream.num_frames - 1)
        self.position_slider.setValue(new_frame)
        self.video_stream.seek(new_frame)
        self.update_frame(self.video_stream.read())

    def previous_frame(self):
        """
        go back to the previous frame and display it
        """
        new_frame = max(self.position_slider.value() - 1, 0)
        self.position_slider.setValue(new_frame)
        self.video_stream.seek(new_frame)
        self.update_frame(self.video_stream.read())

    def enable_frame_buttons(self):
        """
        enable the previous/next frame buttons
        """
        self.next_frame_button.setEnabled(True)
        self.previous_frame_button.setEnabled(True)

    def disable_frame_buttons(self):
        """
        disable the previous/next frame buttons
        """
        self.next_frame_button.setEnabled(False)
        self.previous_frame_button.setEnabled(False)

    def update_frame(self, frame):
        """
        updat ethe displayed frame using the dict returned by video_stream.read
        """
        image = QtGui.QImage(frame['data'], frame['data'].shape[1],
                             frame['data'].shape[0],
                             QtGui.QImage.Format_RGB888).rgbSwapped()
        self.display_image(image)
        self.updateFrameNumber.emit(frame['index'])

    @QtCore.pyqtSlot(QtGui.QImage)
    def display_image(self, image):
        """
        display a new QImage sent from the player thread
        """
        self.frame_widget.setPixmap(QtGui.QPixmap.fromImage(image))

    @QtCore.pyqtSlot(int)
    def set_position(self, frame_number):
        """
        update the value of the position slider to the frame number sent from
        the player thread
        :param frame_number: new value for the progress slider
        """
        # don't update the slider value if user is seeking, since that can
        # interfere
        if not self.seeking:
            self.position_slider.setValue(frame_number)
            self.updateFrameNumber.emit(frame_number)

    def start_player_thread(self):
        """
        start a new player thread and connect it
        """
        self.player_thread = _PlayerThread(self.video_stream)
        self.player_thread.newImage.connect(self.display_image)
        self.player_thread.updatePosition.connect(self.set_position)
        self.player_thread.endOfFile.connect(self.stop)
        self.player_thread.start()
        self.playing = True



