import time

import numpy as np
from PySide6 import QtCore, QtGui

from jabs.video_reader import VideoReader


class PlayerThread(QtCore.QThread):
    """Thread for grabbing frames from a video stream and converting them to QImage.

    Handles timing to achieve correct playback speed and emits signals to update UI components.

    Decoding is all this thread does. Everything drawn on a frame is painted by the
    overlays in the frame widget, so nothing here depends on the pose file or on which
    identity is selected.

    Args:
        video_reader (VideoReader): The video reader instance.
        playback_speed (float, optional): Playback rate multiplier. Defaults to 1.0.

    Signals:
        newImage (QImage, int): Emitted with a new QImage for the PlayerWidget to
            display, and the index of the frame it was decoded from. The index travels
            with the image rather than being read from ``updatePosition``: the overlays
            are drawn from the frame number the widget was given, so an image paired
            with a stale number draws every overlay a frame behind the video.
        updatePosition (int): Emitted with the current frame index
        endOfFile: Emitted when the end of the video is reached.
    """

    # signals used to update the UI components from the thread
    newImage = QtCore.Signal(QtGui.QImage, int)
    updatePosition = QtCore.Signal(int)
    endOfFile = QtCore.Signal()

    # signals used to update the properties of PlayerThread in a thread-safe manner
    setPlaybackSpeed = QtCore.Signal(float)

    def __init__(self, video_reader: VideoReader, playback_speed: float = 1.0):
        super().__init__()

        self._video_reader = video_reader
        self._playback_speed = playback_speed

        self.setPlaybackSpeed.connect(self._set_playback_speed)

    def stop_playback(self):
        """tell run thread to stop playback"""
        self.requestInterruption()

    @QtCore.Slot(float)
    def _set_playback_speed(self, playback_speed: float):
        self._playback_speed = playback_speed

    def _read_and_emit_frame(self):
        frame = self._video_reader.load_next_frame()
        image = self._prepare_image(frame)
        self.updatePosition.emit(frame["index"])
        self.newImage.emit(image, frame["index"])

    def _prepare_image(self, frame: dict) -> QtGui.QImage | None:
        """Convert one decoded frame to a QImage, or None at end of file."""
        if frame["data"] is None:
            return None

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
            # copy() so the QImage owns its pixels. The constructor above only wraps
            # `img_rgb`, which this method drops on return, and during playback the
            # image reaches the GUI thread through a queued signal - the array is long
            # gone by the time anything reads from it. Do not remove this to save a
            # copy: what it buys is that the buffer cannot be freed underneath Qt.
            return image.copy()
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
                    self.newImage.emit(image, frame["index"])
                    self.updatePosition.emit(frame["index"])

                # update timestamp for when should the next frame be shown
                next_timestamp += frame["duration"] / self._playback_speed

            else:
                # if the video stream reached the end of file let the UI know
                self.endOfFile.emit()
                # and terminate the loop
                end_of_file = True
