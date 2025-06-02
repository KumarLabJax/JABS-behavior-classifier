from PySide6 import QtCore, QtGui, QtWidgets


class FrameWidget(QtWidgets.QLabel):
    """widget that implements a resizable pixmap label

    Used for displaying the current frame of the video. If necessary, the pixmap size is scaled down to fit the
    available area.
    """

    pixmap_clicked = QtCore.Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self.firstFrame = True
        self._scaled_pix_x = 0
        self._scaled_pix_y = 0
        self._scaled_pix_width = 0
        self._scaled_pix_height = 0
        self.setMinimumSize(400, 400)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Process mousePressEvent to emit a signal with the clicked pixel coordinates."""
        pix_x, pix_y = self._frame_xy_to_pixmap_xy(event.x(), event.y())
        self.pixmap_clicked.emit({"x": pix_x, "y": pix_y})

        super().mousePressEvent(event)

    def _frame_xy_to_pixmap_xy(self, x: int, y: int) -> tuple[int, int]:
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

    def sizeHint(self) -> QtCore.QSize:
        """Override QLabel.sizeHint to give an initial starting size."""
        return QtCore.QSize(1024, 1024)

    def reset(self) -> None:
        """reset state of frame widget"""
        self._scaled_pix_x = 0
        self._scaled_pix_y = 0
        self._scaled_pix_width = 0
        self._scaled_pix_height = 0
        self.clear()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
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

            # save the scaled pixmap dimensions for use by the mousePressEvent
            self._scaled_pix_x = point.x()
            self._scaled_pix_y = point.y()
            self._scaled_pix_width = pix.width()
            self._scaled_pix_height = pix.height()

        else:
            # if we don't have a pixmap to display just call the original QLabel
            # paintEvent
            super().paintEvent(event)
