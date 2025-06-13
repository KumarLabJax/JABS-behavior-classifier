from PySide6 import QtCore, QtGui, QtWidgets

from .frame_widget import FrameWidget


class FrameWidgetWithOverlay(FrameWidget):
    """
    A `FrameWidget` subclass that adds an interactive overlay for playback controls.

    This widget displays a video control overlay on top of the video frame
    when the mouse is over the image.

    Currently, the overlay has one control: playback speed adjustment.
    This control consists of a badge that shows the current playback speed
    and allows the user to change the speed via a popup menu. The overlay
    and menu are only visible when the mouse is over the displayed pixmap area.

    Signals:
        playback_speed_changed (float): Emitted when the playback speed is changed by the user.
    """

    playback_speed_changed = QtCore.Signal(float)

    _BADGE_OFFSET = 10
    _BADGE_PADDING_HORIZONTAL = 16
    _BADGE_PADDING_VERTICAL = 8
    _BADGE_CORNER_RADIUS = 8
    _BADGE_BACKGROUND_COLOR = QtGui.QColor(0, 0, 0, 180)
    _BADGE_TEXT_COLOR = QtGui.QColor(255, 255, 255)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._over_pixmap = False
        self._menu_open = False
        self._badge_rect = QtCore.QRect()
        self._playback_speed = 1.0
        self._speeds = [0.5, 1.0, 2.0, 4.0]
        self._menu: QtWidgets.QMenu | None = None
        self._font = QtGui.QFont()
        self._font.setBold(True)
        self._font.setPointSize(12)
        self._font_metrics = QtGui.QFontMetrics(self._font)

    @property
    def playback_speed(self) -> float:
        """Get the current playback speed."""
        return self._playback_speed

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Handle leave events to hide the overlay and close the menu if it is open.

        This method is called when the mouse leaves the widget area.

        Note: we can't rely on the mouseMoveEvent alone to hide the overlay if the pixmap is
        the same size as the widget, because we never get a mouseMoveEvent with coordinates
        outside the pixmap area.
        """
        self._over_pixmap = False
        self.update()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move events to update the overlay state and menu visibility.

        Moving the mouse over the pixmap area will show the playback speed badge.
        Moving the mouse away from the pixmap area will hide the badge and close the menu if it is open.

        Note: we don't rely on the enterEvent and leaveEvent methods to control the overlay visibility
        because the widget might be larger than the pixmap, and we want to show the overlay only
        when the mouse is over the actual pixmap area.
        """
        x, y = event.x(), event.y()
        self._over_pixmap = (
            self._scaled_pix_x <= x < self._scaled_pix_x + self._scaled_pix_width
            and self._scaled_pix_y <= y < self._scaled_pix_y + self._scaled_pix_height
        )
        # Hide menu if open and mouse leaves pixmap
        if self._menu_open and self._menu and not self._over_pixmap:
            self._menu.close()
        self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the frame widget with an overlay for playback speed badge."""
        # draw the FrameWidget
        super().paintEvent(event)

        # now handle the video control overlay
        pixmap = self.pixmap()
        if pixmap is not None and not pixmap.isNull() and (self._over_pixmap or self._menu_open):
            # currently the only overlay is the playback speed badge
            self._draw_badge()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press events to show the playback speed menu when the badge is clicked.

        If click is outside the badge area, fall back to the default FrameWidget behavior.
        """
        # Update _over_pixmap based on actual mouse position
        x, y = event.x(), event.y()
        self._over_pixmap = (
            self._scaled_pix_x <= x < self._scaled_pix_x + self._scaled_pix_width
            and self._scaled_pix_y <= y < self._scaled_pix_y + self._scaled_pix_height
        )
        if (
            self.pixmap() is not None
            and not self.pixmap().isNull()
            and (self._badge_rect.contains(event.pos()) or self._menu_open)
        ):
            # Always reset _menu_open before trying to open the menu
            self._menu_open = False
            self._show_speed_menu()
        else:
            super().mousePressEvent(event)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Filter events for the menu to handle mouse movements and close it when necessary.

        This is necessary to close the menu when the mouse moves away from the badge area.

        Args:
            obj: The object the event is being sent to.
            event: The event being processed.

        Returns:
            bool: True if the event was handled, False otherwise.
        """
        if obj is self._menu and event.type() == QtCore.QEvent.Type.MouseMove:
            global_pos = event.globalPos()
            local_pos = self.mapFromGlobal(global_pos)

            # are we over the pixmap area? If not, close the menu.
            over_pixmap = (
                self._scaled_pix_x <= local_pos.x() < self._scaled_pix_x + self._scaled_pix_width
                and self._scaled_pix_y
                <= local_pos.y()
                < self._scaled_pix_y + self._scaled_pix_height
            )
            if not over_pixmap:
                self._menu.close()
        return super().eventFilter(obj, event)

    def _draw_badge(self) -> None:
        """Draw the playback speed badge."""
        badge_text = f"{self._playback_speed}x"

        # figure out the size and position of the badge
        w, h = (
            self._font_metrics.horizontalAdvance(badge_text) + self._BADGE_PADDING_HORIZONTAL,
            self._font_metrics.height() + self._BADGE_PADDING_VERTICAL,
        )
        x = self._scaled_pix_x + self._BADGE_OFFSET
        y = self._scaled_pix_y + self._scaled_pix_height - h - self._BADGE_OFFSET
        self._badge_rect = QtCore.QRect(x, y, w, h)

        # draw playback speed badge
        painter = QtGui.QPainter(self)
        painter.setFont(self._font)
        painter.setBrush(self._BADGE_BACKGROUND_COLOR)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            self._badge_rect, self._BADGE_CORNER_RADIUS, self._BADGE_CORNER_RADIUS
        )
        painter.setPen(self._BADGE_TEXT_COLOR)
        painter.drawText(self._badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, badge_text)
        painter.end()

    def _show_speed_menu(self) -> None:
        """Show the playback speed menu when the playback speed badge is clicked."""
        # menu is already open, do nothing
        if self._menu_open:
            return

        self.update()

        # Create the menu with playback speed options
        self._menu = QtWidgets.QMenu(self)
        for speed in self._speeds:
            action = self._menu.addAction(f"{speed}x")
            action.setData(speed)
            action.setCheckable(True)
            if speed == self._playback_speed:
                action.setChecked(True)
            action.triggered.connect(lambda checked, a=action: self._on_menu_triggered(a))

        # Connect the menu actions and event filter
        self._menu.aboutToHide.connect(self._on_menu_closed)
        self._menu.installEventFilter(self)

        # Calculate the position to show the menu above the badge
        badge_top_left = self.mapToGlobal(self._badge_rect.topLeft())
        menu_size = self._menu.sizeHint()
        pos = badge_top_left - QtCore.QPoint(0, menu_size.height())

        # show menu
        self._menu.popup(pos)
        self._menu_open = True

    def _on_menu_triggered(self, action: QtGui.QAction) -> None:
        """Handle the menu action triggered event."""
        new_speed = action.data()
        if new_speed != self._playback_speed:
            self._playback_speed = new_speed
            self.playback_speed_changed.emit(self._playback_speed)

    def _on_menu_closed(self) -> None:
        """Handle the menu closed event.

        Creates a fake mouseMoveEvent on the widget to keep the overlay visible
        after the menu closes. This is necessary because the menu takes mouse focus
        triggering an exitEvent on the widget, which would otherwise hide the overlay.
        """
        self._menu_open = False
        if self._menu:
            self._menu.aboutToHide.disconnect(self._on_menu_closed)
            self._menu.removeEventFilter(self)
            self._menu.deleteLater()
            self._menu = None

        # Synthesize a mouse move event to update _over_pixmap
        cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        mouse_event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            cursor_pos,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.mouseMoveEvent(mouse_event)
        self.update()
