from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_control_overlay import FrameWidgetWithInteractiveOverlays


class ControlOverlay(Overlay):
    """Interactive overlay for controlling video playback.

    Supports playback speed adjustment via a badge displaying current playback speed
    that is drawn when the mouse is over the video frame, with popup menu that is opened when the badge is clicked.
    """

    playback_speed_changed = QtCore.Signal(float)

    _BADGE_OFFSET = 10
    _BADGE_PADDING_HORIZONTAL = 16
    _BADGE_PADDING_VERTICAL = 8
    _BADGE_CORNER_RADIUS = 8
    _BADGE_BACKGROUND_COLOR = QtGui.QColor(220, 220, 220, 230)
    _BADGE_TEXT_COLOR = QtGui.QColor(0, 0, 0)
    _BADGE_FONT_SIZE = 12

    def __init__(self, parent: "FrameWidgetWithInteractiveOverlays"):
        super().__init__(parent)
        self._over_pixmap = False
        self._menu_open = False
        self._badge_rect = QtCore.QRect()
        self._playback_speed = 1.0
        self._speeds = [0.5, 1.0, 1.5, 2.0, 4.0]
        self._menu = None
        self._badge_font = QtGui.QFont()
        self._badge_font.setBold(True)
        self._badge_font.setPointSize(self._BADGE_FONT_SIZE)
        self._badge_font_metrics = QtGui.QFontMetrics(self._badge_font)

    @property
    def playback_speed(self) -> float:
        """Returns the current playback speed."""
        return self._playback_speed

    def paint(self, painter: QtGui.QPainter) -> None:
        """Paints the control overlay on the parent widget.

        Args:
            painter (QtGui.QPainter): The painter used to draw the overlay.
        """
        if not self._enabled or self.parent.pixmap().isNull():
            return

        if self._over_pixmap or self._menu_open:
            x = self.parent.scaled_pix_x + self._BADGE_OFFSET
            y = self.parent.scaled_pix_y + self.parent.scaled_pix_height - self._BADGE_OFFSET
            self._draw_playback_speed_badge(painter, x, y)

    def handle_mouse_move(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse move events to determine if the mouse is over the pixmap area and update overlay state accordingly.

        Args:
            event (QtGui.QMouseEvent): The mouse move event.
        """
        x, y = event.x(), event.y()
        self._over_pixmap = (
            self.parent.scaled_pix_x <= x < self.parent.scaled_pix_x + self.parent.scaled_pix_width
            and self.parent.scaled_pix_y
            <= y
            < self.parent.scaled_pix_y + self.parent.scaled_pix_height
        )
        if self._menu_open and self._menu and not self._over_pixmap:
            self._menu.close()
        self.parent.update()

    def handle_leave(self, event: QtCore.QEvent) -> None:
        """Handles leave events to hide the overlay when the mouse leaves the pixmap area.

        Args:
            event (QtCore.QEvent): The leave event.
        """
        self._over_pixmap = False
        self.parent.update()

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> bool:
        """Handles mouse press events on the parent widget.

        Check if the mouse press event is interacting with this overlay. If click is
        on the playback speed badge update the overlay state to show the speed menu.

        Args:
            event (QtGui.QMouseEvent): The mouse press event.

        Returns:
            bool: True if the event was handled and the menu was shown, False otherwise.
        """
        x, y = event.x(), event.y()
        self._over_pixmap = (
            self.parent.scaled_pix_x <= x < self.parent.scaled_pix_x + self.parent.scaled_pix_width
            and self.parent.scaled_pix_y
            <= y
            < self.parent.scaled_pix_y + self.parent.scaled_pix_height
        )
        if (
            self.parent.pixmap() is not None
            and not self.parent.pixmap().isNull()
            and (self._badge_rect.contains(event.pos()) or self._menu_open)
        ):
            self._menu_open = False
            self._show_speed_menu()
            return True
        return False

    def event_filter(
        self,
        obj: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        """
        Implements overlay-specific event filtering logic for parent widget.

        Currently, it filters mouse move events and closes the menu if the mouse moves outside the pixmap area.

        Args:
            obj (QtCore.QObject): The object that is the target of the event.
            event (QtCore.QEvent): The event to be filtered.

        Returns:
            bool: True if the event is handled and should be filtered out, False otherwise.
        """
        if (
            obj is self._menu
            and isinstance(event, QtGui.QMouseEvent)
            and event.type() == QtCore.QEvent.Type.MouseMove
        ):
            # are we over the pixmap area?
            global_pos = event.globalPos()
            local_pos = self.parent.mapFromGlobal(global_pos)
            over_pixmap = (
                self.parent.scaled_pix_x
                <= local_pos.x()
                < self.parent.scaled_pix_x + self.parent.scaled_pix_width
                and self.parent.scaled_pix_y
                <= local_pos.y()
                < self.parent.scaled_pix_y + self.parent.scaled_pix_height
            )

            # close the menu if the mouse is not over the pixmap area
            if not over_pixmap:
                self._menu.close()
        return False

    def _draw_playback_speed_badge(self, painter: QtGui.QPainter, x: int, y: int) -> None:
        """Draws the playback speed badge at the specified position on the parent widget.

        Args:
            painter (QtGui.QPainter): The painter used to draw the badge.
            x (int): The x-coordinate for the top-left corner of the badge.
            y (int): The y-coordinate for the top-left corner of the badge.
        """
        badge_text = f"{self._playback_speed}x"

        # create the badge, which is a QRect with rounded corners and a size determined by the text size
        w = self._badge_font_metrics.horizontalAdvance(badge_text) + self._BADGE_PADDING_HORIZONTAL
        h = self._badge_font_metrics.height() + self._BADGE_PADDING_VERTICAL
        self._badge_rect = QtCore.QRect(x, y - h, w, h)

        # paint badge background and text
        font = painter.font()
        painter.setFont(self._badge_font)
        painter.setBrush(self._BADGE_BACKGROUND_COLOR)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            self._badge_rect, self._BADGE_CORNER_RADIUS, self._BADGE_CORNER_RADIUS
        )
        painter.setPen(self._BADGE_TEXT_COLOR)
        painter.drawText(self._badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, badge_text)

        # restore the original font
        painter.setFont(font)

    def _show_speed_menu(self) -> None:
        """Displays the playback speed selection menu anchored to the playback speed badge.

        Creates and shows a popup menu with available playback speeds. The menu is positioned
        above the badge and highlights the current speed. Handles menu actions and cleanup
        when the menu is closed.

        """
        if self._menu_open:
            # If the menu is already open, do nothing
            return

        self.parent.update()

        # create the menu
        self._menu = QtWidgets.QMenu(self.parent)
        for speed in self._speeds:
            action = self._menu.addAction(f"{speed}x")
            action.setData(speed)
            action.setCheckable(True)
            if speed == self._playback_speed:
                action.setChecked(True)
            action.triggered.connect(lambda checked, a=action: self._on_menu_triggered(a))
        self._menu.aboutToHide.connect(lambda: self._on_menu_closed())
        self._menu.installEventFilter(self.parent)

        # position the menu above the badge
        badge_top_left = self.parent.mapToGlobal(self._badge_rect.topLeft())
        menu_size = self._menu.sizeHint()
        pos = badge_top_left - QtCore.QPoint(0, menu_size.height())

        # show the menu
        self._menu.popup(pos)
        self._menu_open = True

    def _on_menu_triggered(self, action: QtGui.QAction) -> None:
        """Handles the selection of a playback speed from the menu.

        Updates the playback speed if a new speed is selected and emits the playback_speed_changed signal.

        Args:
            action (QtGui.QAction): The action representing the selected playback speed.
        """
        new_speed = action.data()
        if new_speed != self._playback_speed:
            self._playback_speed = new_speed
            self.playback_speed_changed.emit(self._playback_speed)

    def _on_menu_closed(self) -> None:
        """Handles cleanup when the playback speed menu is closed.

        Disconnects signals, removes event filters, deletes the menu, and updates the overlay state.
        Creates a synthetic mouseMoveEvent on the frame widget to keep the overlay visible
        after the menu closes. This is necessary because the menu takes mouse focus
        triggering an exitEvent on the widget, which would otherwise hide the overlay. The badge shouldn't
        get hidden until the mouse moves away from the pixmap area.

        Args:
            parent (FrameWidgetWithInteractiveOverlays): The parent widget containing the overlay.
        """
        self._menu_open = False
        if self._menu:
            self._menu.aboutToHide.disconnect()
            self._menu.removeEventFilter(self.parent)
            self._menu.deleteLater()
            self._menu = None

        # Synthesize a mouse move event to update _over_pixmap
        cursor_pos = self.parent.mapFromGlobal(QtGui.QCursor.pos())
        mouse_event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            cursor_pos,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.parent.mouseMoveEvent(mouse_event)
        self.parent.update()
