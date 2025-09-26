from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from qt_material_icons import MaterialIcon

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class ControlOverlay(Overlay):
    """Interactive overlay for controlling video playback.

    This adds interactive controls to the video frame when the mouse is over
    the pixmap area. The controls include:

    * playback speed adjustment:
        This badge shows the current playback speed. When clicked, it opens a menu
        that lets the user select a new playback speed from a predefined list.

    * video cropping:
        This badge allows the user to select a cropping area on the video frame.
        When clicked, it toggles the cropping mode. In cropping mode, the user can
        click and drag to select a rectangular area on the video frame. When the
        user releases the mouse button, the selected area is emitted as a signal
        to the parent widget, which can then apply the cropping to the video frame.

    * brightness adjustment:
        This badge shows a brightness icon. When clicked, it opens a vertical slider
        that allows the user to adjust the brightness of the video frame. The slider
        value is emitted as a signal to the parent widget, which can then apply the
        brightness adjustment to the video frame.
    """

    playback_speed_changed = QtCore.Signal(float)
    cropping_changed = QtCore.Signal(object, object)
    brightness_changed = QtCore.Signal(float)
    contrast_changed = QtCore.Signal(float)

    _PLAYBACK_SPEED_BADGE_OFFSET = 10
    _BADGE_PADDING_HORIZONTAL = 16
    _BADGE_PADDING_VERTICAL = 8
    _BADGE_CORNER_RADIUS = 8
    _BADGE_BACKGROUND_COLOR = QtGui.QColor(220, 220, 220, 230)
    _BADGE_BACKGROUND_COLOR_SELECTED = QtGui.QColor(120, 120, 120, 230)
    _BADGE_TEXT_COLOR = QtGui.QColor(0, 0, 0)
    _BADGE_FONT_SIZE = 12
    _CROPPING_PEN_WIDTH = 2
    _ICON_BADGE_WIDTH = 25
    _CROPPING_BADGE_X_OFFSET = 40
    _BRIGHTNESS_BADGE_X_OFFSET = _CROPPING_BADGE_X_OFFSET + _ICON_BADGE_WIDTH + 5
    _CONTRAST_BADGE_X_OFFSET = _BRIGHTNESS_BADGE_X_OFFSET + _ICON_BADGE_WIDTH + 5
    _BADGE_Y_OFFSET = 10

    def __init__(self, parent: "FrameWithOverlaysWidget"):
        super().__init__(parent)
        self._over_pixmap = False
        self._badge_font = QtGui.QFont()
        self._badge_font.setBold(True)
        self._badge_font.setPointSize(self._BADGE_FONT_SIZE)
        self._badge_font_metrics = QtGui.QFontMetrics(self._badge_font)

        self._playback_speed_menu_open = False
        self._playback_speed_badge = QtCore.QRect()
        self._playback_speed = 1.0  # Default playback speed
        self._speeds = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
        self._playback_speed_menu: QtWidgets.QMenu | None = None

        self._brightness_badge = QtCore.QRect()
        self._brightness_slider_open = False
        self._brightness_slider = None
        self._brightness_slider_bg = None
        self._brightness_icon = MaterialIcon("brightness_6").pixmap(
            16, color=QtGui.QColor(0, 0, 0)
        )

        self._contrast_badge = QtCore.QRect()
        self._contrast_slider_open = False
        self._contrast_slider = None
        self._contrast_slider_bg = None
        self._contrast_icon = MaterialIcon("contrast_circle").pixmap(
            16, color=QtGui.QColor(0, 0, 0)
        )

        self._cropping_badge = QtCore.QRect()
        self._select_mode = False
        self._select_start: QtCore.QPoint | None = None
        self._select_end: QtCore.QPoint | None = None
        self._crop_icon = MaterialIcon("crop").pixmap(16, color=QtGui.QColor(0, 0, 0))
        self._restore_icon = MaterialIcon("zoom_out_map").pixmap(16, color=QtGui.QColor(0, 0, 0))

    @property
    def playback_speed(self) -> float:
        """Returns the current playback speed."""
        return self._playback_speed

    def paint(self, painter: QtGui.QPainter, crop_rect: QtCore.QRect) -> None:
        """Paints the control overlay on the parent widget.

        Args:
            painter (QtGui.QPainter): The painter used to draw the overlay.
            crop_rect (QtCore.QRect): The rectangle defining the cropped area of the frame.

        Image coordinates will be translated into widget coordinates, taking into account that
        the image might be scaled and cropped. If the image coordinates are outside the crop_rect,
        then the overlay will not be drawn.
        """
        if not self._enabled or self.parent.pixmap().isNull():
            return

        # draw the control overlay.
        if self._over_pixmap or self._playback_speed_menu_open:
            x = self.parent.scaled_pix_x + self._PLAYBACK_SPEED_BADGE_OFFSET
            y = self.parent.scaled_pix_y + self.parent.scaled_pix_height - self._BADGE_Y_OFFSET
            self._draw_playback_speed_badge(painter, x, y)

            x = (
                self.parent.scaled_pix_x
                + self.parent.scaled_pix_width
                - self._CROPPING_BADGE_X_OFFSET
            )
            y = self.parent.scaled_pix_y + self.parent.scaled_pix_height - self._BADGE_Y_OFFSET
            self._draw_cropping_badge(painter, x, y)

            x = (
                self.parent.scaled_pix_x
                + self.parent.scaled_pix_width
                - self._BRIGHTNESS_BADGE_X_OFFSET
            )
            y = self.parent.scaled_pix_y + self.parent.scaled_pix_height - self._BADGE_Y_OFFSET
            self._draw_brightness_badge(painter, x, y)

            x = (
                self.parent.scaled_pix_x
                + self.parent.scaled_pix_width
                - self._CONTRAST_BADGE_X_OFFSET
            )
            y = self.parent.scaled_pix_y + self.parent.scaled_pix_height - self._BADGE_Y_OFFSET
            self._draw_contrast_badge(painter, x, y)

        # if user is actively selecting a crop area, draw the selection rectangle
        if self._select_start is not None and self._select_end is not None:
            x1, y1 = self._select_start.x(), self._select_start.y()
            x2, y2 = self._select_end.x(), self._select_end.y()
            rect = QtCore.QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            accent_color = self.parent.palette().color(QtGui.QPalette.ColorRole.Accent)
            painter.setPen(
                QtGui.QPen(accent_color, self._CROPPING_PEN_WIDTH, QtCore.Qt.PenStyle.DashLine)
            )
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

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

        if self._over_pixmap and self._select_mode:
            self.parent.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.parent.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        if self._over_pixmap and self._select_mode and self._select_start:
            self._select_end = QtCore.QPoint(x, y)

        if self._playback_speed_menu_open and self._playback_speed_menu and not self._over_pixmap:
            self._playback_speed_menu.close()

        if self._brightness_slider_open and self._brightness_slider and not self._over_pixmap:
            self._hide_brightness_slider()

        if self._contrast_slider_open and self._contrast_slider and not self._over_pixmap:
            self._hide_contrast_slider()

        self.parent.update()

    def handle_leave(self, event: QtCore.QEvent) -> None:
        """Handles leave events to hide the overlay when the mouse leaves the pixmap area.

        Args:
            event (QtCore.QEvent): The leave event.
        """
        self._over_pixmap = False
        self._hide_brightness_slider()
        self._hide_contrast_slider()
        self.parent.update()

    def handle_mouse_press(self, event: QtGui.QMouseEvent) -> bool:
        """Handles mouse press events on the parent widget.

        Check if the mouse press event is interacting with this overlay. If this overlay
        consumes the event, it will return True, indicating that the event was handled and
        the parent widget should not process it further.

        Args:
            event (QtGui.QMouseEvent): The mouse press event.

        Returns:
            bool: True if the event was handled, False otherwise.
        """
        if self.parent.pixmap() is None or self.parent.pixmap().isNull():
            return False

        x, y = event.x(), event.y()
        point = event.position().toPoint()
        self._over_pixmap = (
            self.parent.scaled_pix_x <= x < self.parent.scaled_pix_x + self.parent.scaled_pix_width
            and self.parent.scaled_pix_y
            <= y
            < self.parent.scaled_pix_y + self.parent.scaled_pix_height
        )

        if self._brightness_slider_open and not self._brightness_badge.contains(point):
            # if the user clicks outside the brightness badge while the slider is open, close the slider
            # don't return True, so that the parent widget can handle the event if needed
            self._hide_brightness_slider()

        if self._contrast_slider_open and not self._contrast_badge.contains(point):
            # if the user clicks outside the contrast badge while the slider is open, close the slider
            # don't return True, so that the parent widget can handle the event if needed
            self._hide_contrast_slider()

        if self._cropping_badge.contains(point):
            if self.parent.is_cropped:
                self.cropping_changed.emit(None, None)
            else:
                self._select_mode = not self._select_mode
                self._select_start = None
                self._select_end = None
            self.parent.update()
            return True

        # next cropping selection will take precedence over clicking other controls.
        if self._select_mode and self._over_pixmap:
            self._select_start = point
            self._select_end = point
            self.parent.update()
            return True

        if self._playback_speed_badge.contains(point) or self._playback_speed_menu_open:
            self._playback_speed_menu_open = False
            self._show_speed_menu()
            return True

        if self._brightness_badge.contains(point):
            self._brightness_slider_open = not self._brightness_slider_open
            if self._brightness_slider_open:
                self._show_brightness_slider()
            else:
                self._hide_brightness_slider()
            return True

        if self._contrast_badge.contains(point):
            self._contrast_slider_open = not self._contrast_slider_open
            if self._contrast_slider_open:
                self._show_contrast_slider()
            else:
                self._hide_contrast_slider()
            return True

        return False

    def handle_mouse_release(self, event: QtGui.QMouseEvent) -> None:
        """Handles mouse release events on parent widget for this overlay."""
        if self._select_mode and self._select_start:
            x1, y1 = self._select_start.x(), self._select_start.y()
            x2, y2 = self._select_end.x(), self._select_end.y()

            # Convert widget coordinates to image coordinates.
            # also sort the points so img_p1 is always top-left and img_p2 is always bottom-right
            img_p1 = self.parent.widget_to_image_coords(min(x1, x2), min(y1, y2))
            img_p2 = self.parent.widget_to_image_coords(max(x1, x2), max(y1, y2))
            if img_p1 and img_p2:
                self.cropping_changed.emit(
                    QtCore.QPoint(img_p1[0], img_p1[1]), QtCore.QPoint(img_p2[0], img_p2[1])
                )
            self._select_mode = False
            self._select_start = None
            self._select_end = None
            self.parent.update()

    def event_filter(
        self,
        obj: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        """Implements overlay-specific event filtering logic for parent widget.

        Currently, it filters mouse move events and closes the menu if the mouse moves outside the pixmap area.

        Args:
            obj (QtCore.QObject): The object that is the target of the event.
            event (QtCore.QEvent): The event to be filtered.

        Returns:
            bool: True if the event is handled and should be filtered out, False otherwise.
        """
        if (
            obj is self._playback_speed_menu
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
                self._playback_speed_menu.close()
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
        self._playback_speed_badge = QtCore.QRect(x, y - h, w, h)

        # paint badge background and text
        font = painter.font()
        painter.setFont(self._badge_font)
        painter.setBrush(self._BADGE_BACKGROUND_COLOR)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            self._playback_speed_badge, self._BADGE_CORNER_RADIUS, self._BADGE_CORNER_RADIUS
        )
        painter.setPen(self._BADGE_TEXT_COLOR)
        painter.drawText(
            self._playback_speed_badge, QtCore.Qt.AlignmentFlag.AlignCenter, badge_text
        )

        # restore the original font
        painter.setFont(font)

    def _draw_cropping_badge(self, painter: QtGui.QPainter, x: int, y: int) -> None:
        """Draw a cropping badge as part of the control overlay."""
        # make the badge the same height as the playback speed badge, which will already have been created.
        w = self._ICON_BADGE_WIDTH
        h = self._playback_speed_badge.height()
        self._cropping_badge = QtCore.QRect(x, y - h, w, h)

        painter.setBrush(
            self._BADGE_BACKGROUND_COLOR
            if not self._select_mode
            else self._BADGE_BACKGROUND_COLOR_SELECTED
        )
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            self._cropping_badge, self._BADGE_CORNER_RADIUS, self._BADGE_CORNER_RADIUS
        )

        icon = self._restore_icon if self.parent.is_cropped else self._crop_icon
        icon_size = min(w, h) - self._BADGE_PADDING_VERTICAL
        icon_x = self._cropping_badge.left() + (w - icon_size) // 2
        icon_y = self._cropping_badge.top() + (h - icon_size) // 2
        painter.drawPixmap(icon_x, icon_y, icon_size, icon_size, icon)

    def _draw_brightness_badge(self, painter: QtGui.QPainter, x: int, y: int) -> None:
        """Draw a brightness badge as part of the control overlay."""
        # make the badge the same height as the playback speed badge, which will already have been created.
        w = self._ICON_BADGE_WIDTH
        h = self._playback_speed_badge.height()
        self._brightness_badge = QtCore.QRect(x, y - h, w, h)

        painter.setBrush(self._BADGE_BACKGROUND_COLOR)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            self._brightness_badge, self._BADGE_CORNER_RADIUS, self._BADGE_CORNER_RADIUS
        )

        icon_size = min(w, h) - self._BADGE_PADDING_VERTICAL
        icon_x = self._brightness_badge.left() + (w - icon_size) // 2
        icon_y = self._brightness_badge.top() + (h - icon_size) // 2
        painter.drawPixmap(icon_x, icon_y, icon_size, icon_size, self._brightness_icon)

    def _draw_contrast_badge(self, painter: QtGui.QPainter, x: int, y: int) -> None:
        """Draw a contrast badge as part of the control overlay."""
        # make the badge the same height as the playback speed badge, which will already have been created.
        w = self._ICON_BADGE_WIDTH
        h = self._playback_speed_badge.height()
        self._contrast_badge = QtCore.QRect(x, y - h, w, h)

        painter.setBrush(self._BADGE_BACKGROUND_COLOR)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            self._contrast_badge, self._BADGE_CORNER_RADIUS, self._BADGE_CORNER_RADIUS
        )

        icon_size = min(w, h) - self._BADGE_PADDING_VERTICAL
        icon_x = self._contrast_badge.left() + (w - icon_size) // 2
        icon_y = self._contrast_badge.top() + (h - icon_size) // 2
        painter.drawPixmap(icon_x, icon_y, icon_size, icon_size, self._contrast_icon)

    def _show_speed_menu(self) -> None:
        """Displays the playback speed selection menu anchored to the playback speed badge.

        Creates and shows a popup menu with available playback speeds. The menu is positioned
        above the badge and highlights the current speed. Handles menu actions and cleanup
        when the menu is closed.
        """
        if self._playback_speed_menu_open:
            # If the menu is already open, do nothing
            return

        self.parent.update()

        # create the menu
        self._playback_speed_menu = QtWidgets.QMenu(self.parent)
        for speed in self._speeds:
            action = self._playback_speed_menu.addAction(f"{speed}x")
            action.setData(speed)
            action.setCheckable(True)
            if speed == self._playback_speed:
                action.setChecked(True)
            action.triggered.connect(
                lambda checked, a=action: self._on_playback_speed_menu_triggered(a)
            )
        self._playback_speed_menu.aboutToHide.connect(
            lambda: self._on_playback_speed_menu_closed()
        )
        self._playback_speed_menu.installEventFilter(self.parent)

        # position the menu above the badge
        badge_top_left = self.parent.mapToGlobal(self._playback_speed_badge.topLeft())
        menu_size = self._playback_speed_menu.sizeHint()
        pos = badge_top_left - QtCore.QPoint(0, menu_size.height())

        # show the menu
        self._playback_speed_menu.popup(pos)
        self._playback_speed_menu_open = True

    def _on_playback_speed_menu_triggered(self, action: QtGui.QAction) -> None:
        """Handles the selection of a playback speed from the menu.

        Updates the playback speed if a new speed is selected and emits the playback_speed_changed signal.

        Args:
            action (QtGui.QAction): The action representing the selected playback speed.
        """
        new_speed = action.data()
        if new_speed != self._playback_speed:
            self._playback_speed = new_speed
            self.playback_speed_changed.emit(self._playback_speed)

    def _on_playback_speed_menu_closed(self) -> None:
        """Handles cleanup when the playback speed menu is closed.

        Disconnects signals, removes event filters, deletes the menu, and updates the overlay state.
        Creates a synthetic mouseMoveEvent on the frame widget to keep the overlay visible
        after the menu closes. This is necessary because the menu takes mouse focus
        triggering an exitEvent on the widget, which would otherwise hide the overlay. The badge shouldn't
        get hidden until the mouse moves away from the pixmap area.

        """
        self._playback_speed_menu_open = False
        if self._playback_speed_menu:
            self._playback_speed_menu.aboutToHide.disconnect()
            self._playback_speed_menu.removeEventFilter(self.parent)
            self._playback_speed_menu.deleteLater()
            self._playback_speed_menu = None

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

    def _show_brightness_slider(self) -> None:
        """Displays a brightness slider for adjusting the video brightness.

        This method creates a slider widget and positions it above the brightness badge.
        The slider allows the user to adjust the brightness of the video frame.
        """
        if self._brightness_slider and self._brightness_slider.isVisible():
            return

        # Create a translucent gray widget as the background for the slider
        # this helps the slider stand out against the video
        # it will not be clickable
        self._brightness_slider_bg = QtWidgets.QWidget(self.parent)
        self._brightness_slider_bg.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._brightness_slider_bg.setStyleSheet(
            "background-color: rgba(60, 60, 60, 180); border-radius: 8px;"
        )

        # Create a slider widget
        self._brightness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical, self.parent)
        self._brightness_slider.setRange(50, 200)
        self._brightness_slider.valueChanged.connect(self._on_brightness_slider_value_changed)
        self._brightness_slider.setValue(100)

        # Position both widgets above the brightness badge
        badge_top_left = self.parent.mapToGlobal(self._brightness_badge.topLeft())
        slider_size = self._brightness_slider.sizeHint()
        pos = badge_top_left - QtCore.QPoint(0, slider_size.height())

        # Set geometry for background and slider
        self._brightness_slider_bg.setGeometry(
            self.parent.mapFromGlobal(pos).x(),
            self.parent.mapFromGlobal(pos).y(),
            slider_size.width(),
            slider_size.height(),
        )
        self._brightness_slider.move(self.parent.mapFromGlobal(pos))

        self._brightness_slider_bg.show()
        self._brightness_slider.show()

    def _hide_brightness_slider(self) -> None:
        """Hides the brightness slider and its background widget."""
        if self._brightness_slider:
            self._brightness_slider.deleteLater()
            self._brightness_slider = None
        if self._brightness_slider_bg:
            self._brightness_slider_bg.deleteLater()
            self._brightness_slider_bg = None
        self._brightness_slider_open = False

    def _on_brightness_slider_value_changed(self, value: int) -> None:
        """Handles the brightness slider value change event.

        Emits the brightness_changed signal with the new brightness value.

        Args:
            value (int): The new brightness value from the slider.
        """
        self.brightness_changed.emit(value / 100.0)

    def _show_contrast_slider(self) -> None:
        """Displays a contrast slider for adjusting the video contrast.

        This method creates a slider widget and positions it above the contrast badge.
        The slider allows the user to adjust the contrast of the video frame.
        """
        if self._contrast_slider and self._contrast_slider.isVisible():
            return

        # Create a translucent gray widget as the background for the slider
        # this helps the slider stand out against the video
        # it will not be clickable
        self._contrast_slider_bg = QtWidgets.QWidget(self.parent)
        self._contrast_slider_bg.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._contrast_slider_bg.setStyleSheet(
            "background-color: rgba(60, 60, 60, 180); border-radius: 8px;"
        )

        # Create a slider widget
        self._contrast_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical, self.parent)
        self._contrast_slider.setRange(100, 200)
        self._contrast_slider.valueChanged.connect(self._on_contrast_slider_value_changed)
        self._contrast_slider.setValue(100)
        self.contrast_changed.emit(1.0)

        # Position both widgets above the brightness badge
        badge_top_left = self.parent.mapToGlobal(self._contrast_badge.topLeft())
        slider_size = self._contrast_slider.sizeHint()
        pos = badge_top_left - QtCore.QPoint(0, slider_size.height())

        # Set geometry for background and slider
        self._contrast_slider_bg.setGeometry(
            self.parent.mapFromGlobal(pos).x(),
            self.parent.mapFromGlobal(pos).y(),
            slider_size.width(),
            slider_size.height(),
        )
        self._contrast_slider.move(self.parent.mapFromGlobal(pos))

        self._contrast_slider_bg.show()
        self._contrast_slider.show()

    def _hide_contrast_slider(self) -> None:
        """Hides the contrast slider and its background widget."""
        if self._contrast_slider:
            self._contrast_slider.deleteLater()
            self._contrast_slider = None
        if self._contrast_slider_bg:
            self._contrast_slider_bg.deleteLater()
            self._contrast_slider_bg = None
        self._contrast_slider_open = False

    def _on_contrast_slider_value_changed(self, value: int) -> None:
        """Handles the contrast slider value change event.

        Emits the contrast_changed signal with the new contrast value.

        Args:
            value (int): The new contrast value from the slider.
        """
        self.contrast_changed.emit(value / 100.0)
