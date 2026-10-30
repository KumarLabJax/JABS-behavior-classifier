from typing import TYPE_CHECKING, Final

from PySide6 import QtCore, QtGui, QtWidgets
from qt_material_icons import MaterialIcon

from .overlay import Overlay

if TYPE_CHECKING:
    from ..frame_with_overlays import FrameWithOverlaysWidget


class _ImageAdjustment:
    """State for one image-adjustment control of a :class:`ControlOverlay`.

    Brightness and contrast are the same control: a badge carrying an icon that
    toggles a vertical slider, whose value is emitted as a multiplier for the
    parent widget to apply to the frame. Only the icon, the slider's range and
    the signal differ, so ControlOverlay keeps one of these per adjustment and
    drives both through the same draw/show/hide methods.

    Args:
        name: Identifier used for the slider container's Qt object name, which
            is what its stylesheet rule selects on. Must be unique within the
            overlay and usable in a stylesheet selector.
        icon_name: Material icon drawn on the badge.
        minimum: Smallest slider value, as a percentage.
        maximum: Largest slider value, as a percentage.
        signal: Overlay signal emitted with the new multiplier when the slider
            moves.

    Attributes:
        badge: Widget-coordinate rectangle the badge was last drawn at. Empty
            until the overlay has painted once, so a click cannot land in it
            before then.
        slider_open: Whether the badge is toggled on. Tracked separately from
            ``slider`` because the badge click handler toggles it before the
            slider widget exists.
        slider: The open slider, or ``None`` when it is hidden.
        container: Widget holding the slider and its reset button, or ``None``
            when the slider is hidden.
        value: Last slider value, kept so reopening the slider restores it
            rather than resetting to :attr:`NEUTRAL_VALUE`.
    """

    #: Slider value that leaves the frame unchanged, and what the reset button
    #: restores. The slider reports a percentage, so it is also the divisor that
    #: turns a slider value into the emitted multiplier.
    NEUTRAL_VALUE: Final[int] = 100

    def __init__(
        self,
        name: str,
        icon_name: str,
        minimum: int,
        maximum: int,
        signal: QtCore.SignalInstance,
    ) -> None:
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.signal = signal
        self.icon = MaterialIcon(icon_name).pixmap(16, color=QtGui.QColor(0, 0, 0))

        self.badge = QtCore.QRect()
        self.slider_open = False
        self.slider: QtWidgets.QSlider | None = None
        self.container: QtWidgets.QWidget | None = None
        self.value: int = self.NEUTRAL_VALUE


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

    * contrast adjustment:
        The same control as the brightness badge, for contrast. Both are
        :class:`_ImageAdjustment` instances and share one implementation.
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

        self._brightness = _ImageAdjustment(
            name="brightness",
            icon_name="brightness_6",
            minimum=50,
            maximum=200,
            signal=self.brightness_changed,
        )
        self._contrast = _ImageAdjustment(
            name="contrast",
            icon_name="contrast_circle",
            minimum=100,
            maximum=200,
            signal=self.contrast_changed,
        )
        # Drawn, hit-tested and closed in this order.
        self._adjustments: tuple[_ImageAdjustment, ...] = (self._brightness, self._contrast)

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
            self._draw_adjustment_badge(painter, x, y, self._brightness)

            x = (
                self.parent.scaled_pix_x
                + self.parent.scaled_pix_width
                - self._CONTRAST_BADGE_X_OFFSET
            )
            y = self.parent.scaled_pix_y + self.parent.scaled_pix_height - self._BADGE_Y_OFFSET
            self._draw_adjustment_badge(painter, x, y, self._contrast)

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

        for adjustment in self._adjustments:
            if adjustment.slider_open and adjustment.slider and not self._over_pixmap:
                self._hide_adjustment_slider(adjustment)

        self.parent.update()

    def handle_leave(self, event: QtCore.QEvent) -> None:
        """Handles leave events to hide the overlay when the mouse leaves the pixmap area.

        Args:
            event (QtCore.QEvent): The leave event.
        """
        self._over_pixmap = False
        for adjustment in self._adjustments:
            self._hide_adjustment_slider(adjustment)
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

        for adjustment in self._adjustments:
            # if the user clicks outside an adjustment's badge while its slider is open,
            # close the slider. don't return True, so that the parent widget can handle
            # the event if needed
            if adjustment.slider_open and not adjustment.badge.contains(point):
                self._hide_adjustment_slider(adjustment)

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

        for adjustment in self._adjustments:
            if adjustment.badge.contains(point):
                adjustment.slider_open = not adjustment.slider_open
                if adjustment.slider_open:
                    self._show_adjustment_slider(adjustment)
                else:
                    self._hide_adjustment_slider(adjustment)
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

    def _draw_adjustment_badge(
        self, painter: QtGui.QPainter, x: int, y: int, adjustment: _ImageAdjustment
    ) -> None:
        """Draw an image-adjustment badge as part of the control overlay.

        Args:
            painter (QtGui.QPainter): The painter used to draw the badge.
            x (int): Left edge of the badge, in widget coordinates.
            y (int): Bottom edge of the badge, in widget coordinates.
            adjustment (_ImageAdjustment): Adjustment to draw. Its
                :attr:`~_ImageAdjustment.badge` is updated to the rectangle drawn,
                which is what later clicks are hit-tested against.
        """
        # make the badge the same height as the playback speed badge, which will already have been created.
        w = self._ICON_BADGE_WIDTH
        h = self._playback_speed_badge.height()
        adjustment.badge = QtCore.QRect(x, y - h, w, h)

        painter.setBrush(self._BADGE_BACKGROUND_COLOR)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            adjustment.badge, self._BADGE_CORNER_RADIUS, self._BADGE_CORNER_RADIUS
        )

        icon_size = min(w, h) - self._BADGE_PADDING_VERTICAL
        icon_x = adjustment.badge.left() + (w - icon_size) // 2
        icon_y = adjustment.badge.top() + (h - icon_size) // 2
        painter.drawPixmap(icon_x, icon_y, icon_size, icon_size, adjustment.icon)

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

    def _show_adjustment_slider(self, adjustment: _ImageAdjustment) -> None:
        """Displays an adjustment's slider, anchored above its badge.

        Creates a container with a reset button above the slider. The slider is
        restored to the adjustment's last-used value rather than resetting to
        :attr:`_ImageAdjustment.NEUTRAL_VALUE`.

        Args:
            adjustment (_ImageAdjustment): Adjustment whose slider to show. Does
                nothing if its slider is already visible.
        """
        if adjustment.slider and adjustment.slider.isVisible():
            return

        container = QtWidgets.QWidget(self.parent)
        object_name = f"{adjustment.name}SliderContainer"
        container.setObjectName(object_name)
        container.setStyleSheet(
            f"#{object_name} {{ background-color: rgba(60, 60, 60, 180); border-radius: 8px; }}"
        )
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        reset_btn = QtWidgets.QPushButton(container)
        reset_btn.setIcon(
            MaterialIcon("restart_alt").pixmap(14, color=QtGui.QColor(255, 255, 255))
        )
        reset_btn.setFixedSize(20, 20)
        reset_btn.setFlat(True)
        layout.addWidget(reset_btn, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical, container)
        slider.setRange(adjustment.minimum, adjustment.maximum)
        slider.setValue(adjustment.value)
        slider.valueChanged.connect(
            lambda value: self._on_adjustment_slider_value_changed(adjustment, value)
        )
        layout.addWidget(slider)
        adjustment.slider = slider

        reset_btn.clicked.connect(lambda: slider.setValue(adjustment.NEUTRAL_VALUE))

        container.adjustSize()
        badge_top_left = self.parent.mapToGlobal(adjustment.badge.topLeft())
        pos = badge_top_left - QtCore.QPoint(0, container.sizeHint().height())
        container.move(self.parent.mapFromGlobal(pos))
        container.show()

        adjustment.container = container

    def _hide_adjustment_slider(self, adjustment: _ImageAdjustment) -> None:
        """Hides an adjustment's slider container.

        Args:
            adjustment (_ImageAdjustment): Adjustment whose slider to hide. Safe
                to call when the slider is already hidden.
        """
        if adjustment.container:
            adjustment.container.deleteLater()
            adjustment.container = None
        adjustment.slider = None
        adjustment.slider_open = False

    def _on_adjustment_slider_value_changed(
        self, adjustment: _ImageAdjustment, value: int
    ) -> None:
        """Handles an adjustment slider's value change event.

        Records the value so reopening the slider restores it, and emits the
        adjustment's signal with the value as a multiplier.

        Args:
            adjustment (_ImageAdjustment): Adjustment the slider belongs to.
            value (int): The new value from the slider, as a percentage.
        """
        adjustment.value = value
        adjustment.signal.emit(value / adjustment.NEUTRAL_VALUE)
