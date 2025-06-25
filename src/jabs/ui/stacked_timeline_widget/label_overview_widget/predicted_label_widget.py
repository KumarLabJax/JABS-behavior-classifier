import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPaintEvent

from .label_overview_util import render_search_hits
from .manual_label_widget import ManualLabelWidget


class PredictedLabelWidget(ManualLabelWidget):
    """
    Widget for visualizing predicted behavior labels and their probabilities in a video timeline.

    Displays a horizontal bar where each frame is color-coded according to the predicted label,
    with transparency indicating the model's confidence (probability). Unlike ManualLabelWidget,
    this widget is read-only and does not support selection or manual label editing.

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._predictions: np.ndarray | None = None
        self._probabilities: np.ndarray | None = None

    def paintEvent(self, event: QPaintEvent) -> None:
        """handle the paint event to render the widget.

        Render the predicted label bar with color-coded frames and confidence transparency. Draws the
        timeline bar, including padding, background, predicted labels with probability-based
        transparency, and overlays position markers and ticks.

        Args:
            event (QPaintEvent): The paint event containing region to update.
        """
        # starting and ending frames of the current view
        # since the current frame is centered start might be negative and end might be > num_frames
        # out of bounds frames will be padded with a pattern
        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        # calculate the start and end of the slice to draw
        slice_start = max(start, 0)
        slice_end = min(end, self._num_frames - 1)

        start_padding = max(0, -start) * self._frame_width
        in_bounds_frames = slice_end - slice_start + 1
        in_bounds_width = in_bounds_frames * self._frame_width
        end_padding_frames = max(0, end - (self._num_frames - 1))
        end_padding_width = end_padding_frames * self._frame_width

        qp = QPainter(self)
        qp.setPen(Qt.PenStyle.NoPen)

        # Draw start padding
        if start_padding > 0:
            qp.setBrush(self._padding_brush)
            qp.drawRect(self._offset, 0, start_padding, self._bar_height)

        # Draw in-bounds white background
        qp.setBrush(Qt.GlobalColor.white)
        qp.drawRect(self._offset + start_padding, 0, in_bounds_width, self._bar_height)

        # Draw end padding
        if end_padding_width > 0:
            qp.setBrush(self._padding_brush)
            qp.drawRect(
                self._offset + start_padding + in_bounds_width,
                0,
                end_padding_width,
                self._bar_height,
            )

        # Draw predictions using color_lut
        # will be overlayed on top of the white background, lower probability will be more transparent
        if self._predictions is not None:
            # Convert predictions to color_lut indices (assume -1, 0, 1 corresponds to no prediction, not behavior, behavior)
            # add 1 to the predictions to convert to indices in color_lut
            color_indices = self._predictions[slice_start : slice_end + 1] + 1

            # Map to RGBA colors
            colors = self.COLOR_LUT[color_indices]

            # Set alpha from probabilities if available
            if self._probabilities is not None:
                alphas = (self._probabilities[slice_start : slice_end + 1] * 255).astype(np.uint8)
                colors[:, 3] = alphas

            # Expand to bar height: shape = (bar_height, frames in view, 4)
            colors_bar = np.repeat(colors[np.newaxis, :, :], self._bar_height, axis=0)

            # Expand each frame horizontally: shape = (bar_height, frames in view * frame pixel width, 4)
            colors_bar = np.repeat(colors_bar, self._frame_width, axis=1)

            # Draw the bar
            img = QImage(
                colors_bar.data,
                colors_bar.shape[1],
                colors_bar.shape[0],
                QImage.Format.Format_RGBA8888,
            )
            qp.drawImage(self._offset + start_padding, 0, img)

        render_search_hits(
            qp,
            self._search_results,
            self._offset,
            start,
            self._frame_width,
            self._bar_height,
            self._window_frames_total,
        )

        self._draw_position_marker(qp)
        self._draw_bounding_box(qp)
        self._draw_second_ticks(qp, start, end)

        # done drawing
        qp.end()

    def set_labels(self, predictions: np.ndarray, probabilities: np.ndarray) -> None:
        """Set the predicted labels and their probabilities for display.

        Args:
            predictions (np.ndarray): Array of predicted labels for each frame.
            probabilities (np.ndarray): Array of prediction probabilities for each frame.
        """
        self._predictions = predictions
        self._probabilities = probabilities
        self.update()

    def start_selection(self, start_frame: int) -> None:
        """Not supported in PredictedLabelWidget"""
        raise NotImplementedError

    def clear_selection(self) -> None:
        """Not supported in PredictedLabelWidget"""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the widget to its initial state.

        Clears any displayed predictions and probabilities, and calls the parent
        class's reset method to perform additional cleanup.
        """
        self._predictions = None
        self._probabilities = None
        super().reset()
