import math

import numpy as np
import numpy.typing as npt
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
        self._predictions: npt.NDArray[np.int16] | None = None
        self._probabilities: npt.NDArray[np.floating] | None = None

    def paintEvent(self, event: QPaintEvent) -> None:
        """handle the paint event to render the widget.

        Render the predicted label bar with color-coded frames and confidence transparency. Draws the
        timeline bar, including padding, background, predicted labels with probability-based
        transparency, and overlays position markers and ticks.

        Args:
            event (QPaintEvent): The paint event containing region to update.
        """
        widget_width = self.size().width()
        if widget_width == 0 or self._window_frames_total == 0:
            return
        self._frame_width = widget_width / self._window_frames_total
        fw = self._frame_width

        start = self._current_frame - self._window_size
        end = self._current_frame + self._window_size

        slice_start = max(start, 0)
        slice_end = min(end, self._num_frames - 1)
        n_in_bounds = slice_end - slice_start + 1
        n_start_pad = max(0, -start)

        in_bounds_x0 = math.floor(n_start_pad * fw)
        in_bounds_x1 = math.floor((n_start_pad + n_in_bounds) * fw)
        in_bounds_px = in_bounds_x1 - in_bounds_x0

        qp = QPainter(self)
        qp.setPen(Qt.PenStyle.NoPen)

        # Fill entire bar with padding pattern, then overdraw in-bounds region
        qp.setBrush(self._padding_brush)
        qp.drawRect(0, 0, widget_width, self._bar_height)

        if n_in_bounds > 0 and in_bounds_px > 0:
            # White background for the in-bounds region; predictions are overlaid with alpha
            qp.setBrush(Qt.GlobalColor.white)
            qp.drawRect(in_bounds_x0, 0, in_bounds_px, self._bar_height)

        # Draw predictions overlaid on the white background; lower probability = more transparent.
        if self._predictions is not None and n_in_bounds > 0 and in_bounds_px > 0:
            color_indices = self._predictions[slice_start : slice_end + 1]

            # Map to RGBA colors
            colors = self._color_lut[color_indices]

            # Set alpha from probabilities if available
            if self._probabilities is not None:
                probs = np.clip(self._probabilities[slice_start : slice_end + 1], 0.0, 1.0)
                alphas = (probs * 255).astype(np.uint8)

                # some post-processed predictions may have zero probability, specifically the interpolation stage
                # which fills in short gaps where there was no prediction. To ensure these interpolated classes
                # are still visible, set a minimum alpha anywhere there is a prediction but probability is zero
                # to ensure visibility. Interpolated predictions will show as having a low confidence.
                mask = (color_indices != 0) & (alphas == 0)
                alphas[mask] = 125

                colors[:, 3] = alphas

            # Per-frame pixel widths using floating-point positions
            win_indices = np.arange(n_start_pad, n_start_pad + n_in_bounds + 1, dtype=np.float64)
            frame_px_widths = np.diff(np.floor(win_indices * fw)).astype(int)

            # Expand to bar height and per-frame pixel widths
            colors_bar = np.repeat(colors[np.newaxis, :, :], self._bar_height, axis=0)
            colors_bar = np.repeat(colors_bar, frame_px_widths, axis=1)

            img = QImage(
                colors_bar.data,
                colors_bar.shape[1],
                colors_bar.shape[0],
                QImage.Format.Format_RGBA8888,
            )
            qp.drawImage(in_bounds_x0, 0, img)

        render_search_hits(
            qp,
            self._search_results,
            0,
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

    def set_labels(
        self,
        predictions: npt.NDArray[np.int16] | None,
        probabilities: npt.NDArray[np.floating] | None,
    ) -> None:
        """Set the predicted labels and their probabilities for display.

        Args:
            predictions: Class-index array of shape ``(n_frames,)`` with dtype ``int16``, or ``None``.
            probabilities: Per-frame prediction confidence of shape ``(n_frames,)``, or ``None``.
        """
        if predictions is None:
            if probabilities is not None:
                raise ValueError("probabilities must be None when predictions is None")
            self._predictions = None
            self._probabilities = None
            self.update()
            return

        if predictions.ndim != 1:
            raise ValueError("predictions must be a 1D array")
        if self._num_frames and predictions.shape[0] != self._num_frames:
            raise ValueError(
                "predictions length must match num_frames: "
                f"{predictions.shape[0]} != {self._num_frames}"
            )
        if np.any(predictions < 0) or np.any(predictions >= len(self._color_lut)):
            raise ValueError("predictions contain indices outside the active color LUT range")

        if probabilities is not None:
            if probabilities.ndim != 1 or probabilities.shape[0] != predictions.shape[0]:
                raise ValueError(
                    "probabilities must be a 1D array with same length as predictions"
                )
            self._probabilities = np.clip(probabilities, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            self._probabilities = None

        self._predictions = predictions
        self.update()

    def start_selection(self, start_frame: int, end_frame: int | None = None) -> None:
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
