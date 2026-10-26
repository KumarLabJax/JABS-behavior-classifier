import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPaintEvent

from .label_detail_bar import LabelDetailBar
from .timeline_util import draw_frame_colors, render_search_hits, window_geometry


class ClassPredictionDetailBar(LabelDetailBar):
    """
    Widget for visualizing predicted behavior labels and their probabilities in a video timeline.

    Displays a horizontal bar where each frame is color-coded according to the predicted label,
    with transparency indicating the model's confidence (probability). Unlike LabelDetailBar,
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

        geometry = window_geometry(self._current_frame, self._window_size, self._num_frames, fw)

        qp = QPainter(self)
        qp.setPen(Qt.PenStyle.NoPen)

        # Fill entire bar with padding pattern, then overdraw in-bounds region
        qp.setBrush(self._padding_brush)
        qp.drawRect(0, 0, widget_width, self._bar_height)

        if geometry.has_in_bounds_content:
            # White background for the in-bounds region; predictions are overlaid with alpha
            qp.setBrush(Qt.GlobalColor.white)
            qp.drawRect(geometry.x0, 0, geometry.width_px, self._bar_height)

        # Draw predictions overlaid on the white background; lower probability = more transparent.
        if self._predictions is not None and geometry.has_in_bounds_content:
            frames = slice(geometry.slice_start, geometry.slice_end + 1)
            color_indices = self._predictions[frames]
            probs_slice = (
                np.clip(self._probabilities[frames], 0.0, 1.0)
                if self._probabilities is not None
                else None
            )
            colors = self._build_frame_colors(color_indices, probs_slice)

            draw_frame_colors(qp, colors, geometry, fw, self._bar_height)

        render_search_hits(
            qp,
            self._search_results,
            0,
            geometry.start,
            self._frame_width,
            self._bar_height,
            self._window_frames_total,
        )

        self._draw_position_marker(qp)
        self._draw_bounding_box(qp)
        self._draw_second_ticks(qp, geometry.start, geometry.end)

        # done drawing
        qp.end()

    def _build_frame_colors(
        self,
        color_indices: npt.NDArray[np.int16],
        probs_slice: npt.NDArray[np.floating] | None,
    ) -> npt.NDArray[np.uint8]:
        """Build per-frame RGBA color array from LUT indices and probabilities.

        Subclasses can override this to change how colors are computed without
        duplicating the full paint pipeline.

        Args:
            color_indices: LUT index per in-bounds frame, shape ``(n,)``.
            probs_slice: Clipped probabilities ``[0, 1]`` for the same frames, or ``None``.

        Returns:
            RGBA array of shape ``(n, 4)`` with dtype ``uint8``.
        """
        colors = self._color_lut[color_indices].copy()
        if probs_slice is not None:
            alphas = (probs_slice * 255).astype(np.uint8)
            # Interpolated post-processed frames can have pred!=0 but prob==0; keep them visible.
            mask = (color_indices != 0) & (alphas == 0)
            alphas[mask] = 125
            colors[:, 3] = alphas
        return colors

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
        """Not supported in ClassPredictionDetailBar.

        Raises:
            NotImplementedError: Always; selection is not supported on prediction widgets.
        """
        raise NotImplementedError

    def clear_selection(self) -> None:
        """Not supported in ClassPredictionDetailBar"""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the widget to its initial state.

        Clears any displayed predictions and probabilities, and calls the parent
        class's reset method to perform additional cleanup.
        """
        self._predictions = None
        self._probabilities = None
        super().reset()
