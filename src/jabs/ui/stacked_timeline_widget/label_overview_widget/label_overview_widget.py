import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QSize, Slot
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from jabs.behavior_search import SearchHit

from .manual_label_widget import ManualLabelWidget
from .timeline_label_widget import TimelineLabelWidget


class LabelOverviewWidget(QWidget):
    """Widget for displaying an overview of manual behavior labels for one labeling subject.

    Combines a timeline widget and a manual label widget in a vertically stacked layout,
    allowing visualization and of frame-wise labels. Designed to be extensible
    via factory methods for swapping out child widgets in subclasses.

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.


    Properties:
        framerate: int write-only property to set the framerate of the video.
        num_frames: int number of frames in the current video.
    """

    def __init__(self, *args, **kwargs) -> None:
        # need to strip "compact" out of kwargs before calling super
        compact = kwargs.pop("compact", False)
        super().__init__(*args, **kwargs)

        self._num_frames = 0

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Use a plain QWidget as the container
        self._container = QWidget(self)

        self._timeline_widget = self._timeline_widget_factory(self._container)
        self._label_widget = self._label_widget_factory(self._container, compact)

        self._set_layout()

    @property
    def compact(self) -> bool:
        """Whether the label widget is in compact mode.

        Returns:
            True if compact mode is active, False otherwise.
        """
        return self._label_widget.compact

    @compact.setter
    def compact(self, value: bool) -> None:
        """Set compact mode on the label widget.

        Args:
            value: True to enable compact mode, False to disable.
        """
        self._label_widget.compact = value

    @classmethod
    def _timeline_widget_factory(cls, parent: QWidget) -> TimelineLabelWidget:
        """factory method to create the timeline widget

        This is done to make it easier to subclass the widget and swap out the timeline widget
        """
        return TimelineLabelWidget(parent)

    @classmethod
    def _label_widget_factory(cls, parent: QWidget, compact: bool = False) -> ManualLabelWidget:
        """factory method to create the label widget

        This is done to make it easier to subclass the widget and swap out the label widget
        """
        return ManualLabelWidget(parent, compact=compact)

    def _set_layout(self) -> None:
        """Set up the vertical layout for the widget.

        Arranges the timeline and manual label widgets in a vertical stack within the container.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 6)
        layout.setSpacing(4)
        layout.addWidget(self._timeline_widget)
        layout.addWidget(self._label_widget)
        self.setLayout(layout)

    def sizeHint(self) -> QSize:
        """Return the recommended size for the widget.

        Calculates the preferred size based on the visible child widgets only, so
        that hiding the detail bar via :meth:`set_detail_bar_visible` actually
        shrinks the widget.

        Returns:
            QSize: The recommended size for the widget.
        """
        timeline_hint = self._timeline_widget.sizeHint()
        width = max(timeline_hint.width(), self._label_widget.sizeHint().width())
        height = timeline_hint.height()
        if self._label_widget.isVisible():
            height += self._label_widget.sizeHint().height()
        return QSize(width, height)

    @property
    def num_frames(self) -> int:
        """Get the number of frames in the current video.

        Returns:
            int: The total number of frames loaded in the widget.
        """
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value: int) -> None:
        """Set the number of frames in the video.

        Args:
            value: number of frames in the video
        """
        self._num_frames = value
        self._timeline_widget.set_num_frames(value)
        self._label_widget.set_num_frames(value)

    @property
    def framerate(self) -> int:
        """framerate is a write-only property, only included so we can have a setter"""
        raise AttributeError("framerate is a write-only property")

    @framerate.setter
    def framerate(self, fps: int) -> None:
        """Set the framerate of the video.

        Args:
            fps: framerate of the video
        """
        self._label_widget.set_framerate(fps)

    def set_color_lut(self, lut: npt.NDArray[np.uint8]) -> None:
        """Set the color lookup table for both child widgets.

        Propagates a custom LUT to the timeline and label widgets, enabling
        multi-class color rendering.

        Args:
            lut: RGBA array of shape ``(N, 4)`` mapping class indices to colors.

        Raises:
            ValueError: If ``lut`` is not a 2-D array with exactly 4 columns.
        """
        if lut.ndim != 2 or lut.shape[1] != 4:
            raise ValueError(f"lut must have shape (N, 4), got {lut.shape}")
        self._timeline_widget.set_color_lut(lut)
        self._label_widget.set_color_lut(lut)

    def set_detail_bar_visible(self, visible: bool) -> None:
        """Show or hide the detail (manual-label) bar below the overview strip.

        Used in multi-class + all-animals mode to collapse the detail bar on
        non-active identities and recover vertical space.  Also tightens the
        widget's internal margins when collapsed so overview bars pack together
        with minimal whitespace.

        Args:
            visible: ``True`` to show the detail bar; ``False`` to hide it.
        """
        self._label_widget.setVisible(visible)
        if visible:
            self.layout().setContentsMargins(2, 4, 2, 6)
            self.layout().setSpacing(4)
        else:
            self.layout().setContentsMargins(2, 1, 2, 1)
            self.layout().setSpacing(0)
        self.updateGeometry()

    def set_labels(self, labels: npt.NDArray[np.int16], mask: np.ndarray) -> None:
        """Set the label data for the widget.

        ``labels`` must be a direct LUT-index array whose values correspond to
        rows in the active color LUT.  Callers are responsible for producing this
        array before calling.

        Args:
            labels: Class-index array of shape ``(n_frames,)`` with dtype ``int16``.
            mask: Mask array indicating valid identity frames.
        """
        self._timeline_widget.set_labels(labels)
        self._label_widget.set_labels(labels, mask)

    def set_search_results(self, search_results: list[SearchHit]) -> None:
        """Set the search results for the widget.

        Args:
            search_results: List of SearchHit objects to display in the label widget.
        """
        self._timeline_widget.set_search_results(search_results)
        self._label_widget.set_search_results(search_results)

    @Slot(int)
    def set_current_frame(self, current_frame: int) -> None:
        """Set the current frame in the timeline and label widgets.

        Args:
            current_frame: The index of the frame to set as current.
        """
        self._timeline_widget.set_current_frame(current_frame)
        self._label_widget.set_current_frame(current_frame)

    def reset(self) -> None:
        """Reset the widget and its child widgets to their initial state.

        Clears all internal data and resets the number of frames.
        """
        self._timeline_widget.reset()
        self._label_widget.reset()
        self._num_frames = 0

    def start_selection(self, starting_frame: int, ending_frame: int | None = None) -> None:
        """Start a selection from the given frame to the current frame.

        Args:
            starting_frame: The frame index where the selection begins.
            ending_frame: Optional; the frame index where the selection ends. If None,
                selection continues to current frame.
        """
        self._label_widget.start_selection(starting_frame, ending_frame)

    def clear_selection(self) -> None:
        """Clear the current selection.

        exit selection mode
        """
        self._label_widget.clear_selection()

    def update_labels(self) -> None:
        """Update the labels in the widget.

        This is called when the underlying data has changed and the widget needs to be redrawn.
        """
        self._timeline_widget.update_labels()
        self.update()

    def update(self) -> None:
        """Update the widget.

        This is called when the widget needs to be redrawn because the underlying data has changed.
        """
        self._timeline_widget.update()
        self._label_widget.update()
