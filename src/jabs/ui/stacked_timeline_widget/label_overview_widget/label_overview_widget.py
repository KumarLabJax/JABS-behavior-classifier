import numpy as np
from PySide6.QtCore import QSize, Slot
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from jabs.behavior_search import SearchHit
from jabs.project import TrackLabels

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
        super().__init__(*args, **kwargs)

        self._num_frames = 0

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Use a plain QWidget as the container
        self._container = QWidget(self)

        self._timeline_widget = self._timeline_widget_factory(self._container)
        self._label_widget = self._label_widget_factory(self._container)

        self._set_layout()

    @classmethod
    def _timeline_widget_factory(cls, parent: QWidget) -> TimelineLabelWidget:
        """factory method to create the timeline widget

        This is done to make it easier to subclass the widget and swap out the timeline widget
        """
        return TimelineLabelWidget(parent)

    @classmethod
    def _label_widget_factory(cls, parent: QWidget) -> ManualLabelWidget:
        """factory method to create the label widget

        This is done to make it easier to subclass the widget and swap out the label widget
        """
        return ManualLabelWidget(parent)

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

        Calculates the preferred size based on the size hints of the timeline and manual label widgets.

        Returns:
            QSize: The recommended size for the widget.
        """
        timeline_hint = self._timeline_widget.sizeHint()
        manual_hint = self._label_widget.sizeHint()
        width = max(timeline_hint.width(), manual_hint.width())
        height = timeline_hint.height() + manual_hint.height()
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

    def set_labels(self, labels: TrackLabels, mask: np.ndarray) -> None:
        """Set the data for the widget.

        Args:
            labels: TrackLabels object containing labels.
            mask:  mask array
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
