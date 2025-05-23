import numpy as np
from PySide6.QtCore import QSize, Slot
from PySide6.QtWidgets import QFrame, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget

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
    """

    _BORDER_COLOR = "#0078d7"
    _FRAME_VERTICAL_SPACING = 6
    _FRAME_HORIZONTAL_SPACING = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._active = False
        self._num_frames = 0

        # allow widget to expand horizontally but maintain fixed vertical size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._frame = QFrame(self)
        self._frame.setFrameShape(QFrame.Shape.NoFrame)
        self._frame.setLineWidth(2)

        self._timeline_widget = self._timeline_widget_factory(self._frame)
        self._label_widget = self._label_widget_factory(self._frame)

        self._set_layout()
        self._update_border()

    @staticmethod
    def _timeline_widget_factory(parent):
        """factory method to create the timeline widget

        This is done to make it easier to subclass the widget and swap out the timeline widget
        """
        return TimelineLabelWidget(parent)

    @staticmethod
    def _label_widget_factory(parent):
        """factory method to create the label widget

        This is done to make it easier to subclass the widget and swap out the label widget
        """
        return ManualLabelWidget(parent)

    def _set_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._frame)
        self.setLayout(layout)

        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(0, 8, 0, 8)
        frame_layout.setSpacing(0)

        frame_layout.addWidget(self._timeline_widget)
        frame_layout.addWidget(self._timeline_widget)
        frame_layout.addItem(
            QSpacerItem(
                self._FRAME_HORIZONTAL_SPACING,
                self._FRAME_VERTICAL_SPACING,
                QSizePolicy.Policy.Minimum,
                QSizePolicy.Policy.Fixed,
            )
        )
        frame_layout.addWidget(self._label_widget)
        frame_layout.addWidget(self._label_widget)

    def sizeHint(self):
        """Return the recommended size for the widget based on its children."""
        timeline_hint = self._timeline_widget.sizeHint()
        manual_hint = self._label_widget.sizeHint()
        width = max(timeline_hint.width(), manual_hint.width())
        height = timeline_hint.height() + manual_hint.height()
        return QSize(width, height)

    @property
    def active(self) -> bool:
        """Return whether the widget is the active identity or not."""
        return self._active

    @active.setter
    def active(self, value: bool):
        if self._active != value:
            self._active = value
            self._update_border()

    @property
    def num_frames(self):
        """number of frames in the current video"""
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value: int):
        """Set the number of frames in the video.

        Args:
            value: number of frames in the video
        """
        self._timeline_widget.set_num_frames(value)
        self._label_widget.set_num_frames(value)

    @property
    def framerate(self):
        """framerate is a write-only property, only included so we can have a setter"""
        raise AttributeError("framerate is a write-only property")

    @framerate.setter
    def framerate(self, fps: int):
        """Set the framerate of the video.

        Args:
            fps: framerate of the video
        """
        self._label_widget.set_framerate(fps)

    def set_labels(self, labels: TrackLabels, mask: np.ndarray | None = None):
        """Set the data for the widget.

        Args:
            labels: TrackLabels object containing labels.
            mask: optional mask array
        """
        self._timeline_widget.set_labels(labels)
        self._label_widget.set_labels(labels, mask)

    @Slot(int)
    def set_current_frame(self, current_frame: int):
        """Receive current frame and forward to child widgets."""
        self._timeline_widget.set_current_frame(current_frame)
        self._label_widget.set_current_frame(current_frame)

    def reset(self):
        """Reset the widget to its initial state."""
        self._timeline_widget.reset()
        self._label_widget.set_labels(None)
        self._num_frames = 0
        self._active = False

    def start_selection(self, starting_frame: int):
        """Start a selection from the given frame to the current frame."""
        self._label_widget.start_selection(starting_frame)

    def clear_selection(self):
        """Clear the current selection.

        exit selection mode
        """
        self._label_widget.clear_selection()

    def update_labels(self):
        """Update the labels in the widget.

        This is called when the underlying data has changed and the widget needs to be redrawn.
        """
        self._timeline_widget.update_labels()
        self.update()

    def update(self):
        """Update the widget.

        This is called when the widget needs to be redrawn because the underlying data has changed.
        """
        self._timeline_widget.update()
        self._label_widget.update()

    def _update_border(self):
        if self._active:
            self._frame.setStyleSheet(
                f"QFrame {{ border: 2px solid {self._BORDER_COLOR}; border-radius: 4px; }}"
            )
        else:
            self._frame.setStyleSheet("QFrame { border: none; }")
