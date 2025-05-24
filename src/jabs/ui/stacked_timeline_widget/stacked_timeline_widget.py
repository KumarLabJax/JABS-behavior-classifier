from enum import IntEnum

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

from ..label_overview_widget import LabelOverviewWidget, PredictionOverviewWidget


class StackedTimelineWidget(QWidget):
    """A widget that manages and displays multiple LabelOverviewWidgets, one for each identity.

    This widget allows toggling between showing all identities or only the active one,
    manages selection transfer between identities, and forwards label and frame updates
    to its child widgets. It is designed for use in multi-identity labeling interfaces,
    such as behavioral video annotation tools.

    Properties:
        num_identities (int): Number of identities to display.
        num_frames (int): Number of frames in the video.
        framerate (int): Framerate of the video.
        active_identity_index (int): Index of the currently active identity.
        show_only_active_identity (bool): Whether to display only the active identity.
    """

    _BORDER_COLOR = "#0078d7"

    class ViewMode(IntEnum):
        """Enum for view modes of the widget."""

        LABELS_AND_PREDICTIONS = 0
        LABELS = 1
        PREDICTIONS = 2

    class IdentityMode(IntEnum):
        """Enum for identity modes of the widget."""

        ALL = 0
        ACTIVE = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._active_identity_index = None
        self._selection_starting_frame = None
        self._view_mode = self.ViewMode.LABELS_AND_PREDICTIONS
        self._identity_mode = self.IdentityMode.ACTIVE
        self._num_identities = 0
        self._num_frames = 0
        self._framerate = 0
        self._label_overview_widgets = []
        self._prediction_overview_widgets = []
        self._identity_frames = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.setLayout(self._layout)

        self._show_all_predictions = False

    def _label_overview_widget_factory(self, parent) -> LabelOverviewWidget:
        """Factory method to create a label overview widget."""
        widget = LabelOverviewWidget(parent)
        widget.num_frames = self.num_frames
        widget.framerate = self.framerate
        return widget

    def _prediction_overview_widget_factory(self, parent) -> PredictionOverviewWidget:
        """Factory method to create a prediction overview widget."""
        widget = PredictionOverviewWidget(parent)
        widget.num_frames = self.num_frames
        widget.framerate = self.framerate
        return widget

    @property
    def num_identities(self):
        """Get the number of identities."""
        return self._num_identities

    @num_identities.setter
    def num_identities(self, value: int):
        """Set the number of identities and reset the layout."""
        if value != self._num_identities:
            self._num_identities = value
            self._reset_layout()

    @property
    def view_mode(self):
        """Get the current view mode."""
        return self._view_mode

    @view_mode.setter
    def view_mode(self, value: ViewMode):
        """Set the view mode and update the layout accordingly."""
        if value != self._view_mode:
            self._view_mode = value
            self._update_widget_visibility()

    @property
    def identity_mode(self):
        """Get the current identity mode."""
        return self._identity_mode

    @identity_mode.setter
    def identity_mode(self, value: IdentityMode):
        """Set the identity mode and update the layout accordingly."""
        if value != self._identity_mode:
            self._identity_mode = value
            self._update_widget_visibility()

    def _reset_layout(self):
        # Remove old widgets and frames
        for frame in self._identity_frames:
            self._layout.removeWidget(frame)
            frame.setParent(None)
            frame.deleteLater()
        self._label_overview_widgets = []
        self._prediction_overview_widgets = []
        self._identity_frames = []

        # Create new frames and widgets
        for _ in range(self._num_identities):
            frame = QFrame(self)
            frame.setFrameShape(QFrame.Shape.NoFrame)
            frame.setStyleSheet("QFrame {border: none; padding: 2px;}")
            vbox = QVBoxLayout(frame)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(4)

            label_widget = self._label_overview_widget_factory(frame)
            prediction_widget = self._prediction_overview_widget_factory(frame)
            label_widget.setVisible(False)
            prediction_widget.setVisible(False)

            vbox.addWidget(label_widget)
            vbox.addWidget(prediction_widget)

            self._label_overview_widgets.append(label_widget)
            self._prediction_overview_widgets.append(prediction_widget)
            self._identity_frames.append(frame)
            self._layout.addWidget(frame)
        self._update_frame_border()
        if self._num_identities > 0:
            self._active_identity_index = 0
        else:
            self._active_identity_index = None
        self._update_widget_visibility()

    def _set_active_frame_border(self, active_index):
        for i, frame in enumerate(self._identity_frames):
            if i == active_index:
                frame.setStyleSheet(
                    f"QFrame {{border: 2px solid {self._BORDER_COLOR}; border-radius: 4px;}}"
                )
            else:
                frame.setStyleSheet("QFrame {border: none; padding: 2px;}")

    @property
    def num_frames(self):
        """Get the number of frames."""
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value: int):
        """Set the number of frames."""
        self._num_frames = value

    @property
    def framerate(self):
        """Get the framerate."""
        return self._framerate

    @framerate.setter
    def framerate(self, value: int):
        """Set the framerate."""
        self._framerate = value

    @property
    def active_identity_index(self):
        """Get the index of the active identity."""
        return self._active_identity_index

    @active_identity_index.setter
    def active_identity_index(self, value: int):
        if value != self._active_identity_index:
            old_index = self._active_identity_index
            selection_frame = self._selection_starting_frame

            # Clear selection on old active widget if selection is active
            if old_index is not None and selection_frame is not None:
                self._label_overview_widgets[old_index].clear_selection()

            self._active_identity_index = value

            # Transfer selection to new active widget if selection is active
            if selection_frame is not None:
                self._label_overview_widgets[
                    self._active_identity_index
                ].start_selection(selection_frame)

            # Set active state or frame border depending on display mode
            if self._identity_mode == self.IdentityMode.ALL:
                self._set_active_frame_border(self._active_identity_index)

            self._update_widget_visibility()

    def _update_frame_border(self):
        """Update the frame border for the active identity."""
        if self._identity_mode == self.IdentityMode.ALL:
            self._set_active_frame_border(self._active_identity_index)
        else:
            self._set_active_frame_border(-1)

    def _update_widget_visibility(self):
        # Remove all frames from the layout
        for frame in self._identity_frames:
            self._layout.removeWidget(frame)
            frame.setParent(None)

        if self._identity_mode == self.IdentityMode.ALL:
            # Add all frames
            for i, frame in enumerate(self._identity_frames):
                self._layout.addWidget(frame)
                # Set visibility for each widget based on view_mode
                if self._view_mode == self.ViewMode.LABELS_AND_PREDICTIONS:
                    self._label_overview_widgets[i].setVisible(True)
                    self._prediction_overview_widgets[i].setVisible(True)
                elif self._view_mode == self.ViewMode.LABELS:
                    self._label_overview_widgets[i].setVisible(True)
                    self._prediction_overview_widgets[i].setVisible(False)
                elif self._view_mode == self.ViewMode.PREDICTIONS:
                    self._label_overview_widgets[i].setVisible(False)
                    self._prediction_overview_widgets[i].setVisible(True)
        elif (
            self._identity_mode == self.IdentityMode.ACTIVE
            and self._active_identity_index is not None
        ):
            # Add only the active frame
            idx = self._active_identity_index
            frame = self._identity_frames[idx]
            self._layout.addWidget(frame)
            if self._view_mode == self.ViewMode.LABELS_AND_PREDICTIONS:
                self._label_overview_widgets[idx].setVisible(True)
                self._prediction_overview_widgets[idx].setVisible(True)
            elif self._view_mode == self.ViewMode.LABELS:
                self._label_overview_widgets[idx].setVisible(True)
                self._prediction_overview_widgets[idx].setVisible(False)
            elif self._view_mode == self.ViewMode.PREDICTIONS:
                self._label_overview_widgets[idx].setVisible(False)
                self._prediction_overview_widgets[idx].setVisible(True)

        self._update_frame_border()

    @Slot(int)
    def set_current_frame(self, current_frame: int):
        """Forward current frame to all LabelOverviewWidgets and PredictionOverviewWidgets."""
        for label_widget, prediction_widget in zip(
            self._label_overview_widgets, self._prediction_overview_widgets, strict=True
        ):
            label_widget.set_current_frame(current_frame)
            prediction_widget.set_current_frame(current_frame)

    def set_labels(self, labels_list, masks_list=None):
        """
        Set labels for all LabelOverviewWidgets.

        Args:
            labels_list: List of TrackLabels, one per identity.
            masks_list: Optional list of masks, one per identity.
        """
        if len(labels_list) != self._num_identities:
            raise ValueError(
                f"Number of labels ({len(labels_list)}) does not match number of identities ({self._num_identities})."
            )

        for i, widget in enumerate(self._label_overview_widgets):
            labels = labels_list[i]
            mask = masks_list[i] if masks_list else None

            # need to set the number of frames and framerate on the child widgets because they were zero when they
            # were created. Now that data has been loaded, they can be set to the correct values.
            widget.num_frames = self._num_frames
            widget.framerate = self._framerate

            widget.set_labels(labels, mask)

    def set_predictions(self, predictions_list, probabilities_list):
        """
        Set predictions for all PredictionOverviewWidgets.

        Args:
            predictions_list: List of np.ndarray, one per identity.
            probabilities_list: Optional list of np.ndarray, one per identity.
        """
        if len(predictions_list) != self._num_identities:
            raise ValueError(
                f"Number of predictions ({len(predictions_list)}) does not match number of identities ({self._num_identities})."
            )

        for i, widget in enumerate(self._prediction_overview_widgets):
            widget.num_frames = self._num_frames
            widget.framerate = self._framerate

            predictions = predictions_list[i]
            probabilities = probabilities_list[i]
            widget.set_predictions(predictions, probabilities)

    def start_selection(self, starting_frame: int):
        """Start selection on the active identity's widget and record the starting frame."""
        if self._active_identity_index is not None:
            self._selection_starting_frame = starting_frame
            self._label_overview_widgets[self._active_identity_index].start_selection(
                starting_frame
            )
            self._label_overview_widgets[self._active_identity_index].update()

    def clear_selection(self):
        """Clear selection on the active identity's widget and reset the starting frame."""
        if self._active_identity_index is not None:
            self._label_overview_widgets[self._active_identity_index].clear_selection()
            self._selection_starting_frame = None
            self._label_overview_widgets[self._active_identity_index].update_labels()

    def reset(self):
        """reset state of all child widgets"""
        for widget in self._label_overview_widgets:
            widget.reset()

        for widget in self._prediction_overview_widgets:
            widget.reset()
