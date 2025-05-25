from enum import IntEnum

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFrame, QSizePolicy, QVBoxLayout, QWidget

from jabs.project import TrackLabels

from .frame_labels_widget import FrameLabelsWidget
from .label_overview_widget import LabelOverviewWidget, PredictionOverviewWidget


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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

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
        self._frame_labels = FrameLabelsWidget(self)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

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
    def num_identities(self) -> int:
        """Get the number of identities."""
        return self._num_identities

    @num_identities.setter
    def num_identities(self, value: int) -> None:
        """Set the number of identities and reset the layout."""
        if value != self._num_identities:
            self._num_identities = value
            self._reset_layout()

    @property
    def view_mode(self) -> ViewMode:
        """Get the current view mode."""
        return self._view_mode

    @view_mode.setter
    def view_mode(self, value: ViewMode) -> None:
        """Set the view mode and update the layout accordingly."""
        if value != self._view_mode:
            self._view_mode = value
            self._update_widget_visibility()

    @property
    def identity_mode(self) -> IdentityMode:
        """Get the current identity mode."""
        return self._identity_mode

    @identity_mode.setter
    def identity_mode(self, value: IdentityMode) -> None:
        """Set the identity mode and update the layout accordingly."""
        if value != self._identity_mode:
            self._identity_mode = value
            self._update_widget_visibility()

    def _reset_layout(self) -> None:
        """Recreate the layout and child widgets for all identities.

        Removes existing frames and overview widgets, then creates new frames,
        label overview widgets, and prediction overview widgets for each identity.
        Updates the internal lists and layout, and sets the active identity index.

        This method is called when the number of identities changes.
        """
        # Remove old widgets and frames
        for frame in self._identity_frames:
            self._layout.removeWidget(frame)
            frame.setParent(None)
            frame.deleteLater()
        self._layout.removeWidget(self._frame_labels)
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

    def _set_active_frame_border(self, active_index: int | None = None) -> None:
        """Update the visual border for the active identity frame.

        Highlights the QFrame corresponding to the active identity by applying a border style,
        and removes the border from all other frames. If `active_index` is not provided,
        uses the current active identity index.

        Args:
            active_index: Optional index of the frame to highlight. If None, uses the current active identity.
        """
        active_index = (
            self._active_identity_index if active_index is None else active_index
        )
        for i, frame in enumerate(self._identity_frames):
            if i == active_index:
                frame.setStyleSheet(
                    f"QFrame {{border: 2px solid {self._BORDER_COLOR}; border-radius: 8px;}}"
                )
            else:
                frame.setStyleSheet("QFrame {border: none; padding: 2px;}")

    @property
    def num_frames(self) -> int:
        """Get the number of frames."""
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value: int) -> None:
        """Set the number of frames."""
        self._num_frames = value
        self._frame_labels.set_num_frames(value)

    @property
    def framerate(self) -> int:
        """Get the framerate."""
        return self._framerate

    @framerate.setter
    def framerate(self, value: int) -> None:
        """Set the framerate."""
        self._framerate = value

    @property
    def active_identity_index(self) -> int:
        """Get the index of the active identity."""
        return self._active_identity_index

    @active_identity_index.setter
    def active_identity_index(self, value: int) -> None:
        """Set the index of the currently active identity.

        In addition to setting the active identity index, this method also handles the case where
        the selection is active. It clears the selection on the old active widget and transfers it to
        the new active widget. The method also updates the frame border and visibility of the widgets
        depending on the current identity and view modes.
        """
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
                self._set_active_frame_border()

            self._update_widget_visibility()

    def _update_frame_border(self):
        """Update the frame border for the active identity."""
        if self._identity_mode == self.IdentityMode.ALL:
            self._set_active_frame_border()
        else:
            self._set_active_frame_border(-1)

    def _update_widget_visibility(self) -> None:
        """Update which identity frames and widgets are visible based on the current identity and view modes.

        Removes all frames from the layout, then adds back either all identity frames or only the active one,
        depending on the identity mode. Sets the visibility of label and prediction widgets within each frame
        according to the current view mode. Also updates the frame border to reflect the active identity.
        """
        # Remove all frames from the layout
        for frame in self._identity_frames:
            self._layout.removeWidget(frame)

        if self._identity_mode == self.IdentityMode.ALL:
            # Add all frames
            for i, frame in enumerate(self._identity_frames):
                self._layout.addWidget(frame)
                # Set visibility for each widget based on view_mode
                self._set_widget_visibility(
                    self._label_overview_widgets[i],
                    self._prediction_overview_widgets[i],
                )

        elif (
            self._identity_mode == self.IdentityMode.ACTIVE
            and self._active_identity_index is not None
        ):
            # Add only the active frame
            idx = self._active_identity_index
            frame = self._identity_frames[idx]
            self._layout.addWidget(frame)
            self._set_widget_visibility(
                self._label_overview_widgets[idx],
                self._prediction_overview_widgets[idx],
            )

        self._layout.addWidget(self._frame_labels)
        self._update_frame_border()

    def _set_widget_visibility(
        self,
        label_widget: LabelOverviewWidget,
        prediction_widget: PredictionOverviewWidget,
    ) -> None:
        """Set the visibility of label and prediction widgets based on the current view mode.

        Shows or hides the provided label and prediction widgets according to the selected
        view mode (labels only, predictions only, or both).

        Args:
            label_widget: The LabelOverviewWidget to show or hide.
            prediction_widget: The PredictionOverviewWidget to show or hide.
        """
        if self._view_mode == self.ViewMode.LABELS_AND_PREDICTIONS:
            label_widget.setVisible(True)
            prediction_widget.setVisible(True)
        elif self._view_mode == self.ViewMode.LABELS:
            label_widget.setVisible(True)
            prediction_widget.setVisible(False)
        elif self._view_mode == self.ViewMode.PREDICTIONS:
            label_widget.setVisible(False)
            prediction_widget.setVisible(True)

    @Slot(int)
    def set_current_frame(self, current_frame: int) -> None:
        """Forward current frame to all LabelOverviewWidgets and PredictionOverviewWidgets."""
        for label_widget, prediction_widget in zip(
            self._label_overview_widgets, self._prediction_overview_widgets, strict=True
        ):
            label_widget.set_current_frame(current_frame)
            prediction_widget.set_current_frame(current_frame)
        self._frame_labels.set_current_frame(current_frame)

    def set_labels(
        self, labels_list: list[TrackLabels], masks_list: list[np.ndarray]
    ) -> None:
        """Set labels for all LabelOverviewWidgets.

        Args:
            labels_list: List of TrackLabels, one per identity.
            masks_list: Optional list of masks, one per identity.
        """
        if (
            len(labels_list) != len(masks_list)
            or len(labels_list) != self._num_identities
        ):
            raise ValueError("Input length does not match number of identities.")

        for i, widget in enumerate(self._label_overview_widgets):
            labels = labels_list[i]
            mask = masks_list[i]

            # need to set the number of frames and framerate on the child widgets because they were zero when they
            # were created. Now that data has been loaded, they can be set to the correct values.
            widget.num_frames = self.num_frames
            widget.framerate = self.framerate

            widget.set_labels(labels, mask)

    def set_predictions(
        self, predictions_list: list[np.ndarray], probabilities_list: list[np.ndarray]
    ) -> None:
        """
        Set predictions for all PredictionOverviewWidgets.

        Args:
            predictions_list: List of np.ndarray, one per identity.
            probabilities_list: List of np.ndarray, one per identity.
        """
        if len(predictions_list) != self._num_identities:
            raise ValueError(
                f"Number of predictions ({len(predictions_list)}) does not match number of identities ({self._num_identities})."
            )

        for i, widget in enumerate(self._prediction_overview_widgets):
            widget.num_frames = self.num_frames
            widget.framerate = self.framerate
            widget.set_predictions(predictions_list[i], probabilities_list[i])

    def start_selection(self, starting_frame: int) -> None:
        """Start a selection from the given frame on the active identity's widget.

        Records the starting frame and initiates selection mode on the currently active identity.

        Args:
            starting_frame: The frame index where the selection begins.
        """
        if self._active_identity_index is not None:
            self._selection_starting_frame = starting_frame
            self._label_overview_widgets[self._active_identity_index].start_selection(
                starting_frame
            )
            self._label_overview_widgets[self._active_identity_index].update()

    def clear_selection(self) -> None:
        """Clear the current selection on the active identity's widget.

        Exits selection mode and resets the selection starting frame.
        """
        if self._active_identity_index is not None:
            self._label_overview_widgets[self._active_identity_index].clear_selection()
            self._selection_starting_frame = None
            self._label_overview_widgets[self._active_identity_index].update_labels()

    def reset(self) -> None:
        """Reset all child widgets to their initial state.

        Clears all internal data and resets the state of label and prediction overview widgets.
        """
        for widget in self._label_overview_widgets:
            widget.reset()

        for widget in self._prediction_overview_widgets:
            widget.reset()
