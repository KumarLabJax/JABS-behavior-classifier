from enum import IntEnum

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from jabs.behavior_search import (
    BehaviorSearchQuery,
    LabelBehaviorSearchQuery,
    PredictionBehaviorSearchQuery,
    SearchHit,
    TimelineAnnotationSearchQuery,
)
from jabs.pose_estimation import PoseEstimation
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
        self._selection_ending_frame = None
        self._view_mode = self.ViewMode.LABELS_AND_PREDICTIONS
        self._identity_mode = self.IdentityMode.ACTIVE
        self._num_identities = 0
        self._num_frames = 0
        self._framerate = 0
        self._label_overview_widgets: list[LabelOverviewWidget] = []
        self._prediction_overview_widgets = []
        self._identity_frames = []
        self._frame_labels = FrameLabelsWidget(self)
        self._pose: PoseEstimation | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # handle palette changes
        app = QApplication.instance()
        if app and hasattr(app, "paletteChanged"):
            app.paletteChanged.connect(self._on_palette_changed)

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

    @property
    def pose(self) -> PoseEstimation | None:
        """Get the PoseEstimation object used by the label overview widgets."""
        return self._pose

    @pose.setter
    def pose(self, pose_est: PoseEstimation) -> None:
        """Set the PoseEstimation object used by the label overview widgets.

        Args:
            pose_est: PoseEstimation object to set.
        """
        self._pose = pose_est
        self._num_identities = pose_est.num_identities
        self._num_frames = pose_est.num_frames
        self._reset_layout()

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
        for identity_index in range(self._num_identities):
            if self._pose:
                identity_display_name = self._pose.identity_index_to_display(identity_index)
            else:
                identity_display_name = str(identity_index)

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

            vbox.addWidget(QLabel(f"{identity_display_name}:"))
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

        self._layout.addWidget(self._frame_labels)
        self._update_widget_visibility()

    def _set_active_frame_border(self, active_index: int | None = None) -> None:
        """Update the visual border for the active identity frame.

        Highlights the QFrame corresponding to the active identity by applying a border style,
        and removes the border from all other frames. If `active_index` is not provided,
        uses the current active identity index.

        Args:
            active_index: Optional index of the frame to highlight. If None, uses the current active identity.
                If active_index does not match any identity, no border is applied. (we pass -1 to clear all
                borders)
        """
        accent_color = self._get_accent_color()
        active_index = self._active_identity_index if active_index is None else active_index
        for i, frame in enumerate(self._identity_frames):
            if i == active_index:
                frame.setStyleSheet(
                    f"QFrame > QWidget {{border: none;}} QFrame {{border: 2px solid {accent_color}; border-radius: 8px; padding: 2px;}}"
                )
            else:
                frame.setStyleSheet(
                    "QFrame > QWidget {border: none;} QFrame {border: 2px solid transparent; border-radius: 8px; padding: 2px;}"
                )

    @staticmethod
    def _get_accent_color() -> str:
        """Get the accent color from the application palette."""
        palette = QApplication.palette()
        return palette.color(palette.ColorRole.Accent).name()

    @property
    def num_frames(self) -> int:
        """Get the number of frames."""
        return self._num_frames

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
                self._label_overview_widgets[self._active_identity_index].start_selection(
                    selection_frame,
                    self._selection_ending_frame,
                )

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
        # Remove all widgets from the layout
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()

        # Add only the widgets needed for the current mode
        if self._identity_mode == self.IdentityMode.ALL:
            for i, frame in enumerate(self._identity_frames):
                self._layout.addWidget(frame)
                frame.show()
                self._set_widget_visibility(
                    self._label_overview_widgets[i],
                    self._prediction_overview_widgets[i],
                )
        elif (
            self._identity_mode == self.IdentityMode.ACTIVE
            and self._active_identity_index is not None
        ):
            idx = self._active_identity_index
            frame = self._identity_frames[idx]
            self._layout.addWidget(frame)
            frame.show()
            self._set_widget_visibility(
                self._label_overview_widgets[idx],
                self._prediction_overview_widgets[idx],
            )

        # Add FrameLabelsWidget last
        self._layout.addWidget(self._frame_labels)
        self._frame_labels.show()
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

    def set_labels(self, labels_list: list[TrackLabels], masks_list: list[np.ndarray]) -> None:
        """Set labels for all LabelOverviewWidgets.

        Args:
            labels_list: List of TrackLabels, one per identity.
            masks_list: Optional list of masks, one per identity.
        """
        if len(labels_list) != self._num_identities:
            raise ValueError(
                f"Number of TrackLabels in labels_list ({len(labels_list)}) "
                f"does not match number of identities ({self._num_identities})."
            )
        if len(masks_list) != self._num_identities:
            raise ValueError(
                f"Number of mask arrays in masks_list ({len(masks_list)}) "
                f"does not match number of identities ({self._num_identities})."
            )

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
                f"Number of prediction arrays ({len(predictions_list)}) "
                f"does not match number of identities ({self._num_identities})."
            )
        if len(probabilities_list) != self._num_identities:
            raise ValueError(
                f"Number of probability arrays ({len(predictions_list)}) "
                f"does not match number of identities ({self._num_identities})."
            )

        for i, widget in enumerate(self._prediction_overview_widgets):
            widget.num_frames = self.num_frames
            widget.framerate = self.framerate
            widget.set_labels(predictions_list[i], probabilities_list[i])

    def set_search_results(
        self, behavior_search_query: BehaviorSearchQuery | None, search_results: list[SearchHit]
    ) -> None:
        """Set search results for timelines.

        Args:
            behavior_search_query: The BehaviorSearchQuery used to obtain the search results.
            search_results: List of SearchHit objects containing search results.
        """
        for i, label_overview_widget in enumerate(self._label_overview_widgets):
            curr_search_results: list[SearchHit] = []
            match behavior_search_query:
                case LabelBehaviorSearchQuery() | TimelineAnnotationSearchQuery():
                    for hit in search_results:
                        if hit.identity is None or hit.identity == str(i):
                            curr_search_results.append(hit)

            label_overview_widget.set_search_results(curr_search_results)

        for i, prediction_overview_widget in enumerate(self._prediction_overview_widgets):
            curr_search_results: list[SearchHit] = []
            match behavior_search_query:
                case PredictionBehaviorSearchQuery() | TimelineAnnotationSearchQuery():
                    for hit in search_results:
                        if hit.identity is None or hit.identity == str(i):
                            curr_search_results.append(hit)

            prediction_overview_widget.set_search_results(curr_search_results)

    def start_selection(self, starting_frame: int, ending_frame: int | None = None) -> None:
        """Start a selection from the given frame(s) on the active identity's widget.

        Records the starting frame and initiates selection mode on the currently active identity.
        If `ending_frame` is provided, it sets the selection range to include that frame as well.
        If `ending_frame` is None, the selection continues to the current frame.

        Args:
            starting_frame: The frame index where the selection begins.
            ending_frame: Optional; the frame index where the selection ends. If None,
                selection continues to current frame.
        """
        if self._active_identity_index is not None:
            self._selection_starting_frame = starting_frame
            self._selection_ending_frame = ending_frame
            self._label_overview_widgets[self._active_identity_index].start_selection(
                starting_frame,
                ending_frame,
            )
            self._label_overview_widgets[self._active_identity_index].update()

    def clear_selection(self) -> None:
        """Clear the current selection on the active identity's widget.

        Exits selection mode and resets the selection starting frame.
        """
        if self._active_identity_index is not None:
            self._label_overview_widgets[self._active_identity_index].clear_selection()
            self._selection_starting_frame = None
            self._selection_ending_frame = None
            self._label_overview_widgets[self._active_identity_index].update_labels()

    def reset(self) -> None:
        """Reset all child widgets to their initial state.

        Clears all internal data and resets the state of label and prediction overview widgets.
        """
        for widget in self._label_overview_widgets:
            widget.reset()

        for widget in self._prediction_overview_widgets:
            widget.reset()

    def _on_palette_changed(self) -> None:
        """Handle required updates if app palette changes."""
        self._update_frame_border()
