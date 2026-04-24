from __future__ import annotations

from enum import IntEnum

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from jabs.behavior_search import (
    BehaviorSearchQuery,
    LabelBehaviorSearchQuery,
    PredictionBehaviorSearchQuery,
    SearchHit,
    TimelineAnnotationSearchQuery,
)
from jabs.core.enums import ClassifierMode
from jabs.pose_estimation import PoseEstimation

from ..colors import (
    BACKGROUND_COLOR,
    NOT_BEHAVIOR_COLOR,
    build_multiclass_color_lut,
    make_behavior_color_map,
)
from .frame_labels_widget import FrameLabelsWidget
from .label_overview_widget import LabelOverviewWidget, PredictionOverviewWidget


class _BehaviorLegendWidget(QWidget):
    """Horizontal strip showing a color swatch and name for each behavior class.

    Displayed above the per-identity timeline rows in multi-class mode to help
    readers identify which color corresponds to which behavior.  Always includes a
    "None" entry (blue) as the first item, matching the None prediction row.

    Args:
        behavior_names: Ordered list of behavior names to display.
        color_map: Mapping from behavior name to ``QColor``.
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.
    """

    _SWATCH_SIZE = 14

    def __init__(
        self,
        behavior_names: list[str],
        color_map: dict[str, QColor],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        # "None" / background class always comes first
        entries: list[tuple[str, QColor]] = [("None", NOT_BEHAVIOR_COLOR)] + [
            (name, color_map[name]) for name in behavior_names
        ]
        for name, color in entries:
            swatch = QLabel()
            swatch.setFixedSize(self._SWATCH_SIZE, self._SWATCH_SIZE)
            swatch.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")
            layout.addWidget(swatch)
            layout.addWidget(QLabel(name))
        layout.addStretch()


class StackedTimelineWidget(QWidget):
    """A widget that manages and displays multiple LabelOverviewWidgets, one for each identity.

    This widget allows toggling between showing all identities or only the active one,
    manages selection transfer between identities, and forwards label and frame updates
    to its child widgets. It is designed for use in multi-identity labeling interfaces,
    such as behavioral video annotation tools.

    In multi-class mode (set via :meth:`set_classifier_mode`), each identity displays a
    multi-color combined label bar and one :class:`PredictionOverviewWidget` per behavior
    class.  In binary mode the original single-behavior view is retained.

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

        self._active_identity_index: int | None = None
        self._selection_starting_frame: int | None = None
        self._selection_ending_frame: int | None = None
        self._view_mode = self.ViewMode.LABELS_AND_PREDICTIONS
        self._identity_mode = self.IdentityMode.ACTIVE
        self._num_identities = 0
        self._num_frames = 0
        self._framerate = 0
        self._label_overview_widgets: list[LabelOverviewWidget] = []
        self._combined_prediction_widgets: list[PredictionOverviewWidget | None] = []
        self._prediction_overview_widgets: list[list[PredictionOverviewWidget]] = []
        self._identity_frames: list[QFrame] = []
        self._frame_labels = FrameLabelsWidget(self)
        self._pose: PoseEstimation | None = None

        self._classifier_mode: ClassifierMode = ClassifierMode.BINARY
        self._behavior_names: list[str] = []
        self._behavior_color_map: dict[str, QColor] = {}
        self._multiclass_color_lut: npt.NDArray[np.uint8] | None = None
        self._legend_widget: _BehaviorLegendWidget | None = None
        self._collapse_inactive_label_bar: bool = False
        self._collapse_inactive_combined_bar: bool = False
        self._collapse_inactive_per_class_bars: bool = True

        self._layout: QVBoxLayout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # handle palette changes
        app = QApplication.instance()
        if app and hasattr(app, "paletteChanged"):
            app.paletteChanged.connect(self._on_palette_changed)

    def _label_overview_widget_factory(self, parent) -> LabelOverviewWidget:
        """Factory method to create a label overview widget."""
        widget = LabelOverviewWidget(parent, compact=self._identity_mode == self.IdentityMode.ALL)
        widget.num_frames = self.num_frames
        widget.framerate = self.framerate
        return widget

    def _prediction_overview_widget_factory(self, parent) -> PredictionOverviewWidget:
        """Factory method to create a prediction overview widget."""
        widget = PredictionOverviewWidget(
            parent, compact=self._identity_mode == self.IdentityMode.ALL
        )
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
    def collapse_inactive_label_bar(self) -> bool:
        """Whether the manual-label detail bar is collapsed for non-active identities.

        Only has a visual effect in multi-class + all-animals mode.
        """
        return self._collapse_inactive_label_bar

    @collapse_inactive_label_bar.setter
    def collapse_inactive_label_bar(self, value: bool) -> None:
        """Set whether non-active identity label bars are collapsed in all-animals mode."""
        if value != self._collapse_inactive_label_bar:
            self._collapse_inactive_label_bar = value
            self._update_widget_visibility()

    @property
    def collapse_inactive_combined_bar(self) -> bool:
        """Whether the combined argmax prediction bar is collapsed for non-active identities.

        Only has a visual effect in multi-class + all-animals mode.
        """
        return self._collapse_inactive_combined_bar

    @collapse_inactive_combined_bar.setter
    def collapse_inactive_combined_bar(self, value: bool) -> None:
        """Set whether non-active combined prediction bars are collapsed in all-animals mode."""
        if value != self._collapse_inactive_combined_bar:
            self._collapse_inactive_combined_bar = value
            self._update_widget_visibility()

    @property
    def collapse_inactive_per_class_bars(self) -> bool:
        """Whether per-class prediction detail bars are collapsed for non-active identities.

        Only has a visual effect in multi-class + all-animals mode.
        """
        return self._collapse_inactive_per_class_bars

    @collapse_inactive_per_class_bars.setter
    def collapse_inactive_per_class_bars(self, value: bool) -> None:
        """Set whether non-active per-class prediction bars are collapsed in all-animals mode."""
        if value != self._collapse_inactive_per_class_bars:
            self._collapse_inactive_per_class_bars = value
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

    def set_classifier_mode(self, mode: ClassifierMode, behavior_names: list[str]) -> None:
        """Set the classifier mode and rebuild the layout for multi-class or binary display.

        In multi-class mode, each identity gets one :class:`PredictionOverviewWidget` per
        behavior class, a multi-color label bar, and a behavior legend widget is added
        above the identity rows.  In binary mode the original single-widget view is used.

        Args:
            mode: The classifier mode (``BINARY`` or ``MULTICLASS``).
            behavior_names: Ordered list of project behavior names. Must not include the
                reserved ``"None"`` behavior.
        """
        self._classifier_mode = mode
        self._behavior_names = list(behavior_names)
        if mode == ClassifierMode.MULTICLASS and behavior_names:
            self._behavior_color_map = make_behavior_color_map(behavior_names)
            self._multiclass_color_lut = build_multiclass_color_lut(
                behavior_names, self._behavior_color_map
            )
        else:
            self._behavior_color_map = {}
            self._multiclass_color_lut = None
        self._reset_layout()

    def _build_none_lut(self) -> npt.NDArray[np.uint8]:
        """Build a 3-entry binary color LUT for the None/background prediction row.

        Returns:
            RGBA array of shape ``(3, 4)`` where index 0 = no pose (gray),
            index 1 = not None class (gray), index 2 = None class predicted (blue).
        """
        return np.array(
            [
                BACKGROUND_COLOR.getRgb(),
                BACKGROUND_COLOR.getRgb(),
                NOT_BEHAVIOR_COLOR.getRgb(),
            ],
            dtype=np.uint8,
        )

    def _build_class_lut(self, behavior_name: str) -> npt.NDArray[np.uint8]:
        """Build a 3-entry binary color LUT for a single behavior class prediction row.

        Args:
            behavior_name: Name of the behavior class.

        Returns:
            RGBA array of shape ``(3, 4)`` where index 0 = no pose (gray),
            index 1 = not this class (gray), index 2 = this class (behavior color).
        """
        color = self._behavior_color_map[behavior_name]
        return np.array(
            [
                BACKGROUND_COLOR.getRgb(),
                BACKGROUND_COLOR.getRgb(),
                color.getRgb(),
            ],
            dtype=np.uint8,
        )

    def _reset_layout(self) -> None:
        """Recreate the layout and child widgets for all identities.

        Removes existing frames and overview widgets, then creates new frames,
        label overview widgets, and prediction overview widgets for each identity.
        Updates the internal lists and layout, and sets the active identity index.

        This method is called when the number of identities changes or when the
        classifier mode changes.
        """
        # Remove old widgets and frames
        for frame in self._identity_frames:
            self._layout.removeWidget(frame)
            frame.setParent(None)
            frame.deleteLater()
        if self._legend_widget is not None:
            self._layout.removeWidget(self._legend_widget)
            self._legend_widget.setParent(None)
            self._legend_widget.deleteLater()
            self._legend_widget = None
        self._layout.removeWidget(self._frame_labels)
        self._label_overview_widgets = []
        self._combined_prediction_widgets = []
        self._prediction_overview_widgets = []
        self._identity_frames = []

        # Build the legend widget in multi-class mode
        if self._classifier_mode == ClassifierMode.MULTICLASS and self._behavior_names:
            self._legend_widget = _BehaviorLegendWidget(
                self._behavior_names, self._behavior_color_map, self
            )
            self._layout.addWidget(self._legend_widget)

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
            label_widget.setVisible(False)

            if (
                self._classifier_mode == ClassifierMode.MULTICLASS
                and self._behavior_names
                and self._multiclass_color_lut is not None
            ):
                label_widget.set_color_lut(self._multiclass_color_lut)
                # Combined argmax prediction bar: uses the full multiclass color LUT
                combined_pw = self._prediction_overview_widget_factory(frame)
                combined_pw.set_color_lut(self._multiclass_color_lut)
                combined_pw.setVisible(False)
                # Per-class detail bars (collapsible): None row first, then behaviors
                prediction_widgets: list[PredictionOverviewWidget] = []
                none_pw = self._prediction_overview_widget_factory(frame)
                none_pw.set_color_lut(self._build_none_lut())
                none_pw.setVisible(False)
                prediction_widgets.append(none_pw)
                for behavior_name in self._behavior_names:
                    pw = self._prediction_overview_widget_factory(frame)
                    pw.set_color_lut(self._build_class_lut(behavior_name))
                    pw.setVisible(False)
                    prediction_widgets.append(pw)
            else:
                combined_pw = None
                pw = self._prediction_overview_widget_factory(frame)
                pw.setVisible(False)
                prediction_widgets = [pw]

            vbox.addWidget(QLabel(f"{identity_display_name}:"))
            vbox.addWidget(label_widget)
            if combined_pw is not None:
                vbox.addWidget(combined_pw)
            for pw in prediction_widgets:
                vbox.addWidget(pw)

            self._label_overview_widgets.append(label_widget)
            self._combined_prediction_widgets.append(combined_pw)
            self._prediction_overview_widgets.append(prediction_widgets)
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
    def active_identity_index(self) -> int | None:
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
            if selection_frame is not None and self._active_identity_index is not None:
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
        """Rebuild the layout to show only the widgets appropriate for the current modes."""
        # Remove all widgets from the layout
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.hide()

        # Legend goes first in multi-class mode
        if self._legend_widget is not None:
            self._layout.addWidget(self._legend_widget)
            self._legend_widget.show()

        # Add only the widgets needed for the current mode
        if self._identity_mode == self.IdentityMode.ALL:
            for i, frame in enumerate(self._identity_frames):
                self._layout.addWidget(frame)
                frame.show()
                self._set_widget_visibility(
                    self._label_overview_widgets[i],
                    self._combined_prediction_widgets[i],
                    self._prediction_overview_widgets[i],
                    is_active=(i == self._active_identity_index),
                )
                self._label_overview_widgets[i].compact = True
                if self._combined_prediction_widgets[i] is not None:
                    self._combined_prediction_widgets[i].compact = True  # type: ignore[union-attr]
                for pw in self._prediction_overview_widgets[i]:
                    pw.compact = True
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
                self._combined_prediction_widgets[idx],
                self._prediction_overview_widgets[idx],
                is_active=True,
            )
            self._label_overview_widgets[idx].compact = False
            if self._combined_prediction_widgets[idx] is not None:
                self._combined_prediction_widgets[idx].compact = False  # type: ignore[union-attr]
            for pw in self._prediction_overview_widgets[idx]:
                pw.compact = False

        # Add FrameLabelsWidget last
        self._layout.addWidget(self._frame_labels)
        self._frame_labels.show()
        self._update_frame_border()

    def _set_widget_visibility(
        self,
        label_widget: LabelOverviewWidget,
        combined_widget: PredictionOverviewWidget | None,
        prediction_widgets: list[PredictionOverviewWidget],
        is_active: bool = True,
    ) -> None:
        """Set the visibility of label and prediction widgets based on the current view mode.

        Shows or hides the provided widgets according to the selected view mode (labels
        only, predictions only, or both).  In multi-class + all-animals mode the combined
        prediction bar is always fully expanded while the per-class detail bars are
        optionally collapsed on non-active identities.

        Args:
            label_widget: The LabelOverviewWidget to show or hide.
            combined_widget: The combined argmax PredictionOverviewWidget, or ``None`` in
                binary mode.
            prediction_widgets: The per-class PredictionOverviewWidgets to show or hide.
            is_active: Whether this identity is the active one (used for layout compaction).
        """
        show_labels = self._view_mode in (
            self.ViewMode.LABELS_AND_PREDICTIONS,
            self.ViewMode.LABELS,
        )
        show_preds = self._view_mode in (
            self.ViewMode.LABELS_AND_PREDICTIONS,
            self.ViewMode.PREDICTIONS,
        )
        label_widget.setVisible(show_labels)
        if combined_widget is not None:
            combined_widget.setVisible(show_preds)
        for pw in prediction_widgets:
            pw.setVisible(show_preds)

        # In multi-class + all-animals mode, optionally collapse detail bars on non-active
        # identities independently for each bar type.
        if (
            self._classifier_mode == ClassifierMode.MULTICLASS
            and self._identity_mode == self.IdentityMode.ALL
        ):
            label_widget.set_detail_bar_visible(is_active or not self._collapse_inactive_label_bar)
            if combined_widget is not None:
                combined_widget.set_detail_bar_visible(
                    is_active or not self._collapse_inactive_combined_bar
                )
            for pw in prediction_widgets:
                pw.set_detail_bar_visible(is_active or not self._collapse_inactive_per_class_bars)

    @Slot(int)
    def set_current_frame(self, current_frame: int) -> None:
        """Forward current frame to all LabelOverviewWidgets and PredictionOverviewWidgets."""
        for label_widget, combined_widget, pred_widgets in zip(
            self._label_overview_widgets,
            self._combined_prediction_widgets,
            self._prediction_overview_widgets,
            strict=True,
        ):
            label_widget.set_current_frame(current_frame)
            if combined_widget is not None:
                combined_widget.set_current_frame(current_frame)
            for pw in pred_widgets:
                pw.set_current_frame(current_frame)
        self._frame_labels.set_current_frame(current_frame)

    def set_labels(
        self,
        labels_list: list[npt.NDArray[np.int16]],
        masks_list: list[np.ndarray],
    ) -> None:
        """Set labels for all LabelOverviewWidgets.

        ``labels_list`` must contain pre-normalized LUT-index arrays.  Callers
        are responsible for converting raw ``TrackLabels`` or multi-class arrays
        before passing.

        Args:
            labels_list: List of class-index arrays, one per identity.
            masks_list: List of identity mask arrays, one per identity.
        """
        if len(labels_list) != self._num_identities:
            raise ValueError(
                f"Number of label arrays in labels_list ({len(labels_list)}) "
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
        self,
        predictions_list: list[list[npt.NDArray[np.int16]]],
        probabilities_list: list[list[npt.NDArray[np.floating]]],
    ) -> None:
        """Set predictions for all PredictionOverviewWidgets.

        In binary mode, each inner list contains exactly one array.  In multi-class mode,
        each inner list contains one array per behavior class, matching the number of
        PredictionOverviewWidgets created by :meth:`set_classifier_mode`.

        Args:
            predictions_list: Nested list of class-index arrays, shape
                ``(n_identities, n_classes_per_identity)``, each element ``(n_frames,)``.
            probabilities_list: Nested list of per-frame confidence arrays, matching
                ``predictions_list`` in shape.
        """
        if len(predictions_list) != self._num_identities:
            raise ValueError(
                f"Number of prediction arrays ({len(predictions_list)}) "
                f"does not match number of identities ({self._num_identities})."
            )
        if len(probabilities_list) != self._num_identities:
            raise ValueError(
                f"Number of probability arrays ({len(probabilities_list)}) "
                f"does not match number of identities ({self._num_identities})."
            )

        for i, pred_widgets in enumerate(self._prediction_overview_widgets):
            for j, widget in enumerate(pred_widgets):
                widget.num_frames = self.num_frames
                widget.framerate = self.framerate
                widget.set_labels(predictions_list[i][j], probabilities_list[i][j])

            combined_widget = self._combined_prediction_widgets[i]
            if combined_widget is not None:
                combined_pred, combined_prob = self._compute_combined_prediction(
                    predictions_list[i], probabilities_list[i]
                )
                combined_widget.num_frames = self.num_frames
                combined_widget.framerate = self.framerate
                combined_widget.set_labels(combined_pred, combined_prob)

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

        for i, pred_widgets in enumerate(self._prediction_overview_widgets):
            curr_search_results = []
            match behavior_search_query:
                case PredictionBehaviorSearchQuery() | TimelineAnnotationSearchQuery():
                    for hit in search_results:
                        if hit.identity is None or hit.identity == str(i):
                            curr_search_results.append(hit)
            combined_widget = self._combined_prediction_widgets[i]
            if combined_widget is not None:
                combined_widget.set_search_results(curr_search_results)
            for pw in pred_widgets:
                pw.set_search_results(curr_search_results)

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
        for combined_widget in self._combined_prediction_widgets:
            if combined_widget is not None:
                combined_widget.reset()
        for pred_widgets in self._prediction_overview_widgets:
            for widget in pred_widgets:
                widget.reset()

    @staticmethod
    def _compute_combined_prediction(
        per_class_preds: list[npt.NDArray[np.int16]],
        per_class_probs: list[npt.NDArray[np.floating]],
    ) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.floating]]:
        """Derive a combined argmax prediction array from per-class binary arrays.

        For each frame finds the class whose prediction == 2 (this class predicted)
        and maps it to the multiclass LUT index (class 0 → LUT index 1, class k → k+1).
        Frames where no class is predicted (no pose) remain at LUT index 0.

        Args:
            per_class_preds: Per-class binary arrays (0=no pose, 1=not this class, 2=this
                class), ordered as [None/background, behavior 0, behavior 1, ...].
            per_class_probs: Per-class probability arrays, same order.

        Returns:
            Tuple of (combined LUT-index array, combined probability array).
        """
        n_frames = per_class_preds[0].shape[0]
        combined_pred = np.zeros(n_frames, dtype=np.int16)
        combined_prob = np.zeros(n_frames, dtype=np.float32)
        for class_idx, (pred, prob) in enumerate(
            zip(per_class_preds, per_class_probs, strict=True)
        ):
            winner_mask = pred == 2
            combined_pred[winner_mask] = class_idx + 1
            combined_prob[winner_mask] = prob[winner_mask]
        return combined_pred, combined_prob

    def _on_palette_changed(self) -> None:
        """Handle required updates if app palette changes."""
        self._update_frame_border()
