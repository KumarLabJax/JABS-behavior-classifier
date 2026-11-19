"""Dialog for choosing which overlays go into an exported video."""

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class VideoExportOptionsDialog(QDialog):
    """Asks which overlays to burn into an exported copy of the current video.

    Shown before the file dialog: the overlays chosen here decide what is rendered,
    and the file dialog that follows only decides where it is written. Saving is
    blocked until at least one overlay is selected, since an export with none is just
    a slow re-encode of the source video.

    An overlay the loaded video cannot provide is shown unchecked and disabled, with
    the reason as its tooltip, rather than hidden: that way its absence is explained,
    and nobody waits out a full export to find the overlay they wanted missing.

    Manual labels and predictions are separate checkboxes rather than a choice
    between them: checking both draws a marker for each beside every animal, the way
    "View > Label Overlay > Labels and Predictions" does in the player.

    Args:
        parent: Parent widget.
        draw_pose: Initial state of the pose checkbox.
        draw_segmentation: Initial state of the segmentation checkbox.
        draw_labels: Initial state of the manual labels checkbox.
        draw_predictions: Initial state of the predictions checkbox.
        segmentation_unavailable: Why segmentation cannot be drawn, or ``None`` when
            it can. A reason disables the checkbox and becomes its tooltip.
        labels_unavailable: Why manual labels cannot be drawn, or ``None`` when they
            can.
        predictions_unavailable: Why predictions cannot be drawn, or ``None`` when
            they can.
        draw_bounding_boxes: Initial state of the bounding box checkbox.
        bounding_boxes_unavailable: Why bounding boxes cannot be drawn, or ``None``
            when they can.
        offer_bounding_boxes: Whether to show the bounding box checkbox at all.
        draw_landmarks: Initial state of the arena landmarks checkbox.
        landmarks_unavailable: Why the arena landmarks cannot be drawn, or ``None``
            when they can.
        offer_landmarks: Whether to show the arena landmarks checkbox at all.
        offer_labels: Whether to show the manual label and prediction checkboxes,
            which need a JABS project; jabs-viewer has none.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        draw_pose: bool = True,
        draw_segmentation: bool = True,
        draw_labels: bool = False,
        draw_predictions: bool = False,
        segmentation_unavailable: str | None = None,
        labels_unavailable: str | None = None,
        predictions_unavailable: str | None = None,
        draw_bounding_boxes: bool = False,
        bounding_boxes_unavailable: str | None = None,
        offer_bounding_boxes: bool = False,
        draw_landmarks: bool = False,
        landmarks_unavailable: str | None = None,
        offer_landmarks: bool = False,
        offer_labels: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Video with Overlays")

        self._pose_checkbox = self._make_checkbox("Pose skeleton", draw_pose)
        self._segmentation_checkbox = self._make_checkbox(
            "Segmentation contours", draw_segmentation
        )
        self._bounding_boxes_checkbox = self._make_checkbox("Bounding boxes", draw_bounding_boxes)
        self._landmarks_checkbox = self._make_checkbox(
            "Landmarks (corners, lixit, food hopper)", draw_landmarks
        )
        self._labels_checkbox = self._make_checkbox("Manual labels", draw_labels)
        self._predictions_checkbox = self._make_checkbox("Behavior predictions", draw_predictions)

        # One row per overlay, in layout order: (checkbox, why it is unavailable or
        # None, whether it is offered at all). Adding an overlay means adding a row
        # here plus its property, rather than touching every step below.
        self._overlays: tuple[tuple[QCheckBox, str | None, bool], ...] = (
            (self._pose_checkbox, None, True),
            (self._segmentation_checkbox, segmentation_unavailable, True),
            (self._bounding_boxes_checkbox, bounding_boxes_unavailable, offer_bounding_boxes),
            (self._landmarks_checkbox, landmarks_unavailable, offer_landmarks),
            (self._labels_checkbox, labels_unavailable, offer_labels),
            (self._predictions_checkbox, predictions_unavailable, offer_labels),
        )

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Overlays to include in the exported video:", self))
        for checkbox, unavailable_reason, offered in self._overlays:
            if offered:
                self._apply_availability(checkbox, unavailable_reason)
                layout.addWidget(checkbox)
            else:
                # A hidden checkbox is also unchecked and disabled, so it never
                # contributes an overlay and its `*_enabled` property reports it was
                # not offered.
                self._apply_availability(checkbox, "")
                checkbox.hide()
            checkbox.toggled.connect(self._update_save_enabled)
        layout.addWidget(self._button_box)

        self._update_save_enabled()

    @property
    def draw_pose(self) -> bool:
        """Whether the pose skeleton should be drawn."""
        return self._pose_checkbox.isChecked()

    @property
    def draw_segmentation(self) -> bool:
        """Whether segmentation contours should be drawn."""
        return self._segmentation_checkbox.isChecked()

    @property
    def draw_bounding_boxes(self) -> bool:
        """Whether bounding boxes should be drawn."""
        return self._bounding_boxes_checkbox.isChecked()

    @property
    def draw_landmarks(self) -> bool:
        """Whether the arena landmarks should be drawn."""
        return self._landmarks_checkbox.isChecked()

    @property
    def draw_labels(self) -> bool:
        """Whether manual labels should be drawn."""
        return self._labels_checkbox.isChecked()

    @property
    def draw_predictions(self) -> bool:
        """Whether behavior predictions should be drawn."""
        return self._predictions_checkbox.isChecked()

    @property
    def segmentation_enabled(self) -> bool:
        """Whether the segmentation checkbox was offered at all.

        Lets the caller persist the segmentation choice only when the user could
        actually make one, rather than saving the forced-off state of a video that
        has no segmentation data.
        """
        return self._segmentation_checkbox.isEnabled()

    @property
    def bounding_boxes_enabled(self) -> bool:
        """Whether the bounding box checkbox was offered and available."""
        return self._bounding_boxes_checkbox.isEnabled()

    @property
    def landmarks_enabled(self) -> bool:
        """Whether the arena landmarks checkbox was offered and available."""
        return self._landmarks_checkbox.isEnabled()

    @property
    def labels_enabled(self) -> bool:
        """Whether the manual labels checkbox was offered at all."""
        return self._labels_checkbox.isEnabled()

    @property
    def predictions_enabled(self) -> bool:
        """Whether the predictions checkbox was offered at all."""
        return self._predictions_checkbox.isEnabled()

    def _make_checkbox(self, label: str, checked: bool) -> QCheckBox:
        """Create an overlay checkbox with its initial state."""
        checkbox = QCheckBox(label, self)
        checkbox.setChecked(checked)
        return checkbox

    @staticmethod
    def _apply_availability(checkbox: QCheckBox, unavailable_reason: str | None) -> None:
        """Disable and uncheck a checkbox whose overlay is unavailable."""
        if unavailable_reason is None:
            return
        checkbox.setChecked(False)
        checkbox.setEnabled(False)
        checkbox.setToolTip(unavailable_reason)

    def _update_save_enabled(self) -> None:
        """Enable Save only while at least one overlay is selected."""
        save_button = self._button_box.button(QDialogButtonBox.StandardButton.Save)
        if save_button is not None:
            save_button.setEnabled(any(checkbox.isChecked() for checkbox, _, _ in self._overlays))
