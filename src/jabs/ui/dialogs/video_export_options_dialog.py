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

    Args:
        parent: Parent widget.
        draw_pose: Initial state of the pose checkbox.
        draw_segmentation: Initial state of the segmentation checkbox.
        draw_predictions: Initial state of the predictions checkbox.
        segmentation_unavailable: Why segmentation cannot be drawn, or ``None`` when
            it can. A reason disables the checkbox and becomes its tooltip.
        predictions_unavailable: Why predictions cannot be drawn, or ``None`` when
            they can.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        draw_pose: bool = True,
        draw_segmentation: bool = True,
        draw_predictions: bool = False,
        segmentation_unavailable: str | None = None,
        predictions_unavailable: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Video with Overlays")

        self._pose_checkbox = QCheckBox("Pose skeleton", self)
        self._pose_checkbox.setChecked(draw_pose)

        self._segmentation_checkbox = QCheckBox("Segmentation contours", self)
        self._segmentation_checkbox.setChecked(draw_segmentation)

        self._predictions_checkbox = QCheckBox("Behavior predictions", self)
        self._predictions_checkbox.setChecked(draw_predictions)

        self._apply_availability(self._segmentation_checkbox, segmentation_unavailable)
        self._apply_availability(self._predictions_checkbox, predictions_unavailable)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        for checkbox in (
            self._pose_checkbox,
            self._segmentation_checkbox,
            self._predictions_checkbox,
        ):
            checkbox.toggled.connect(self._update_save_enabled)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Overlays to include in the exported video:", self))
        layout.addWidget(self._pose_checkbox)
        layout.addWidget(self._segmentation_checkbox)
        layout.addWidget(self._predictions_checkbox)
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
    def predictions_enabled(self) -> bool:
        """Whether the predictions checkbox was offered at all."""
        return self._predictions_checkbox.isEnabled()

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
            save_button.setEnabled(
                self.draw_pose or self.draw_segmentation or self.draw_predictions
            )
