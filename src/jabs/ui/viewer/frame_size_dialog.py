"""Dialog for setting the frame size a pose file is played back at."""

from PySide6 import QtWidgets

from .frame_size import MAX_FRAME_DIMENSION


class FrameSizeDialog(QtWidgets.QDialog):
    """Asks for the width and height of the video a pose file was produced from.

    Pose files do not (yet) record the frame size, so the viewer estimates it; this
    lets the user correct the estimate.
    """

    def __init__(
        self,
        current_size: tuple[int, int],
        estimated_size: tuple[int, int],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the dialog.

        Args:
            current_size: (width, height) the spin boxes start at.
            estimated_size: (width, height) estimated from the pose coordinates, shown
                for reference and restored by the "Use Estimate" button.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Set Frame Size")
        self._estimated_size = estimated_size

        self._width_spin = self._make_spin_box(current_size[0])
        self._height_spin = self._make_spin_box(current_size[1])

        form = QtWidgets.QFormLayout()
        form.addRow("Width:", self._width_spin)
        form.addRow("Height:", self._height_spin)

        estimate_label = QtWidgets.QLabel(
            f"Estimated from pose coordinates: {estimated_size[0]} x {estimated_size[1]}"
        )
        use_estimate_button = QtWidgets.QPushButton("Use Estimate")
        use_estimate_button.clicked.connect(self._use_estimate)
        estimate_layout = QtWidgets.QHBoxLayout()
        estimate_layout.addWidget(estimate_label)
        estimate_layout.addStretch()
        estimate_layout.addWidget(use_estimate_button)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel("Size of the video frames the pose file was produced from:")
        )
        layout.addLayout(form)
        layout.addLayout(estimate_layout)
        layout.addWidget(buttons)

    @property
    def frame_size(self) -> tuple[int, int]:
        """The (width, height) currently entered."""
        return self._width_spin.value(), self._height_spin.value()

    @staticmethod
    def _make_spin_box(value: int) -> QtWidgets.QSpinBox:
        """Create a pixel-dimension spin box starting at the given value."""
        spin = QtWidgets.QSpinBox()
        spin.setRange(1, MAX_FRAME_DIMENSION)
        spin.setSuffix(" px")
        spin.setValue(value)
        return spin

    def _use_estimate(self) -> None:
        """Reset the spin boxes to the size estimated from the pose."""
        self._width_spin.setValue(self._estimated_size[0])
        self._height_spin.setValue(self._estimated_size[1])
