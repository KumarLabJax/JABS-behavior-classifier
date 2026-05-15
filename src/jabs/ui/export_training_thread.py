"""Background thread for exporting classifier training data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from jabs.core.constants import FINAL_TRAIN_SEED
from jabs.core.enums import ClassifierMode, ClassifierType
from jabs.project import export_training_data, export_training_data_multiclass

if TYPE_CHECKING:
    from jabs.project import Project


class ExportTrainingDataThread(QThread):
    """Thread for exporting classifier training data without blocking the GUI.

    Signals:
        export_complete: Emitted with the output file path on success.
        error_callback: Emitted with the raised exception on failure.
        current_status: Emitted with a status string for the status bar.
    """

    export_complete: Signal = Signal(Path)
    error_callback: Signal = Signal(Exception)
    current_status: Signal = Signal(str)

    def __init__(
        self,
        project: Project,
        pose_version: int,
        classifier_type: ClassifierType,
        classifier_mode: ClassifierMode = ClassifierMode.BINARY,
        behavior: str | None = None,
        training_seed: int = FINAL_TRAIN_SEED,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the export thread.

        Args:
            project: The JABS project to export training data from.
            pose_version: Minimum required pose version for the classifier.
            classifier_type: Classifier algorithm type for the export metadata.
            classifier_mode: Whether to export in binary or multi-class format.
            behavior: Name of the behavior to export. Required when
                ``classifier_mode`` is ``BINARY``; ignored for ``MULTICLASS``.
            training_seed: Random seed for reproducible training splits.
            parent: Optional parent widget.
        """
        super().__init__(parent=parent)
        self._project = project
        self._behavior = behavior
        self._pose_version = pose_version
        self._classifier_type = classifier_type
        self._classifier_mode = classifier_mode
        self._training_seed = training_seed

    def run(self) -> None:
        """Run the export in the background thread."""
        try:
            self.current_status.emit("Exporting training data...")
            if self._classifier_mode == ClassifierMode.MULTICLASS:
                out_path = export_training_data_multiclass(
                    self._project,
                    self._pose_version,
                    self._classifier_type,
                    self._training_seed,
                )
            else:
                if self._behavior is None:
                    raise ValueError("behavior is required for binary training-data export")
                out_path = export_training_data(
                    self._project,
                    self._behavior,
                    self._pose_version,
                    self._classifier_type,
                    self._training_seed,
                )
            self.export_complete.emit(out_path)
        except Exception as e:
            self.error_callback.emit(e)
