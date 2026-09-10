"""Background thread for writing an overlay video from the GUI."""

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from jabs.pose_estimation import PoseEstimation
from jabs.video_export import export_overlay_video


class VideoExportThread(QThread):
    """Writes a pose-overlay video in the background, keeping the GUI responsive.

    Signals:
        export_complete: Emitted with the number of frames written when the export
            finishes normally.
        export_cancelled: Emitted when the user cancelled; the partial output file
            has already been removed by then.
        update_progress: Emitted with the number of frames written so far.
        error_callback: Emitted with the exception if the export fails.

    Args:
        video_path: Source video to read.
        output_path: Destination video to write.
        pose_est: Pose estimation for ``video_path``.
        draw_segmentation: Whether to include segmentation contours.
        parent: Optional parent widget.
    """

    export_complete = Signal(int)
    export_cancelled = Signal()
    update_progress = Signal(int)
    error_callback = Signal(Exception)

    def __init__(
        self,
        video_path: Path,
        output_path: Path,
        pose_est: PoseEstimation,
        draw_segmentation: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._video_path = video_path
        self._output_path = output_path
        self._pose_est = pose_est
        self._draw_segmentation = draw_segmentation
        self._should_terminate = False

    def request_termination(self) -> None:
        """Ask the export to stop at the next frame boundary.

        Safe to call from the GUI thread: assignment to a bool is atomic in CPython,
        matching how the training and classification threads handle cancellation.
        """
        self._should_terminate = True

    def run(self) -> None:
        """Thread entry point: write the overlay video, reporting progress."""
        try:
            frames_written = export_overlay_video(
                self._video_path,
                self._output_path,
                self._pose_est,
                draw_segmentation=self._draw_segmentation,
                progress_callback=lambda written, _total: self.update_progress.emit(written),
                should_continue=lambda: not self._should_terminate,
            )
        except Exception as e:
            self.error_callback.emit(e)
            return

        if self._should_terminate:
            self.export_cancelled.emit()
        else:
            self.export_complete.emit(frames_written)
