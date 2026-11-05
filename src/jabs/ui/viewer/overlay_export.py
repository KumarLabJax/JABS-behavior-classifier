"""Frame and overlay-video export for jabs-viewer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from jabs.overlay_drawing import pose_has_bounding_boxes
from jabs.ui.dialogs import MessageDialog
from jabs.ui.dialogs.video_export_options_dialog import VideoExportOptionsDialog
from jabs.ui.video_export_thread import VideoExportThread

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation
    from jabs.ui.player_widget import PlayerWidget
    from jabs.video_reader import BlankFrameReader

logger = logging.getLogger(__name__)

_FRAME_FILTER = "PNG Images (*.png)"
_VIDEO_FILTER = "MP4 Video (*.mp4)"


@dataclass(frozen=True)
class OverlayChoice:
    """Which overlays to draw on an exported frame or video.

    Attributes:
        pose: Draw the pose keypoints and skeleton.
        segmentation: Draw segmentation contours (when the pose file has them).
        bounding_boxes: Draw bounding boxes (when the pose file has them).
    """

    pose: bool
    segmentation: bool
    bounding_boxes: bool


@dataclass(frozen=True)
class ExportSource:
    """What the viewer is showing, as needed to export it.

    Attributes:
        path: The open video, or the pose file when one is open without a video.
        pose_est: Pose for the open file, or None when a video has no pose.
        frames: Solid-color frames the pose is drawn over when there is no video,
            or None for a video. Must be a reader of its own, not the player's, since
            a video export reads it from a background thread.
    """

    path: Path
    pose_est: PoseEstimation | None
    frames: BlankFrameReader | None = None

    @property
    def pose_only(self) -> bool:
        """Whether a pose file is open without a video."""
        return self.frames is not None


class OverlayExporter(QtCore.QObject):
    """Exports the viewer's current frame, or the whole video, with overlays drawn.

    Mirrors the labeling GUI's File > Export Frame and Export Video with Overlays,
    without the label and prediction overlays, which need a JABS project. With a pose
    file open on its own, exports draw the pose over the same solid background the
    viewer plays, and a frame export saves only the overlaid frame, since the bare
    background is not worth a file.

    Args:
        parent: Window that owns the dialogs.
        player: Player whose frames are exported.
        status: Callback that shows a message in the window's status bar.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        player: PlayerWidget,
        status: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self._window = parent
        self._player = player
        self._status = status
        self._last_dir: Path | None = None
        self._save_overlay_copy = True
        self._video_export_thread: VideoExportThread | None = None

    def export_frame(self, source: ExportSource, overlays: OverlayChoice) -> Path | None:
        """Ask where to save the current frame as a PNG, and save it.

        Args:
            source: What the viewer is showing.
            overlays: Overlays to draw on the ``-overlay.png`` copy, or on the only
                copy when a pose file is open without a video.

        Returns:
            Path of the main image written, or None if cancelled or it failed.
        """
        self._player.stop()
        frame_number = self._player.current_frame
        suggested = f"{source.path.stem}_frame{frame_number:06d}.png"

        # Qt's own dialog, since a native one cannot host the extra checkbox.
        dialog = QtWidgets.QFileDialog(self._window, "Export Frame")
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter(_FRAME_FILTER)
        dialog.setDefaultSuffix("png")
        dialog.setDirectory(str(self._start_dir(source)))
        dialog.selectFile(suggested)

        overlay_checkbox = None
        if source.pose_est is not None and not source.pose_only:
            overlay_checkbox = QtWidgets.QCheckBox(
                "Also save a copy with the displayed overlays (…-overlay.png)"
            )
            overlay_checkbox.setChecked(self._save_overlay_copy)
            layout = dialog.layout()
            if isinstance(layout, QtWidgets.QGridLayout):
                layout.addWidget(overlay_checkbox, layout.rowCount(), 0, 1, layout.columnCount())

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        selected = dialog.selectedFiles()
        if not selected:
            return None
        save_path = Path(selected[0])
        if save_path.suffix.lower() != ".png":
            save_path = save_path.with_name(save_path.name + ".png")
        self._last_dir = save_path.parent

        if source.pose_only:
            pixmap = self._overlay_pixmap(frame_number, overlays)
        else:
            pixmap = self._player.get_raw_frame(frame_number)
        if pixmap is None:
            MessageDialog.warning(self._window, message="No frame available to export.")
            return None
        if not pixmap.save(str(save_path), "PNG"):
            MessageDialog.error(self._window, message=f"Failed to export frame to:\n{save_path}")
            return None
        logger.info("Exported frame %d to %s", frame_number, save_path)
        status = f"Frame exported: {save_path}"

        if overlay_checkbox is not None:
            self._save_overlay_copy = overlay_checkbox.isChecked()
            if self._save_overlay_copy:
                overlay_path = save_path.with_name(f"{save_path.stem}-overlay.png")
                overlay_pixmap = self._overlay_pixmap(frame_number, overlays)
                if overlay_pixmap is None or not overlay_pixmap.save(str(overlay_path), "PNG"):
                    MessageDialog.error(
                        self._window,
                        message=f"Failed to export overlay frame to:\n{overlay_path}",
                    )
                else:
                    status += " (+ overlay copy)"

        self._status(status)
        return save_path

    def export_video(self, source: ExportSource, overlays: OverlayChoice) -> bool:
        """Ask which overlays to draw and where to write, then export in the background.

        Args:
            source: What the viewer is showing; must have a pose.
            overlays: Initial overlay choices, normally what the viewer is showing.

        Returns:
            True if an export was started, False if cancelled or there is nothing to
            export.
        """
        if self._video_export_thread is not None:
            # a cancelled export keeps running until its next frame boundary
            MessageDialog.warning(self._window, message="A video export is still in progress.")
            return False
        pose_est = source.pose_est
        if pose_est is None:
            MessageDialog.warning(self._window, message="There is no pose to draw over the video.")
            return False
        self._player.stop()

        options = VideoExportOptionsDialog(
            self._window,
            draw_pose=overlays.pose,
            draw_segmentation=overlays.segmentation,
            draw_bounding_boxes=overlays.bounding_boxes,
            segmentation_unavailable=(
                None
                if getattr(pose_est, "has_segmentation", False)
                else "This pose file has no segmentation data"
            ),
            bounding_boxes_unavailable=(
                None
                if pose_has_bounding_boxes(pose_est)
                else "This pose file has no bounding boxes (optional, pose version 8 or newer)"
            ),
            offer_bounding_boxes=True,
            offer_labels=False,
        )
        if options.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False

        start = self._start_dir(source) / f"{source.path.stem}_overlay.mp4"
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._window, "Export Video with Overlays", str(start), _VIDEO_FILTER
        )
        if not selected:
            return False
        output_path = Path(selected)
        if output_path.suffix.lower() != ".mp4":
            # append rather than with_suffix(), which would replace a dotted name's tail
            output_path = output_path.with_name(output_path.name + ".mp4")
        self._last_dir = output_path.parent

        logger.info("Exporting overlay video of %s to %s", source.path, output_path)
        thread = VideoExportThread(
            source.path,
            output_path,
            pose_est,
            options.draw_segmentation,
            parent=self._window,
            draw_pose=options.draw_pose,
            draw_bounding_boxes=options.draw_bounding_boxes,
            frames=source.frames,
        )
        self._video_export_thread = thread
        self._run_video_export(thread, output_path)
        return True

    def _run_video_export(self, thread: VideoExportThread, output_path: Path) -> None:
        """Show a cancelable progress dialog for the export thread and start it."""
        progress = QtWidgets.QProgressDialog(
            f"Exporting {output_path.name}", "Cancel", 0, self._player.num_frames, self._window
        )
        progress.setWindowTitle("Export Video")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def close_progress() -> None:
            progress.close()
            progress.deleteLater()

        def on_complete(frames_written: int) -> None:
            close_progress()
            logger.info("Exported %d frames to %s", frames_written, output_path)
            self._status(f"Exported {frames_written:,} frames to {output_path}")

        def on_cancelled() -> None:
            close_progress()
            self._status("Video export cancelled")

        def on_error(error: Exception) -> None:
            close_progress()
            logger.error("Video export to %s failed: %s", output_path, error, exc_info=error)
            MessageDialog.error(
                self._window,
                title="Video Export Failed",
                message=f"Unable to export the overlay video to:\n{output_path}",
                details=f"{type(error).__name__}: {error}",
            )

        def release_thread() -> None:
            if self._video_export_thread is thread:
                self._video_export_thread = None

        # The thread deletes itself once finished; these callbacks run while run() is
        # still on its stack, and destroying an unfinished QThread aborts.
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(release_thread)
        thread.update_progress.connect(progress.setValue)
        thread.export_complete.connect(on_complete)
        thread.export_cancelled.connect(on_cancelled)
        thread.error_callback.connect(on_error)
        progress.canceled.connect(thread.request_termination)
        progress.show()
        thread.start()

    def shutdown(self) -> None:
        """Cancel any running video export and wait for its thread to finish.

        Call before the parent window is destroyed: the thread is its child, and
        destroying a running QThread aborts the process.
        """
        thread = self._video_export_thread
        if thread is not None and thread.isRunning():
            logger.info("Cancelling video export before the viewer closes")
            thread.request_termination()
            thread.wait()

    def _overlay_pixmap(self, frame_number: int, overlays: OverlayChoice) -> QtGui.QPixmap | None:
        """Render one frame with the chosen overlays at native resolution."""
        return self._player.get_overlay_frame(
            frame_number,
            draw_pose=overlays.pose,
            draw_segmentation=overlays.segmentation,
            draw_bounding_boxes=overlays.bounding_boxes,
        )

    def _start_dir(self, source: ExportSource) -> Path:
        """Directory a save dialog starts in: the last export's, else the source's."""
        if self._last_dir is not None and self._last_dir.is_dir():
            return self._last_dir
        return source.path.parent
