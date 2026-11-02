"""Main window for jabs-viewer, a standalone player for videos and pose files."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from jabs.pose_estimation import PoseEstimation, get_pose_path, open_pose_file
from jabs.ui.dialogs import MessageDialog
from jabs.ui.player_widget import PlayerWidget
from jabs.video_reader import BlankFrameReader, VideoReader

from .frame_size import estimate_frame_size
from .frame_size_dialog import FrameSizeDialog

logger = logging.getLogger(__name__)

VIDEO_FILE_FILTER = "Video Files (*.mp4 *.avi);;All Files (*)"
POSE_FILE_FILTER = "Pose Files (*_pose_est_v*.h5);;All Files (*)"

# Background (blue, green, red) for playing a pose file without its video
POSE_ONLY_BACKGROUND = (40, 40, 40)


class ViewerWindow(QtWidgets.QMainWindow):
    """Window that plays a video or a pose file with pose and segmentation overlays.

    When the opened video has a JABS pose file beside it (``<video>_pose_est_v<N>.h5``)
    the pose skeletons and, when the pose file has them, segmentation contours are drawn
    over the video. A pose file can also be opened on its own, and is played over a
    solid background. Both overlays can be toggled from the View menu or the toolbar.
    """

    def __init__(self, app_name: str, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the viewer window.

        Args:
            app_name: Name shown in the window title.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._app_name = app_name
        self._pose_est: PoseEstimation | None = None
        self._last_dir: Path | None = None
        # set only while a pose file is open without a video
        self._pose_only_path: Path | None = None
        self._estimated_frame_size: tuple[int, int] | None = None
        self._frame_size: tuple[int, int] | None = None

        self.setWindowTitle(app_name)

        self._player = PlayerWidget()
        # there is no selected animal in the viewer, so every identity is drawn alike
        self._player.emphasize_active_identity = False
        self.setCentralWidget(self._player)

        self._open_action = QtGui.QAction("&Open Video...", self)
        self._open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self._prompt_open_video)

        self._open_pose_action = QtGui.QAction("Open &Pose File...", self)
        self._open_pose_action.triggered.connect(self._prompt_open_pose_file)

        quit_action = QtGui.QAction("&Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)

        self._pose_action = QtGui.QAction("Overlay &Pose", self)
        self._pose_action.setCheckable(True)
        self._pose_action.setChecked(True)
        self._pose_action.setShortcut(QtGui.QKeySequence("P"))
        self._pose_action.toggled.connect(self._set_pose_overlay)

        self._segmentation_action = QtGui.QAction("Overlay &Segmentation", self)
        self._segmentation_action.setCheckable(True)
        self._segmentation_action.setChecked(True)
        self._segmentation_action.setShortcut(QtGui.QKeySequence("S"))
        self._segmentation_action.toggled.connect(self._player.overlay_segmentation)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._open_action)
        file_menu.addAction(self._open_pose_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._pose_action)
        view_menu.addAction(self._segmentation_action)
        view_menu.addSeparator()
        self._frame_size_action = view_menu.addAction("Set &Frame Size...")
        self._frame_size_action.triggered.connect(self._prompt_frame_size)

        toolbar = self.addToolBar("Overlays")
        toolbar.setObjectName("overlay_toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        toolbar.addAction(self._open_action)
        toolbar.addAction(self._open_pose_action)
        toolbar.addSeparator()
        toolbar.addAction(self._pose_action)
        toolbar.addAction(self._segmentation_action)

        # Apply the initial overlay state to the player; the toggled signals only fire on
        # a change.
        self._set_pose_overlay(self._pose_action.isChecked())
        self._player.overlay_segmentation(self._segmentation_action.isChecked())
        self._update_overlay_controls()

        self.statusBar().showMessage("Open a video or pose file to begin (File menu)")

    @property
    def pose_est(self) -> PoseEstimation | None:
        """Pose estimation for the open video or pose file, or None when there is none."""
        return self._pose_est

    def open_file(self, path: Path) -> None:
        """Open a video or, for an ``.h5`` file, a pose file on its own.

        Args:
            path: Path to a video file or a JABS pose file.
        """
        if path.suffix.lower() == ".h5":
            self.open_pose_file(path)
        else:
            self.open_video(path)

    def open_video(self, path: Path) -> None:
        """Open a video, along with its pose file when one is found beside it.

        A pose file that exists but cannot be opened, or whose frame count does not match
        the video's, is reported to the user, and the video is still opened without
        overlays.

        Args:
            path: Path to the video file.
        """
        logger.info("Opening video %s", path)
        pose_est = None
        pose_path = self._find_pose_file(path)
        if pose_path is not None:
            try:
                pose_est = self._load_pose(pose_path)
            except Exception as e:
                logger.error("Unable to open pose file %s", pose_path, exc_info=True)
                MessageDialog.warning(
                    self,
                    title="Unable to Open Pose File",
                    message=f"Unable to open {pose_path.name}; showing the video without "
                    "pose overlays.",
                    details=str(e),
                )

        if pose_est is not None and not self._pose_matches_video(pose_est, pose_path, path):
            pose_status = f"{pose_path.name} does not match the video"
            pose_est = None
        elif pose_est is not None:
            pose_status = f"pose from {pose_path.name}"
        else:
            pose_status = "no pose file found"

        try:
            self._player.load_video(path, pose_est)
        except Exception as e:
            logger.error("Unable to open video %s", path, exc_info=True)
            MessageDialog.error(
                self,
                title="Unable to Open Video",
                message=f"Unable to open {path.name}.",
                details=str(e),
            )
            return

        self._show_opened(path, pose_est, pose_only=False)
        self.statusBar().showMessage(f"{path.name}: {pose_status}")

    def _pose_matches_video(
        self, pose_est: PoseEstimation, pose_path: Path, video_path: Path
    ) -> bool:
        """Check that a pose file has one frame per video frame, warning the user if not.

        The overlays index the pose by video frame, so a pose file with a different frame
        count would fail past its last frame. The labeling GUI rejects such a pair when a
        project is opened; the viewer plays the video without the pose instead.

        Returns:
            True if the frame counts match, or if the video's frame count can't be read
            (opening the video reports that error).
        """
        try:
            video_frames = self._video_frame_count(video_path)
        except Exception:
            return True
        if video_frames == pose_est.num_frames:
            return True
        logger.warning(
            "%s: video frame count (%d) does not match pose file frame count (%d)",
            video_path.name,
            video_frames,
            pose_est.num_frames,
        )
        MessageDialog.warning(
            self,
            title="Pose File Does Not Match Video",
            message=f"{pose_path.name} has {pose_est.num_frames} frames but {video_path.name} "
            f"has {video_frames}; showing the video without pose overlays.",
        )
        return False

    @staticmethod
    def _video_frame_count(video_path: Path) -> int:
        """Return the number of frames in a video file."""
        return VideoReader.get_nframes_from_file(video_path)

    def open_pose_file(self, path: Path) -> None:
        """Open a pose file on its own and play it over a solid background.

        The frames start at the size estimated from the pose (see
        :func:`estimate_frame_size`), which can be corrected with
        :meth:`set_frame_size`, and play at 30 fps, since pose files do not record a
        frame rate.

        Args:
            path: Path to a JABS pose file (``<name>_pose_est_v<N>.h5``).
        """
        logger.info("Opening pose file %s", path)
        try:
            pose_est = self._load_pose(path)
            frame_size = estimate_frame_size(pose_est)
            self._play_pose_only(path, pose_est, frame_size)
        except Exception as e:
            logger.error("Unable to open pose file %s", path, exc_info=True)
            MessageDialog.error(
                self,
                title="Unable to Open Pose File",
                message=f"Unable to open {path.name}.",
                details=str(e),
            )
            return

        self._show_opened(path, pose_est, pose_only=True)
        self._pose_only_path = path
        self._estimated_frame_size = frame_size
        self._frame_size = frame_size
        self._show_pose_only_status()

    def set_frame_size(self, width: int, height: int) -> None:
        """Change the frame size of the open pose file, keeping the current frame.

        Does nothing unless a pose file is open without a video.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        if self._pose_only_path is None or self._pose_est is None:
            return
        logger.info("Setting pose-only frame size to %dx%d", width, height)
        frame = self._player.current_frame
        self._play_pose_only(self._pose_only_path, self._pose_est, (width, height))
        self._player.seek_to_frame(frame)
        self._frame_size = (width, height)
        self._show_pose_only_status()

    def _play_pose_only(
        self, path: Path, pose_est: PoseEstimation, frame_size: tuple[int, int]
    ) -> None:
        """Load solid-color frames of the given size to play the pose over."""
        width, height = frame_size
        frames = BlankFrameReader(
            num_frames=pose_est.num_frames,
            fps=pose_est.fps,
            width=width,
            height=height,
            color=POSE_ONLY_BACKGROUND,
            filename=path.name,
        )
        self._player.load_blank_frames(frames, pose_est)

    def _show_pose_only_status(self) -> None:
        """Show the open pose file and its frame size in the status bar."""
        width, height = self._frame_size
        source = "estimated" if self._frame_size == self._estimated_frame_size else "set"
        self.statusBar().showMessage(
            f"{self._pose_only_path.name}: pose only, no video "
            f"({width}x{height} {source} frame size)"
        )

    def _prompt_frame_size(self) -> None:
        """Ask the user for the frame size of the open pose file and apply it."""
        if self._frame_size is None or self._estimated_frame_size is None:
            return
        dialog = FrameSizeDialog(self._frame_size, self._estimated_frame_size, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            width, height = dialog.frame_size
            if (width, height) != self._frame_size:
                self.set_frame_size(width, height)

    def _show_opened(
        self, path: Path, pose_est: PoseEstimation | None, *, pose_only: bool
    ) -> None:
        """Update the window for a newly opened file and its pose."""
        self._pose_est = pose_est
        self._last_dir = path.parent
        if not pose_only:
            self._pose_only_path = None
            self._estimated_frame_size = None
            self._frame_size = None
        self._update_overlay_controls(pose_only)
        self.setWindowTitle(f"{self._app_name} - {path.name}")

    @staticmethod
    def _find_pose_file(video_path: Path) -> Path | None:
        """Return the newest pose file beside the video, or None if there is none."""
        try:
            return get_pose_path(video_path)
        except ValueError:
            return None

    def _load_pose(self, pose_path: Path) -> PoseEstimation:
        """Open a pose file, showing a busy cursor since large files can take a moment."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            return open_pose_file(pose_path)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _prompt_open_video(self) -> None:
        """Ask the user for a video file and open it."""
        path = self._prompt_for_file("Open Video", VIDEO_FILE_FILTER)
        if path is not None:
            self.open_video(path)

    def _prompt_open_pose_file(self) -> None:
        """Ask the user for a pose file and open it on its own."""
        path = self._prompt_for_file("Open Pose File", POSE_FILE_FILTER)
        if path is not None:
            self.open_pose_file(path)

    def _prompt_for_file(self, caption: str, file_filter: str) -> Path | None:
        """Show a file dialog starting in the last opened file's directory."""
        start_dir = str(self._last_dir) if self._last_dir is not None else ""
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, start_dir, file_filter)
        return Path(file_name) if file_name else None

    def _update_overlay_controls(self, pose_only: bool = False) -> None:
        """Enable only the controls the open file can support.

        Args:
            pose_only: Whether a pose file is open without a video, which is the only
                case where the frame size can be set.
        """
        has_pose = self._pose_est is not None
        has_segmentation = has_pose and getattr(self._pose_est, "has_segmentation", False)
        self._pose_action.setEnabled(has_pose)
        self._segmentation_action.setEnabled(has_segmentation)
        self._frame_size_action.setEnabled(pose_only)

    def _set_pose_overlay(self, enabled: bool) -> None:
        """Show or hide the pose skeletons for every identity."""
        self._player.pose_overlay_mode = (
            PlayerWidget.PoseOverlayMode.ALL if enabled else PlayerWidget.PoseOverlayMode.NONE
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle playback shortcuts: space plays/pauses, arrow keys step frames.

        Args:
            event: The key press event.
        """
        match event.key():
            case QtCore.Qt.Key.Key_Space:
                self._player.toggle_play()
            case QtCore.Qt.Key.Key_Left:
                self._player.previous_frame()
            case QtCore.Qt.Key.Key_Right:
                self._player.next_frame()
            case _:
                super().keyPressEvent(event)
