"""Background thread that opens a pose file for jabs-viewer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal, SignalInstance

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

logger = logging.getLogger(__name__)


class PoseLoaderThread(QThread):
    """Opens a pose file off the GUI thread, so the viewer stays responsive.

    Opening a pose file can take several seconds: a v6+ file with segmentation
    decompresses all of its contours up front, several GB for an hour of video.

    Signals:
        pose_loaded: Emitted with the opened ``PoseEstimation``.
        load_error: Emitted with the exception if the file could not be opened.

    Args:
        pose_path: Pose file to open.
        loader: Function that opens a pose file, normally
            :func:`jabs.pose_estimation.open_pose_file`.
        parent: Optional parent object.
    """

    pose_loaded: SignalInstance = Signal(object)
    load_error: SignalInstance = Signal(Exception)

    def __init__(
        self,
        pose_path: Path,
        loader: Callable[[Path], PoseEstimation],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._pose_path = pose_path
        self._loader = loader

    @property
    def pose_path(self) -> Path:
        """The pose file this thread opens."""
        return self._pose_path

    def run(self) -> None:
        """Thread entry point: open the pose file and emit the result."""
        logger.info("Loading pose file %s", self._pose_path)
        try:
            pose_est = self._loader(self._pose_path)
        except Exception as e:
            # reported to the user by the window, which owns the dialogs
            self.load_error.emit(e)
            return
        logger.info("Loaded pose file %s", self._pose_path)
        self.pose_loaded.emit(pose_est)
