"""Dialog for displaying technical information about a video and its pose file."""

import json
import logging
import re
from pathlib import Path

import h5py
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jabs.video_reader import VideoReader

logger = logging.getLogger(__name__)


class VideoInfoDialog(QDialog):
    """Dialog that displays technical information about a video file.

    This dialog will interrogate the video and pose files directly.

    Args:
        video_path: Absolute path to the video file.
        pose_path: Absolute path to the pose file.
        identity_count: Number of identities tracked in this video.
        parent: Parent widget for the dialog.
    """

    def __init__(
        self,
        video_path: Path,
        pose_path: Path,
        identity_count: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Video Info")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        title = QLabel(f"<b>{video_path.name}</b>")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 14pt;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # get info from video file
        try:
            with VideoReader(video_path) as reader:
                width, height = reader.dimensions
                num_frames = reader.num_frames
                fps = reader.fps

            duration_secs = num_frames / fps
            hours, remainder = divmod(int(duration_secs), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            form.addRow("Resolution:", QLabel(f"{width} × {height} px"))  # noqa: RUF001
            form.addRow("Frames:", QLabel(str(num_frames)))
            form.addRow("Frame rate:", QLabel(f"{fps} fps"))
            form.addRow("Duration:", QLabel(duration_str))
        except (OSError, ValueError):
            logger.exception("Could not open video file for info: %s", video_path)
            form.addRow("Video:", QLabel("Unable to read video file"))

        if identity_count is not None:
            form.addRow("Subjects:", QLabel(str(identity_count)))

        layout.addLayout(form)

        # Pose file section
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        pose_section_label = QLabel("<b>Pose</b>")
        layout.addWidget(pose_section_label)

        pose_form = QFormLayout()
        pose_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        pose_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        pose_form.addRow("File:", QLabel(pose_path.name))
        layout.addLayout(pose_form)

        # if this looks like a JABS style hdf5 pose file:
        #   look for static objects and the model_metadata_json attribute
        if re.search(r"pose_est_v\d+\.h5$", pose_path.name):
            try:
                with h5py.File(pose_path, "r") as pose_h5:
                    # show static objects, if present
                    if "static_objects" in pose_h5:
                        names = ", ".join(pose_h5["static_objects"].keys())
                        pose_form.addRow("Static objects:", QLabel(names))

                    raw_json = pose_h5["poseest"].attrs.get("model_metadata_json")
                    if raw_json:
                        if isinstance(raw_json, bytes):
                            raw_json = raw_json.decode("utf-8")
                        try:
                            # Enforce consistent formatting by round-tripping through json
                            formatted = json.dumps(json.loads(raw_json), indent=2)
                        except json.JSONDecodeError:
                            logger.exception(
                                "Pose file model metadata is not valid JSON: %s", pose_path
                            )
                        else:
                            metadata_label = QLabel("<b>Model Metadata</b>")
                            layout.addWidget(metadata_label)
                            text_view = QPlainTextEdit(formatted)
                            text_view.setReadOnly(True)
                            text_view.setMinimumHeight(150)
                            text_view.setFont(QFont("Monospace"))
                            layout.addWidget(text_view)

            except OSError:
                logger.exception("Could not open pose file for info: %s", pose_path)
            except KeyError as e:
                logger.exception("Missing expected key in pose file %s: %s", pose_path, e)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setDefault(True)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

    def sizeHint(self) -> QSize:
        """Provide size hint for the dialog.

        Returns:
            QSize indicating the recommended size for the dialog.
        """
        return QSize(600, 400)
