"""Dialog for displaying technical information about a video and its pose file."""

import json
import logging
import re
from pathlib import Path

import h5py
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontDatabase
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

from jabs.project import VideoFeatureCacheStatus
from jabs.ui.feature_cache_text import (
    format_byte_size,
    format_cache_formats,
    format_window_sizes,
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
        feature_cache_status: Status of this video's cached features. When
            ``None``, the feature cache section reports that the status could
            not be determined.
        parent: Parent widget for the dialog.
    """

    def __init__(
        self,
        video_path: Path,
        pose_path: Path,
        identity_count: int | None = None,
        feature_cache_status: VideoFeatureCacheStatus | None = None,
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

        # Model metadata is rendered after the feature cache section: it is the
        # tallest element, so it belongs at the bottom of the dialog.
        model_metadata: str | None = None

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
                            model_metadata = formatted

            except OSError:
                logger.exception("Could not open pose file for info: %s", pose_path)
                pose_form.addRow("Pose file:", QLabel("Unable to read pose file"))
            except KeyError as e:
                logger.exception("Missing expected key in pose file %s: %s", pose_path, e)
                pose_form.addRow("Pose file:", QLabel("Unable to parse pose file"))

        self._add_feature_cache_section(layout, feature_cache_status)

        if model_metadata is not None:
            layout.addWidget(QLabel("<b>Model Metadata</b>"))
            text_view = QPlainTextEdit(model_metadata)
            text_view.setReadOnly(True)
            text_view.setMinimumHeight(150)
            text_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
            layout.addWidget(text_view)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setDefault(True)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _path_label(path: Path) -> QLabel:
        """Build a selectable, wrapping label for a filesystem path."""
        label = QLabel(str(path))
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _add_feature_cache_section(
        self, layout: QVBoxLayout, status: VideoFeatureCacheStatus | None
    ) -> None:
        """Add a section describing this video's cached features.

        Args:
            layout: Dialog layout to append the section to.
            status: The video's feature cache status, or ``None`` when it could
                not be determined.
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        layout.addWidget(QLabel("<b>Feature Cache</b>"))

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addLayout(form)

        if status is None:
            form.addRow("Status:", QLabel("Unable to determine"))
            return

        form.addRow("Directory:", self._path_label(status.cache_dir))

        if not status.has_cached_features:
            form.addRow("Cached features:", QLabel("None"))
            return

        identities = str(status.cached_identity_count)
        if status.expected_identity_count is not None:
            identities = f"{status.cached_identity_count} of {status.expected_identity_count}"
            if not status.is_complete:
                identities += " (incomplete)"
        form.addRow("Identities cached:", QLabel(identities))

        form.addRow("Window sizes:", QLabel(format_window_sizes(status.window_sizes)))
        if status.partial_window_sizes:
            form.addRow(
                "Partial window sizes:",
                QLabel(
                    f"{format_window_sizes(status.partial_window_sizes)} "
                    "(cached for some identities only)"
                ),
            )

        form.addRow("Format:", QLabel(format_cache_formats(status.cache_formats)))

        versions = ", ".join(str(version) for version in status.feature_versions)
        if status.is_stale:
            versions += f" (out of date, current is {status.current_feature_version})"
        form.addRow("Feature version:", QLabel(versions))

        if status.cm_units is not None:
            form.addRow("Distance units:", QLabel("cm" if status.cm_units else "pixels"))

        form.addRow("Size on disk:", QLabel(format_byte_size(status.size_bytes)))

        if missing_per_frame := status.identities_missing_per_frame:
            identity_list = ", ".join(str(identity) for identity in missing_per_frame)
            form.addRow(
                "Warning:",
                QLabel(f"Per-frame features are missing for identities: {identity_list}"),
            )

    def sizeHint(self) -> QSize:
        """Provide size hint for the dialog.

        Returns:
            QSize indicating the recommended size for the dialog.
        """
        return QSize(600, 400)
