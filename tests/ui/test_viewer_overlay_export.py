"""Tests for jabs-viewer's frame and overlay video export."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

try:
    from PySide6.QtWidgets import QApplication, QDialog, QWidget

    from jabs.ui.viewer import overlay_export
    from jabs.ui.viewer.overlay_export import ExportSource, OverlayChoice, OverlayExporter

    ALL_OVERLAYS = OverlayChoice(pose=True, segmentation=True, bounding_boxes=True)

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)

VIDEO_PATH = Path("/data/video.mp4")
POSE_PATH = Path("/data/video_pose_est_v8.h5")


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def player() -> SimpleNamespace:
    """A stand-in player whose frames save successfully."""
    raw = mock.Mock()
    raw.save.return_value = True
    overlay = mock.Mock()
    overlay.save.return_value = True
    return SimpleNamespace(
        stop=mock.Mock(),
        current_frame=42,
        num_frames=100,
        get_raw_frame=mock.Mock(return_value=raw),
        get_overlay_frame=mock.Mock(return_value=overlay),
    )


@pytest.fixture
def exporter(player) -> "OverlayExporter":
    """An exporter with a real parent widget and a status spy."""
    return OverlayExporter(QWidget(), player, mock.Mock())


def _fake_pose(*, segmentation: bool = True, boxes: bool = True) -> mock.Mock:
    pose = mock.Mock()
    pose.has_segmentation = segmentation
    pose.has_bounding_boxes = boxes
    pose.format_major_version = 8 if boxes else 6
    return pose


def _patch_frame_dialog(
    monkeypatch,
    *,
    accepted: bool = True,
    selected: str = "/tmp/out.png",
    overlay_checked: bool = True,
) -> tuple[mock.Mock, mock.Mock]:
    """Replace the save dialog and its checkbox; return (checkbox class, dialog)."""
    dialog_cls = mock.MagicMock()
    dialog = dialog_cls.return_value
    dialog.exec.return_value = (
        QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
    )
    dialog.selectedFiles.return_value = [selected]
    dialog.layout.return_value = None
    monkeypatch.setattr(overlay_export.QtWidgets, "QFileDialog", dialog_cls)

    checkbox = mock.Mock()
    checkbox.isChecked.return_value = overlay_checked
    checkbox_cls = mock.Mock(return_value=checkbox)
    monkeypatch.setattr(overlay_export.QtWidgets, "QCheckBox", checkbox_cls)
    return checkbox_cls, dialog


def test_export_frame_saves_raw_frame_and_overlay_copy(monkeypatch, exporter, player) -> None:
    """A video frame is saved bare, plus a copy with the displayed overlays."""
    _patch_frame_dialog(monkeypatch, selected="/tmp/still")
    overlays = OverlayChoice(pose=True, segmentation=False, bounding_boxes=True)

    saved = exporter.export_frame(ExportSource(VIDEO_PATH, _fake_pose()), overlays)

    assert saved == Path("/tmp/still.png")
    player.stop.assert_called_once()
    player.get_raw_frame.assert_called_once_with(42)
    player.get_raw_frame.return_value.save.assert_called_once_with("/tmp/still.png", "PNG")
    player.get_overlay_frame.assert_called_once_with(
        42, draw_pose=True, draw_segmentation=False, draw_bounding_boxes=True
    )
    player.get_overlay_frame.return_value.save.assert_called_once_with(
        "/tmp/still-overlay.png", "PNG"
    )
    exporter._status.assert_called_once_with("Frame exported: /tmp/still.png (+ overlay copy)")


def test_export_frame_without_overlay_copy(monkeypatch, exporter, player) -> None:
    """Unticking the checkbox saves only the bare frame."""
    _patch_frame_dialog(monkeypatch, overlay_checked=False)

    exporter.export_frame(ExportSource(VIDEO_PATH, _fake_pose()), ALL_OVERLAYS)

    player.get_overlay_frame.assert_not_called()
    exporter._status.assert_called_once_with("Frame exported: /tmp/out.png")


def test_export_frame_of_video_without_pose_offers_no_overlay_copy(
    monkeypatch, exporter, player
) -> None:
    """With no pose there is nothing to overlay, so no checkbox is offered."""
    checkbox_cls, _dialog = _patch_frame_dialog(monkeypatch)

    exporter.export_frame(ExportSource(VIDEO_PATH, None), ALL_OVERLAYS)

    checkbox_cls.assert_not_called()
    player.get_overlay_frame.assert_not_called()


def test_export_frame_of_pose_only_saves_just_the_overlaid_frame(
    monkeypatch, exporter, player
) -> None:
    """The blank background alone is not worth a file, so the overlay is the export."""
    checkbox_cls, _dialog = _patch_frame_dialog(monkeypatch)
    source = ExportSource(POSE_PATH, _fake_pose(), frames=mock.Mock())

    exporter.export_frame(source, ALL_OVERLAYS)

    checkbox_cls.assert_not_called()
    player.get_raw_frame.assert_not_called()
    player.get_overlay_frame.return_value.save.assert_called_once_with("/tmp/out.png", "PNG")


def test_export_frame_cancelled_saves_nothing(monkeypatch, exporter, player) -> None:
    """Cancelling the save dialog writes nothing."""
    _patch_frame_dialog(monkeypatch, accepted=False)

    assert exporter.export_frame(ExportSource(VIDEO_PATH, _fake_pose()), ALL_OVERLAYS) is None

    player.get_raw_frame.assert_not_called()


def test_export_frame_reports_a_failed_save(monkeypatch, exporter, player) -> None:
    """A frame that cannot be written is reported as an error."""
    _patch_frame_dialog(monkeypatch)
    player.get_raw_frame.return_value.save.return_value = False
    message_dialog = mock.Mock()
    monkeypatch.setattr(overlay_export, "MessageDialog", message_dialog)

    assert exporter.export_frame(ExportSource(VIDEO_PATH, _fake_pose()), ALL_OVERLAYS) is None

    message_dialog.error.assert_called_once()
    exporter._status.assert_not_called()


def test_export_frame_starts_in_the_last_export_directory(monkeypatch, exporter, tmp_path) -> None:
    """The save dialog starts beside the source, then where the last export went."""
    _, dialog = _patch_frame_dialog(monkeypatch, selected=str(tmp_path / "a.png"))
    source = ExportSource(VIDEO_PATH, _fake_pose())

    exporter.export_frame(source, ALL_OVERLAYS)
    dialog.setDirectory.assert_called_with(str(VIDEO_PATH.parent))
    dialog.selectFile.assert_called_with("video_frame000042.png")

    exporter.export_frame(source, ALL_OVERLAYS)
    dialog.setDirectory.assert_called_with(str(tmp_path))


@pytest.fixture
def video_dialogs(monkeypatch):
    """Replace the options dialog, save dialog, thread and progress dialog."""
    options = mock.Mock()
    options.exec.return_value = QDialog.DialogCode.Accepted
    options.draw_pose = False
    options.draw_segmentation = True
    options.draw_bounding_boxes = True
    options_cls = mock.Mock(return_value=options)
    monkeypatch.setattr(overlay_export, "VideoExportOptionsDialog", options_cls)

    save_dialog = mock.MagicMock()
    save_dialog.getSaveFileName.return_value = ("/tmp/clip", "MP4 Video (*.mp4)")
    monkeypatch.setattr(overlay_export.QtWidgets, "QFileDialog", save_dialog)

    thread_cls = mock.Mock()
    monkeypatch.setattr(overlay_export, "VideoExportThread", thread_cls)
    monkeypatch.setattr(overlay_export.QtWidgets, "QProgressDialog", mock.MagicMock())
    return SimpleNamespace(
        options=options, options_cls=options_cls, save=save_dialog, thread_cls=thread_cls
    )


def test_export_video_starts_thread_with_chosen_overlays(exporter, video_dialogs) -> None:
    """The overlays chosen in the dialog, and the output path, reach the export thread."""
    pose = _fake_pose()

    assert exporter.export_video(ExportSource(VIDEO_PATH, pose), ALL_OVERLAYS)

    args, kwargs = video_dialogs.thread_cls.call_args
    assert args == (VIDEO_PATH, Path("/tmp/clip.mp4"), pose, True)
    assert kwargs["draw_pose"] is False
    assert kwargs["draw_bounding_boxes"] is True
    assert kwargs["frames"] is None
    video_dialogs.thread_cls.return_value.start.assert_called_once()
    start = video_dialogs.save.getSaveFileName.call_args.args[2]
    assert start == str(VIDEO_PATH.parent / "video_overlay.mp4")


def test_export_video_offers_boxes_but_not_labels(exporter, video_dialogs) -> None:
    """The dialog starts from the displayed overlays and has no project-only options."""
    overlays = OverlayChoice(pose=True, segmentation=False, bounding_boxes=True)

    exporter.export_video(ExportSource(VIDEO_PATH, _fake_pose()), overlays)

    kwargs = video_dialogs.options_cls.call_args.kwargs
    assert kwargs["draw_pose"] is True
    assert kwargs["draw_segmentation"] is False
    assert kwargs["draw_bounding_boxes"] is True
    assert kwargs["offer_bounding_boxes"] is True
    assert kwargs["offer_labels"] is False
    assert kwargs["segmentation_unavailable"] is None
    assert kwargs["bounding_boxes_unavailable"] is None


def test_export_video_explains_missing_segmentation_and_boxes(exporter, video_dialogs) -> None:
    """A pose file without segmentation or boxes disables those options with a reason."""
    exporter.export_video(
        ExportSource(VIDEO_PATH, _fake_pose(segmentation=False, boxes=False)), ALL_OVERLAYS
    )

    kwargs = video_dialogs.options_cls.call_args.kwargs
    assert kwargs["segmentation_unavailable"]
    assert "version 8" in kwargs["bounding_boxes_unavailable"]


def test_export_video_of_pose_only_draws_over_the_given_frames(exporter, video_dialogs) -> None:
    """A pose file on its own is exported over its blank frames, named after the pose."""
    frames = mock.Mock()

    exporter.export_video(ExportSource(POSE_PATH, _fake_pose(), frames=frames), ALL_OVERLAYS)

    args, kwargs = video_dialogs.thread_cls.call_args
    assert args[0] == POSE_PATH
    assert kwargs["frames"] is frames
    start = video_dialogs.save.getSaveFileName.call_args.args[2]
    assert start.endswith("video_pose_est_v8_overlay.mp4")


def test_export_video_cancelled_starts_nothing(exporter, video_dialogs) -> None:
    """Cancelling the options or the save dialog starts no export."""
    video_dialogs.options.exec.return_value = QDialog.DialogCode.Rejected
    assert not exporter.export_video(ExportSource(VIDEO_PATH, _fake_pose()), ALL_OVERLAYS)

    video_dialogs.options.exec.return_value = QDialog.DialogCode.Accepted
    video_dialogs.save.getSaveFileName.return_value = ("", "")
    assert not exporter.export_video(ExportSource(VIDEO_PATH, _fake_pose()), ALL_OVERLAYS)

    video_dialogs.thread_cls.assert_not_called()


def test_export_video_without_pose_warns(monkeypatch, exporter, video_dialogs) -> None:
    """There is nothing to draw on a video with no pose."""
    message_dialog = mock.Mock()
    monkeypatch.setattr(overlay_export, "MessageDialog", message_dialog)

    assert not exporter.export_video(ExportSource(VIDEO_PATH, None), ALL_OVERLAYS)

    message_dialog.warning.assert_called_once()
    video_dialogs.options_cls.assert_not_called()
