"""Tests for the background thread that writes an overlay video from the GUI."""

from pathlib import Path

import pytest

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    from jabs.ui import video_export_thread as vet
    from jabs.ui.video_export_thread import VideoExportThread

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


def _thread(tmp_path: Path, **kwargs) -> "VideoExportThread":
    return VideoExportThread(
        tmp_path / "in.avi", tmp_path / "out.mp4", object(), kwargs.pop("segmentation", True)
    )


def test_emits_complete_with_frame_count(monkeypatch, tmp_path: Path) -> None:
    """A successful export reports how many frames were written."""
    monkeypatch.setattr(vet, "export_overlay_video", lambda *a, **k: 42)
    thread = _thread(tmp_path)
    completed: list[int] = []
    thread.export_complete.connect(completed.append)

    thread.run()

    assert completed == [42]


def test_forwards_progress(monkeypatch, tmp_path: Path) -> None:
    """Per-frame progress from the exporter reaches the progress signal."""

    def fake_export(*_args, progress_callback=None, **_kwargs):
        for i in (1, 2, 3):
            progress_callback(i, 3)
        return 3

    monkeypatch.setattr(vet, "export_overlay_video", fake_export)
    thread = _thread(tmp_path)
    seen: list[int] = []
    thread.update_progress.connect(seen.append)

    thread.run()

    assert seen == [1, 2, 3]


def test_cancellation_emits_cancelled_not_complete(monkeypatch, tmp_path: Path) -> None:
    """Requesting termination stops the export and reports it as cancelled."""

    def fake_export(*_args, should_continue=None, **_kwargs):
        written = 0
        while should_continue():
            written += 1
            if written == 2:
                thread.request_termination()
        return written

    monkeypatch.setattr(vet, "export_overlay_video", fake_export)
    thread = _thread(tmp_path)
    cancelled: list[bool] = []
    completed: list[int] = []
    thread.export_cancelled.connect(lambda: cancelled.append(True))
    thread.export_complete.connect(completed.append)

    thread.run()

    assert cancelled == [True]
    assert completed == []


def test_errors_are_reported_not_raised(monkeypatch, tmp_path: Path) -> None:
    """An exporter failure reaches the error signal instead of killing the thread."""
    error = RuntimeError("codec exploded")

    def boom(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(vet, "export_overlay_video", boom)
    thread = _thread(tmp_path)
    errors: list[Exception] = []
    completed: list[int] = []
    thread.error_callback.connect(errors.append)
    thread.export_complete.connect(completed.append)

    thread.run()

    assert errors == [error]
    assert completed == []


def test_overlay_choices_are_passed_through(monkeypatch, tmp_path: Path) -> None:
    """Every checkbox in the options dialog reaches the exporter rather than being dropped."""
    captured: dict = {}

    def fake_export(*_args, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(vet, "export_overlay_video", fake_export)
    overlay = object()
    VideoExportThread(
        tmp_path / "in.avi",
        tmp_path / "out.mp4",
        object(),
        False,
        draw_pose=False,
        label_overlay=overlay,
    ).run()

    assert captured["draw_segmentation"] is False
    assert captured["draw_pose"] is False
    assert captured["label_overlay"] is overlay


def test_parent_keeps_its_positional_slot(tmp_path: Path) -> None:
    """The overlay options are keyword-only, so a positional parent is still a parent.

    Taking positional arguments for them would silently read a caller's ``parent``
    as ``draw_pose`` and leave the thread unparented.
    """
    thread = VideoExportThread(tmp_path / "in.avi", tmp_path / "out.mp4", object(), True, None)

    assert thread._draw_pose is True
    assert thread._label_overlay is None

    with pytest.raises(TypeError):
        VideoExportThread(tmp_path / "in.avi", tmp_path / "out.mp4", object(), True, None, False)


def test_defaults_draw_pose_and_no_label_markers(monkeypatch, tmp_path: Path) -> None:
    """The pose-only export that predates the options dialog still works unchanged."""
    captured: dict = {}

    def fake_export(*_args, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(vet, "export_overlay_video", fake_export)
    VideoExportThread(tmp_path / "in.avi", tmp_path / "out.mp4", object()).run()

    assert captured["draw_pose"] is True
    assert captured["draw_segmentation"] is True
    assert captured["label_overlay"] is None
