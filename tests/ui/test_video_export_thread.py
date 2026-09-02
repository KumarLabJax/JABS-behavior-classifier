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


def test_segmentation_flag_is_passed_through(monkeypatch, tmp_path: Path) -> None:
    """The GUI checkbox reaches the exporter rather than being dropped."""
    captured: dict = {}

    def fake_export(*_args, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(vet, "export_overlay_video", fake_export)
    VideoExportThread(tmp_path / "in.avi", tmp_path / "out.mp4", object(), False).run()

    assert captured["draw_segmentation"] is False
