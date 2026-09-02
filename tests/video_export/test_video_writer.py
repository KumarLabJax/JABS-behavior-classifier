"""Tests for writing a video with the pose overlay burned in."""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    from jabs.video_export import DEFAULT_CODEC, VideoExportError, export_overlay_video
    from jabs.video_export import video_writer as video_writer_module

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

from ._fakes import BACKGROUND, FRAMES, HEIGHT, WIDTH, StubPose


def test_export_overlay_video_writes_every_frame(source_video: Path, tmp_path: Path) -> None:
    """The exported video has the same frame count and dimensions as the source."""
    output = tmp_path / "out.mp4"

    written = export_overlay_video(source_video, output, StubPose(), draw_segmentation=False)

    assert written == FRAMES
    assert output.exists()

    capture = cv2.VideoCapture(str(output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAMES
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == WIDTH
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == HEIGHT
        _, frame = capture.read()
        assert (np.abs(frame.astype(int) - BACKGROUND) > 25).any(), "no overlay in output"
    finally:
        capture.release()


def test_export_overlay_video_reports_progress(source_video: Path, tmp_path: Path) -> None:
    """Progress is reported once per frame, counting up to the total."""
    seen: list[tuple[int, int]] = []

    export_overlay_video(
        source_video,
        tmp_path / "out.mp4",
        StubPose(),
        draw_segmentation=False,
        progress_callback=lambda written, total: seen.append((written, total)),
    )

    assert len(seen) == FRAMES
    assert seen[0] == (1, FRAMES)
    assert seen[-1] == (FRAMES, FRAMES)


def test_export_overlay_video_cancels_and_removes_partial_output(
    source_video: Path, tmp_path: Path
) -> None:
    """Cancelling deletes the partial file rather than leaving an unplayable video."""
    output = tmp_path / "cancelled.mp4"
    calls = {"n": 0}

    def should_continue() -> bool:
        calls["n"] += 1
        return calls["n"] <= 3

    written = export_overlay_video(
        source_video,
        output,
        StubPose(),
        draw_segmentation=False,
        should_continue=should_continue,
    )

    assert written == 3
    assert not output.exists()


def test_export_overlay_video_creates_missing_output_directory(
    source_video: Path, tmp_path: Path
) -> None:
    """A destination in a directory that does not exist yet is still written."""
    output = tmp_path / "nested" / "dir" / "out.mp4"

    export_overlay_video(source_video, output, StubPose(), draw_segmentation=False)

    assert output.exists()


def test_export_overlay_video_rejects_an_unusable_codec(
    source_video: Path, tmp_path: Path
) -> None:
    """An unopenable writer fails loudly instead of producing an empty file."""
    with pytest.raises(VideoExportError, match="codec"):
        export_overlay_video(
            source_video,
            tmp_path / "out.mp4",
            StubPose(),
            draw_segmentation=False,
            codec="ZZZZ",
        )


def test_export_overlay_video_warns_on_frame_count_mismatch(
    source_video: Path, tmp_path: Path, caplog
) -> None:
    """A pose/video length mismatch is surfaced rather than silently truncating."""
    with caplog.at_level(logging.WARNING, logger="jabs.video_export"):
        export_overlay_video(
            source_video,
            tmp_path / "out.mp4",
            StubPose(num_frames=FRAMES + 5),
            draw_segmentation=False,
        )

    assert "Frame count mismatch" in caplog.text


def test_default_codec_is_openable(tmp_path: Path) -> None:
    """The default codec must work in this OpenCV build, or every export fails."""
    path = tmp_path / "probe.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*DEFAULT_CODEC), 30, (64, 64))
    try:
        assert writer.isOpened()
    finally:
        writer.release()


def test_refuses_to_overwrite_the_source_video(source_video: Path) -> None:
    """Writing over the source would destroy it: the writer truncates on open.

    Reachable from both `--output <input> --force` and the GUI save dialog, so it
    is rejected before anything is opened.
    """
    size_before = source_video.stat().st_size

    with pytest.raises(VideoExportError, match="same file as the source"):
        export_overlay_video(source_video, source_video, StubPose(), draw_segmentation=False)

    assert source_video.stat().st_size == size_before, "source was modified"


def test_unreadable_source_raises_video_export_error(tmp_path: Path) -> None:
    """Reader failures are translated, since callers only catch VideoExportError."""
    bad = tmp_path / "not-a-video.avi"
    bad.write_bytes(b"garbage")

    with pytest.raises(VideoExportError, match="Could not read"):
        export_overlay_video(bad, tmp_path / "out.mp4", StubPose(), draw_segmentation=False)


def test_pose_shorter_than_video_writes_the_tail_without_overlay(
    source_video: Path, tmp_path: Path
) -> None:
    """A short pose file must not abort the export partway through.

    A real pose object indexes frame-backed arrays, so requesting a frame past its
    end raises IndexError. Those frames are written un-overlaid instead, keeping the
    exported video the same length as the source.

    Unreachable through a JABS project, which validates frame counts at init, but
    reachable through `jabs-cli export-video`, which takes a loose video/pose pair.
    """
    output = tmp_path / "short.mp4"

    written = export_overlay_video(
        source_video, output, StubPose(num_frames=4), draw_segmentation=False
    )

    assert written == FRAMES
    capture = cv2.VideoCapture(str(output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAMES
    finally:
        capture.release()


def test_failure_mid_export_removes_the_partial_file(source_video: Path, tmp_path: Path) -> None:
    """An exception must not leave an unplayable file that also blocks a retry."""
    output = tmp_path / "boom.mp4"

    class _ExplodingPose(StubPose):
        def get_points(self, frame_index: int, identity: int):
            if frame_index == 3:
                raise RuntimeError("render exploded")
            return super().get_points(frame_index, identity)

    with pytest.raises(RuntimeError, match="render exploded"):
        export_overlay_video(source_video, output, _ExplodingPose(), draw_segmentation=False)

    assert not output.exists()


def test_over_reported_frame_count_keeps_the_completed_export(
    source_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container that over-reports its length must not cost us a good render.

    ``cv2.CAP_PROP_FRAME_COUNT`` is an estimate for many containers - often derived
    from duration times frame rate - so the stream can end before the reported
    count. That is a complete export, not a truncated one. Deciding whether to
    discard the output from ``frames_written < total_frames`` silently deleted
    perfectly good videos; cancellation is now tracked explicitly instead.
    """
    real_reader = video_writer_module.VideoReader

    class _OverReportingReader(real_reader):
        @property
        def num_frames(self) -> int:
            return super().num_frames + 3

    monkeypatch.setattr(video_writer_module, "VideoReader", _OverReportingReader)
    output = tmp_path / "out.mp4"

    written = export_overlay_video(
        source_video, output, StubPose(num_frames=FRAMES + 3), draw_segmentation=False
    )

    assert written == FRAMES, "should write every frame the file actually has"
    assert output.exists(), "a complete export must not be deleted"


@pytest.mark.parametrize("codec", ["mp4", "h2641", ""], ids=["too-short", "too-long", "empty"])
def test_non_fourcc_codec_raises_video_export_error(
    source_video: Path, tmp_path: Path, codec: str
) -> None:
    """A codec that is not four characters must honour the documented contract.

    ``cv2.VideoWriter_fourcc()`` raises TypeError for anything but four characters,
    which would escape before the isOpened() check can report a VideoExportError -
    so the CLI, which catches only VideoExportError, would show a raw traceback.
    """
    with pytest.raises(VideoExportError, match="four-character"):
        export_overlay_video(
            source_video,
            tmp_path / "out.mp4",
            StubPose(),
            draw_segmentation=False,
            codec=codec,
        )


# --- container metadata is not trusted -------------------------------------------
#
# cv2 reports frame count, dimensions and frame rate from container headers, all of
# which are estimates for common formats. Every one of these produced silent data
# loss or a silently truncated export when it was believed.


def test_writer_is_sized_from_a_decoded_frame_not_metadata(
    source_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container that misreports its dimensions must not yield an empty file.

    cv2 silently discards frames whose size differs from the writer's, so sizing
    the writer from metadata produced a tiny unreadable file while reporting every
    frame as written.
    """
    real_reader = video_writer_module.VideoReader

    class _WrongDimensions(real_reader):
        @property
        def dimensions(self) -> tuple[int, int]:
            return (WIDTH + 40, HEIGHT + 30)

    monkeypatch.setattr(video_writer_module, "VideoReader", _WrongDimensions)
    output = tmp_path / "out.mp4"

    written = export_overlay_video(source_video, output, StubPose(), draw_segmentation=False)

    assert written == FRAMES
    capture = cv2.VideoCapture(str(output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAMES, "frames were discarded"
    finally:
        capture.release()


def test_under_reported_frame_count_does_not_truncate(
    source_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading is bounded by end-of-stream, not by the container's frame count."""
    real_reader = video_writer_module.VideoReader

    class _UnderReportingReader(real_reader):
        @property
        def num_frames(self) -> int:
            return 0

    monkeypatch.setattr(video_writer_module, "VideoReader", _UnderReportingReader)
    output = tmp_path / "out.mp4"

    written = export_overlay_video(source_video, output, StubPose(), draw_segmentation=False)

    assert written == FRAMES, "a low frame count must not cut the export short"


def test_zero_frame_rate_is_reported_as_such(
    source_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad frame rate is named, rather than blamed on the codec."""
    real_reader = video_writer_module.VideoReader

    class _NoFps(real_reader):
        @property
        def fps(self) -> int:
            return 0

    monkeypatch.setattr(video_writer_module, "VideoReader", _NoFps)

    with pytest.raises(VideoExportError, match="frame rate"):
        export_overlay_video(
            source_video, tmp_path / "out.mp4", StubPose(), draw_segmentation=False
        )


def test_source_with_no_decodable_frames_is_rejected(tmp_path: Path) -> None:
    """An empty video must not produce a header-only file reported as success."""
    empty = tmp_path / "empty.avi"
    cv2.VideoWriter(str(empty), cv2.VideoWriter_fourcc(*"MJPG"), 30, (WIDTH, HEIGHT)).release()

    with pytest.raises(VideoExportError, match="no decodable frames"):
        export_overlay_video(empty, tmp_path / "out.mp4", StubPose(), draw_segmentation=False)


def test_refuses_a_case_differing_alias_of_the_source(source_video: Path, tmp_path: Path) -> None:
    """The same-file guard must hold on case-insensitive filesystems.

    macOS and Windows resolve `CLIP.AVI` and `clip.avi` to one file, but comparing
    resolved paths reports them as different - so the guard passed and the export
    truncated the source it was reading. os.path.samefile is the check that holds,
    and it covers hard links too.
    """
    alias = source_video.with_name(source_video.name.upper())
    if not alias.exists():  # pragma: no cover - case-sensitive filesystem
        pytest.skip("filesystem is case-sensitive; alias does not resolve to the source")
    size_before = source_video.stat().st_size

    with pytest.raises(VideoExportError, match="same file as the source"):
        export_overlay_video(source_video, alias, StubPose(), draw_segmentation=False)

    assert source_video.stat().st_size == size_before, "source was modified"
