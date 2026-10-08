"""Tests for the link-nwb-video CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner, Result

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types import PoseData
from jabs.io.internal.pose import NWBVideoLinkInfo, PoseNWBAdapter, read_video_link_info
from jabs.scripts.cli import link_nwb_video
from jabs.scripts.cli.cli import cli
from jabs.scripts.cli.link_nwb_video import (
    LinkStatus,
    VideoProbe,
    describe_mismatch,
    find_nwb_files,
    find_video,
    link_one,
    probe_video,
)

NUM_FRAMES = 12
FPS = 30.0
INFO = NWBVideoLinkInfo(num_frames=NUM_FRAMES, rate=FPS, external_files=[])
MATCHING_PROBE = VideoProbe(num_frames=NUM_FRAMES, fps=FPS, width=640, height=480)


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create an NWB directory and a video directory, each holding a ``clip`` file."""
    nwb_dir = tmp_path / "nwb"
    video_dir = tmp_path / "videos"
    nwb_dir.mkdir()
    video_dir.mkdir()
    (nwb_dir / "clip.nwb").write_bytes(b"nwb")
    (video_dir / "clip.mp4").write_bytes(b"mp4")
    return nwb_dir, video_dir


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Replace the NWB reader, video probe and NWB writer used by ``link_one``.

    Returns:
        A mock whose ``read``, ``probe`` and ``link`` attributes are the stand-ins.
    """
    stub = mock.Mock()
    stub.read.return_value = INFO
    stub.probe.return_value = MATCHING_PROBE
    stub.link.return_value = "../videos/clip.mp4"
    monkeypatch.setattr(link_nwb_video, "read_video_link_info", stub.read)
    monkeypatch.setattr(link_nwb_video, "probe_video", stub.probe)
    monkeypatch.setattr(link_nwb_video, "link_external_video", stub.link)
    return stub


# ---------------------------------------------------------------------------
# find_nwb_files / find_video
# ---------------------------------------------------------------------------


def test_find_nwb_files_lists_visible_nwb_files_sorted(tmp_path: Path) -> None:
    """Only .nwb files directly in the directory are listed, without hidden ones."""
    for name in ("b.nwb", "a.nwb", "notes.txt", ".c.nwb.linking", ".hidden.nwb"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.nwb").write_bytes(b"")
    (tmp_path / "dir.nwb").mkdir()

    assert [p.name for p in find_nwb_files(tmp_path)] == ["a.nwb", "b.nwb"]


def test_find_video_matches_on_name(dirs: tuple[Path, Path]) -> None:
    """The video has the NWB file's name with an .mp4 extension."""
    nwb_dir, video_dir = dirs

    assert find_video(nwb_dir / "clip.nwb", video_dir) == video_dir / "clip.mp4"


def test_find_video_missing(dirs: tuple[Path, Path]) -> None:
    """No video is returned when only differently named videos exist."""
    nwb_dir, video_dir = dirs
    (video_dir / "clip.mp4").rename(video_dir / "other.mp4")

    assert find_video(nwb_dir / "clip.nwb", video_dir) is None


# ---------------------------------------------------------------------------
# probe_video / describe_mismatch
# ---------------------------------------------------------------------------


def test_probe_video_reads_header_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frame count, rate and size come from the capture, and the capture is released."""
    capture = mock.Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = lambda prop: {
        link_nwb_video.cv2.CAP_PROP_FRAME_COUNT: 25.0,
        link_nwb_video.cv2.CAP_PROP_FPS: 29.97,
        link_nwb_video.cv2.CAP_PROP_FRAME_WIDTH: 64.0,
        link_nwb_video.cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
    }[prop]
    monkeypatch.setattr(link_nwb_video.cv2, "VideoCapture", mock.Mock(return_value=capture))

    assert probe_video(Path("v.mp4")) == VideoProbe(25, 29.97, 64, 48)
    capture.release.assert_called_once()


def test_probe_video_unopenable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A video that cannot be opened raises OSError and is still released."""
    capture = mock.Mock()
    capture.isOpened.return_value = False
    monkeypatch.setattr(link_nwb_video.cv2, "VideoCapture", mock.Mock(return_value=capture))

    with pytest.raises(OSError, match="Unable to open video file"):
        probe_video(Path("v.mp4"))
    capture.release.assert_called_once()


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        (MATCHING_PROBE, None),
        (VideoProbe(NUM_FRAMES, 29.97, 640, 480), None),
        (VideoProbe(NUM_FRAMES + 1, FPS, 640, 480), "video has 13 frames, pose has 12"),
        (VideoProbe(NUM_FRAMES, 25.0, 640, 480), "video is 25 fps, pose is 30 fps"),
        (
            VideoProbe(NUM_FRAMES - 2, 25.0, 640, 480),
            "video has 10 frames, pose has 12; video is 25 fps, pose is 30 fps",
        ),
    ],
    ids=["match", "ntsc-rate-tolerated", "frame-count", "frame-rate", "both"],
)
def test_describe_mismatch(probe: VideoProbe, expected: str | None) -> None:
    """Differences in frame count or rate are described; a rate within 1% is accepted."""
    assert describe_mismatch(probe, INFO) == expected


# ---------------------------------------------------------------------------
# link_one
# ---------------------------------------------------------------------------


def test_link_one_links_and_passes_timing_and_size(
    dirs: tuple[Path, Path], stubs: mock.Mock
) -> None:
    """A matching video is linked with the pose timing and the video's frame size."""
    nwb_dir, video_dir = dirs

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=True)

    assert (result.status, result.message) == (LinkStatus.LINKED, "../videos/clip.mp4")
    stubs.link.assert_called_once_with(
        nwb_dir / "clip.nwb",
        video_dir / "clip.mp4",
        num_frames=NUM_FRAMES,
        rate=FPS,
        dimension=(640, 480),
    )


def test_link_one_without_verify_does_not_open_the_video(
    dirs: tuple[Path, Path], stubs: mock.Mock
) -> None:
    """--no-verify links without probing the video and without a frame size."""
    nwb_dir, video_dir = dirs

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=False)

    assert result.status is LinkStatus.LINKED
    stubs.probe.assert_not_called()
    assert stubs.link.call_args.kwargs["dimension"] is None


def test_link_one_dry_run_does_not_modify(dirs: tuple[Path, Path], stubs: mock.Mock) -> None:
    """A dry run reports the link that would be stored and writes nothing."""
    nwb_dir, video_dir = dirs

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=True, verify=True)

    assert (result.status, result.message) == (LinkStatus.WOULD_LINK, "../videos/clip.mp4")
    stubs.link.assert_not_called()


def test_link_one_no_video(dirs: tuple[Path, Path], stubs: mock.Mock) -> None:
    """An NWB file with no matching video is reported and left alone."""
    nwb_dir, video_dir = dirs
    (video_dir / "clip.mp4").unlink()

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=True)

    assert result.status is LinkStatus.NO_VIDEO
    assert "clip.mp4" in result.message
    stubs.link.assert_not_called()


def test_link_one_mismatch_is_not_linked(dirs: tuple[Path, Path], stubs: mock.Mock) -> None:
    """A video that disagrees with the pose data is reported, not linked."""
    nwb_dir, video_dir = dirs
    stubs.probe.return_value = VideoProbe(NUM_FRAMES + 5, FPS, 640, 480)

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=True)

    assert result.status is LinkStatus.MISMATCH
    assert "17 frames" in result.message
    stubs.link.assert_not_called()


def test_link_one_unopenable_video(dirs: tuple[Path, Path], stubs: mock.Mock) -> None:
    """A video that cannot be opened is an error, not a crash."""
    nwb_dir, video_dir = dirs
    stubs.probe.side_effect = OSError("Unable to open video file: clip.mp4")

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=True)

    assert result.status is LinkStatus.ERROR
    stubs.link.assert_not_called()


def test_link_one_unreadable_nwb(dirs: tuple[Path, Path], stubs: mock.Mock) -> None:
    """A file that is not a JABS pose NWB file is an error."""
    nwb_dir, video_dir = dirs
    stubs.read.side_effect = KeyError("behavior")

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=True)

    assert result.status is LinkStatus.ERROR
    assert "not a readable JABS pose NWB file" in result.message
    stubs.link.assert_not_called()


def test_link_one_already_linked_to_this_video(dirs: tuple[Path, Path], stubs: mock.Mock) -> None:
    """A file already linked to its video is skipped, so the command can be re-run."""
    nwb_dir, video_dir = dirs
    stubs.read.return_value = INFO._replace(external_files=["../videos/clip.mp4"])

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=True)

    assert result.status is LinkStatus.ALREADY_LINKED
    stubs.link.assert_not_called()


def test_link_one_linked_to_a_different_video(dirs: tuple[Path, Path], stubs: mock.Mock) -> None:
    """A file linked somewhere else is an error and is not changed."""
    nwb_dir, video_dir = dirs
    stubs.read.return_value = INFO._replace(external_files=["elsewhere/clip.mp4"])

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=True)

    assert result.status is LinkStatus.ERROR
    assert "elsewhere/clip.mp4" in result.message
    stubs.link.assert_not_called()


def test_link_one_write_failure_is_reported(dirs: tuple[Path, Path], stubs: mock.Mock) -> None:
    """A failure while modifying the file is reported so the batch can continue."""
    nwb_dir, video_dir = dirs
    stubs.link.side_effect = ValueError("disk on fire")

    result = link_one(nwb_dir / "clip.nwb", video_dir, dry_run=False, verify=True)

    assert (result.status, result.message) == (LinkStatus.ERROR, "disk on fire")


# ---------------------------------------------------------------------------
# CLI, against real NWB files
# ---------------------------------------------------------------------------


@pytest.fixture
def real_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Write two real pose NWB files with a matching stand-in video for the first one.

    ``first.nwb`` has ``first.mp4``; ``second.nwb`` has no video. Video probing is
    replaced so the stand-in does not have to be decodable.
    """
    # The NWB writer needs the optional `nwb` extra, so skip when it is not installed
    pytest.importorskip("pynwb")
    pytest.importorskip("ndx_pose")
    pytest.importorskip("ndx_jabs")
    pytest.importorskip("ndx_multisubjects")

    body_parts = [kpt.name for kpt in PoseEstimation.KeypointIndex]
    data = PoseData(
        points=np.zeros((1, NUM_FRAMES, len(body_parts), 2)),
        point_mask=np.ones((1, NUM_FRAMES, len(body_parts)), dtype=bool),
        identity_mask=np.ones((1, NUM_FRAMES), dtype=bool),
        body_parts=body_parts,
        edges=[],
        fps=int(FPS),
        external_ids=["mouse"],
    )
    nwb_dir = tmp_path / "nwb"
    video_dir = tmp_path / "videos"
    nwb_dir.mkdir()
    video_dir.mkdir()
    for stem in ("first", "second"):
        PoseNWBAdapter().write(data, nwb_dir / f"{stem}.nwb")
        (nwb_dir / f"{stem}_mouse.nwb").rename(nwb_dir / f"{stem}.nwb")
    (video_dir / "first.mp4").write_bytes(b"stand-in video")
    monkeypatch.setattr(link_nwb_video, "probe_video", mock.Mock(return_value=MATCHING_PROBE))
    return nwb_dir, video_dir


def _run(*args: str | Path) -> Result:
    """Run ``jabs-cli link-nwb-video`` with the given arguments."""
    return CliRunner().invoke(cli, ["link-nwb-video", *map(str, args)])


def test_cli_links_files_and_reports_the_ones_without_a_video(
    real_dirs: tuple[Path, Path],
) -> None:
    """The matched file is linked; the unmatched one is reported and fails the run."""
    nwb_dir, video_dir = real_dirs

    result = _run(nwb_dir, video_dir)

    assert result.exit_code == 1, result.output
    assert read_video_link_info(nwb_dir / "first.nwb").external_files == ["../videos/first.mp4"]
    assert read_video_link_info(nwb_dir / "second.nwb").external_files == []
    assert "second.nwb: second.mp4 not found" in result.output
    assert "1 linked, 1 no video" in result.output


def test_cli_second_run_skips_linked_files(real_dirs: tuple[Path, Path]) -> None:
    """Running again leaves a linked file alone and reports it as already linked."""
    nwb_dir, video_dir = real_dirs
    (video_dir / "second.mp4").write_bytes(b"stand-in video")
    assert _run(nwb_dir, video_dir).exit_code == 0
    linked = (nwb_dir / "first.nwb").read_bytes()

    result = _run(nwb_dir, video_dir)

    assert result.exit_code == 0, result.output
    assert "2 already linked" in result.output
    assert (nwb_dir / "first.nwb").read_bytes() == linked


def test_cli_dry_run_modifies_nothing(real_dirs: tuple[Path, Path]) -> None:
    """--dry-run reports what would be linked and leaves every file as it was."""
    nwb_dir, video_dir = real_dirs
    before = {p.name: p.read_bytes() for p in nwb_dir.iterdir()}

    result = _run("--dry-run", nwb_dir, video_dir)

    assert "would link" in result.output
    assert {p.name: p.read_bytes() for p in nwb_dir.iterdir()} == before


def test_cli_mismatched_video_is_not_linked(
    real_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video with the wrong frame count fails the run and leaves the file unlinked."""
    nwb_dir, video_dir = real_dirs
    wrong = VideoProbe(NUM_FRAMES + 1, FPS, 640, 480)
    monkeypatch.setattr(link_nwb_video, "probe_video", mock.Mock(return_value=wrong))

    result = _run(nwb_dir, video_dir)

    assert result.exit_code == 1
    assert "video has 13 frames, pose has 12" in result.output
    assert read_video_link_info(nwb_dir / "first.nwb").external_files == []


def test_cli_same_directory_for_nwb_and_videos(real_dirs: tuple[Path, Path]) -> None:
    """The NWB files and the videos may share one directory."""
    nwb_dir, video_dir = real_dirs
    (video_dir / "first.mp4").rename(nwb_dir / "first.mp4")

    result = _run(nwb_dir, nwb_dir)

    assert "1 linked" in result.output
    assert read_video_link_info(nwb_dir / "first.nwb").external_files == ["first.mp4"]


def test_cli_reports_missing_nwb_support_once(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing NWB dependencies fail the command up front, not once per file."""
    nwb_dir, video_dir = dirs
    monkeypatch.setattr(
        link_nwb_video,
        "require_video_link_support",
        mock.Mock(side_effect=ImportError("needs the nwb extra")),
    )

    result = _run(nwb_dir, video_dir)

    assert result.exit_code != 0
    assert "needs the nwb extra" in result.output


def test_cli_without_nwb_files_is_an_error(tmp_path: Path) -> None:
    """A directory with no .nwb files is reported rather than silently doing nothing."""
    result = _run(tmp_path, tmp_path)

    assert result.exit_code != 0
    assert "No .nwb files found" in result.output
