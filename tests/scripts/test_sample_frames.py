"""Tests for the sample-frames CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from jabs.scripts.cli.cli import cli
from jabs.scripts.cli.sample_frames import (
    collect_behavior_bouts,
    sample_frames_per_bout,
    sample_num_frames_total,
    write_frames,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track_labels(blocks: list[dict]) -> MagicMock:
    """Return a mock TrackLabels whose get_blocks() returns *blocks*."""
    tl = MagicMock()
    tl.get_blocks.return_value = blocks
    return tl


def _make_project(
    videos: list[str],
    video_labels_map: dict[str, MagicMock | None],
    behaviors: list[str],
) -> MagicMock:
    """Return a minimal mock Project.

    Args:
        videos: List of video filenames the project contains.
        video_labels_map: Mapping from video filename to the VideoLabels mock
            (or ``None`` if no annotations exist for that video).
        behaviors: List of behavior names present in project.settings["behavior"].
    """
    project = MagicMock()
    project.video_manager.videos = videos
    project.video_manager.load_video_labels.side_effect = lambda v: video_labels_map.get(v)
    project.settings = {"behavior": {b: {} for b in behaviors}}
    return project


def _make_video_labels(identity_behavior_blocks: list[tuple[str, str, list[dict]]]) -> MagicMock:
    """Build a VideoLabels mock that yields the given (identity, behavior, blocks) triples.

    Args:
        identity_behavior_blocks: Each entry is ``(identity, behavior, blocks)`` where
            *blocks* is a list of dicts with keys ``"start"``, ``"end"``, ``"present"``.
    """
    vl = MagicMock()
    entries = [
        (identity, behavior, _make_track_labels(blocks))
        for identity, behavior, blocks in identity_behavior_blocks
    ]
    vl.iter_identity_behavior_labels.return_value = iter(entries)
    return vl


# ---------------------------------------------------------------------------
# collect_behavior_bouts
# ---------------------------------------------------------------------------


def test_collect_bouts_single_video_single_identity() -> None:
    """Bouts from one identity in one video are collected correctly."""
    vl = _make_video_labels(
        [
            ("0", "walking", [{"start": 10, "end": 20, "present": True}]),
        ]
    )
    project = _make_project(["vid.mp4"], {"vid.mp4": vl}, ["walking"])

    bouts = collect_behavior_bouts(project, "walking")
    assert bouts == [("vid.mp4", 10, 20)]


def test_collect_bouts_filters_not_behavior() -> None:
    """Blocks with present=False are not included."""
    vl = _make_video_labels(
        [
            (
                "0",
                "walking",
                [
                    {"start": 5, "end": 15, "present": True},
                    {"start": 16, "end": 25, "present": False},
                ],
            ),
        ]
    )
    project = _make_project(["vid.mp4"], {"vid.mp4": vl}, ["walking"])

    bouts = collect_behavior_bouts(project, "walking")
    assert bouts == [("vid.mp4", 5, 15)]


def test_collect_bouts_filters_other_behaviors() -> None:
    """Only bouts for the requested behavior are returned."""
    vl = _make_video_labels(
        [
            ("0", "walking", [{"start": 1, "end": 5, "present": True}]),
            ("0", "rearing", [{"start": 10, "end": 20, "present": True}]),
        ]
    )
    project = _make_project(["vid.mp4"], {"vid.mp4": vl}, ["walking", "rearing"])

    bouts = collect_behavior_bouts(project, "walking")
    assert bouts == [("vid.mp4", 1, 5)]


def test_collect_bouts_no_annotations() -> None:
    """Videos without annotations are skipped."""
    project = _make_project(["vid.mp4"], {"vid.mp4": None}, ["walking"])
    bouts = collect_behavior_bouts(project, "walking")
    assert bouts == []


def test_collect_bouts_multiple_videos() -> None:
    """Bouts from multiple videos are all returned."""
    vl_a = _make_video_labels([("0", "walking", [{"start": 0, "end": 9, "present": True}])])
    vl_b = _make_video_labels([("0", "walking", [{"start": 5, "end": 14, "present": True}])])
    project = _make_project(
        ["a.mp4", "b.mp4"],
        {"a.mp4": vl_a, "b.mp4": vl_b},
        ["walking"],
    )

    bouts = collect_behavior_bouts(project, "walking")
    assert ("a.mp4", 0, 9) in bouts
    assert ("b.mp4", 5, 14) in bouts
    assert len(bouts) == 2


# ---------------------------------------------------------------------------
# sample_frames_per_bout
# ---------------------------------------------------------------------------


def test_sample_frames_per_bout_returns_within_bout() -> None:
    """All sampled frames fall within the bout range."""
    rng = np.random.default_rng(0)
    bouts = [("v.mp4", 10, 50)]
    result = sample_frames_per_bout(bouts, frames_per_bout=5, rng=rng)
    assert len(result) <= 5
    for video_name, frame in result:
        assert video_name == "v.mp4"
        assert 10 <= frame <= 50


def test_sample_frames_per_bout_short_bout() -> None:
    """When a bout is shorter than frames_per_bout, all frames are returned."""
    rng = np.random.default_rng(0)
    bouts = [("v.mp4", 0, 2)]  # 3 frames: 0, 1, 2
    result = sample_frames_per_bout(bouts, frames_per_bout=10, rng=rng)
    assert len(result) == 3
    assert {f for _, f in result} == {0, 1, 2}


def test_sample_frames_per_bout_deduplicates() -> None:
    """Overlapping bouts from different identities produce deduplicated output."""
    rng = np.random.default_rng(0)
    # Two identities covering the same 5-frame bout
    bouts = [("v.mp4", 0, 4), ("v.mp4", 0, 4)]
    result = sample_frames_per_bout(bouts, frames_per_bout=5, rng=rng)
    # Result must be unique (no duplicate (video, frame) pairs)
    assert len(result) == len(set(result))
    assert len(result) <= 5


def test_sample_frames_per_bout_multiple_bouts() -> None:
    """Frames are sampled from every bout."""
    rng = np.random.default_rng(0)
    bouts = [("v.mp4", 0, 9), ("v.mp4", 100, 109)]
    result = sample_frames_per_bout(bouts, frames_per_bout=3, rng=rng)
    first_bout_frames = [f for _, f in result if 0 <= f <= 9]
    second_bout_frames = [f for _, f in result if 100 <= f <= 109]
    assert len(first_bout_frames) <= 3
    assert len(second_bout_frames) <= 3
    assert len(first_bout_frames) > 0
    assert len(second_bout_frames) > 0


# ---------------------------------------------------------------------------
# sample_num_frames_total
# ---------------------------------------------------------------------------


def test_sample_num_frames_total_count() -> None:
    """Exactly num_frames unique frames are returned when pool is large enough."""
    rng = np.random.default_rng(0)
    bouts = [("v.mp4", 0, 99)]  # 100 frames
    result = sample_num_frames_total(bouts, num_frames=20, rng=rng)
    assert len(result) == 20


def test_sample_num_frames_total_capped_at_pool() -> None:
    """When num_frames > pool size, all available frames are returned."""
    rng = np.random.default_rng(0)
    bouts = [("v.mp4", 0, 4)]  # 5 frames
    result = sample_num_frames_total(bouts, num_frames=100, rng=rng)
    assert len(result) == 5


def test_sample_num_frames_total_all_within_bouts() -> None:
    """Every returned frame falls within a labeled bout."""
    rng = np.random.default_rng(42)
    bouts = [("v.mp4", 10, 30), ("v.mp4", 50, 70)]
    result = sample_num_frames_total(bouts, num_frames=15, rng=rng)
    for _, frame in result:
        assert (10 <= frame <= 30) or (50 <= frame <= 70)


def test_sample_num_frames_total_deduplicates_across_bouts() -> None:
    """Overlapping bouts are merged before sampling; no duplicate frames."""
    rng = np.random.default_rng(0)
    bouts = [("v.mp4", 0, 9), ("v.mp4", 0, 9)]  # same range twice
    result = sample_num_frames_total(bouts, num_frames=10, rng=rng)
    assert len(result) == len(set(result))
    assert len(result) == 10


# ---------------------------------------------------------------------------
# CLI integration tests (via click.testing.CliRunner + mocks)
# ---------------------------------------------------------------------------


def test_cli_requires_at_least_one_mode(tmp_path: Path) -> None:
    """Omitting both --num-frames and --frames-per-bout shows a usage error."""
    runner = CliRunner()
    with patch("jabs.scripts.cli.sample_frames.Project") as MockProject:
        MockProject.is_valid_project_directory.return_value = True
        result = runner.invoke(cli, ["sample-frames", "--behavior", "walking", str(tmp_path)])
    assert result.exit_code != 0
    assert "required" in result.output.lower() or (
        result.exception is not None and "required" in str(result.exception).lower()
    )


def test_cli_mutual_exclusion(tmp_path: Path) -> None:
    """Providing both --num-frames and --frames-per-bout is an error."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sample-frames",
            "--behavior",
            "walking",
            "--num-frames",
            "10",
            "--frames-per-bout",
            "5",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower() or (
        result.exception is not None and "mutually exclusive" in str(result.exception).lower()
    )


def test_cli_invalid_project_dir(tmp_path: Path) -> None:
    """A directory that is not a JABS project produces a clear error."""
    runner = CliRunner()
    with patch("jabs.scripts.cli.sample_frames.Project") as MockProject:
        MockProject.is_valid_project_directory.return_value = False
        result = runner.invoke(
            cli,
            ["sample-frames", "--behavior", "walking", "--num-frames", "5", str(tmp_path)],
        )
    assert result.exit_code != 0
    assert "not a valid jabs project" in result.output.lower()


def test_cli_unknown_behavior(tmp_path: Path) -> None:
    """An unknown behavior label produces a clear error."""
    runner = CliRunner()
    with patch("jabs.scripts.cli.sample_frames.Project") as MockProject:
        MockProject.is_valid_project_directory.return_value = True
        mock_project = MagicMock()
        mock_project.settings = {"behavior": {"rearing": {}}}
        MockProject.return_value = mock_project
        result = runner.invoke(
            cli,
            ["sample-frames", "--behavior", "walking", "--num-frames", "5", str(tmp_path)],
        )
    assert result.exit_code != 0
    assert "walking" in result.output


def test_cli_out_dir_created(tmp_path: Path) -> None:
    """--out-dir is created if it does not yet exist."""
    out_dir = tmp_path / "new" / "subdir"
    assert not out_dir.exists()

    runner = CliRunner()
    with (
        patch("jabs.scripts.cli.sample_frames.Project") as MockProject,
        patch("jabs.scripts.cli.sample_frames.collect_behavior_bouts") as mock_collect,
        patch("jabs.scripts.cli.sample_frames.sample_num_frames_total") as mock_sample,
        patch("jabs.scripts.cli.sample_frames.write_frames") as mock_write,
    ):
        MockProject.is_valid_project_directory.return_value = True
        mock_project = MagicMock()
        mock_project.settings = {"behavior": {"walking": {}}}
        MockProject.return_value = mock_project
        mock_collect.return_value = [("vid.mp4", 0, 99)]
        mock_sample.return_value = [("vid.mp4", 42)]
        mock_write.return_value = None

        result = runner.invoke(
            cli,
            [
                "sample-frames",
                "--behavior",
                "walking",
                "--num-frames",
                "1",
                "--out-dir",
                str(out_dir),
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert out_dir.exists()


def test_cli_num_frames_calls_correct_sampler(tmp_path: Path) -> None:
    """--num-frames invokes sample_num_frames_total (not sample_frames_per_bout)."""
    runner = CliRunner()
    with (
        patch("jabs.scripts.cli.sample_frames.Project") as MockProject,
        patch("jabs.scripts.cli.sample_frames.collect_behavior_bouts") as mock_collect,
        patch("jabs.scripts.cli.sample_frames.sample_num_frames_total") as mock_total,
        patch("jabs.scripts.cli.sample_frames.sample_frames_per_bout") as mock_per_bout,
        patch("jabs.scripts.cli.sample_frames.write_frames"),
    ):
        MockProject.is_valid_project_directory.return_value = True
        mock_project = MagicMock()
        mock_project.settings = {"behavior": {"walking": {}}}
        MockProject.return_value = mock_project
        mock_collect.return_value = [("v.mp4", 0, 9)]
        mock_total.return_value = [("v.mp4", 5)]

        result = runner.invoke(
            cli,
            ["sample-frames", "--behavior", "walking", "--num-frames", "1", str(tmp_path)],
        )

    assert result.exit_code == 0, result.output
    mock_total.assert_called_once()
    mock_per_bout.assert_not_called()


def test_cli_frames_per_bout_calls_correct_sampler(tmp_path: Path) -> None:
    """--frames-per-bout invokes sample_frames_per_bout (not sample_num_frames_total)."""
    runner = CliRunner()
    with (
        patch("jabs.scripts.cli.sample_frames.Project") as MockProject,
        patch("jabs.scripts.cli.sample_frames.collect_behavior_bouts") as mock_collect,
        patch("jabs.scripts.cli.sample_frames.sample_num_frames_total") as mock_total,
        patch("jabs.scripts.cli.sample_frames.sample_frames_per_bout") as mock_per_bout,
        patch("jabs.scripts.cli.sample_frames.write_frames"),
    ):
        MockProject.is_valid_project_directory.return_value = True
        mock_project = MagicMock()
        mock_project.settings = {"behavior": {"walking": {}}}
        MockProject.return_value = mock_project
        mock_collect.return_value = [("v.mp4", 0, 9)]
        mock_per_bout.return_value = [("v.mp4", 3)]

        result = runner.invoke(
            cli,
            ["sample-frames", "--behavior", "walking", "--frames-per-bout", "5", str(tmp_path)],
        )

    assert result.exit_code == 0, result.output
    mock_per_bout.assert_called_once()
    mock_total.assert_not_called()


# ---------------------------------------------------------------------------
# collect_behavior_bouts — additional edge cases
# ---------------------------------------------------------------------------


def test_collect_bouts_empty_project() -> None:
    """A project with no videos returns an empty list."""
    project = _make_project([], {}, ["walking"])
    assert collect_behavior_bouts(project, "walking") == []


def test_collect_bouts_multiple_identities_same_video() -> None:
    """Bouts from multiple identities in the same video are all collected."""
    vl = _make_video_labels(
        [
            ("0", "walking", [{"start": 0, "end": 5, "present": True}]),
            ("1", "walking", [{"start": 3, "end": 8, "present": True}]),
        ]
    )
    project = _make_project(["v.mp4"], {"v.mp4": vl}, ["walking"])

    bouts = collect_behavior_bouts(project, "walking")
    assert len(bouts) == 2
    assert ("v.mp4", 0, 5) in bouts
    assert ("v.mp4", 3, 8) in bouts


def test_collect_bouts_mixed_present_flags() -> None:
    """Only blocks with present=True are included; present=False blocks are excluded."""
    vl = _make_video_labels(
        [
            (
                "0",
                "walking",
                [
                    {"start": 0, "end": 10, "present": True},
                    {"start": 11, "end": 20, "present": False},
                    {"start": 21, "end": 30, "present": True},
                ],
            ),
        ]
    )
    project = _make_project(["v.mp4"], {"v.mp4": vl}, ["walking"])

    bouts = collect_behavior_bouts(project, "walking")
    assert len(bouts) == 2
    assert ("v.mp4", 0, 10) in bouts
    assert ("v.mp4", 21, 30) in bouts


# ---------------------------------------------------------------------------
# sample_frames_per_bout — additional edge cases
# ---------------------------------------------------------------------------


def test_sample_frames_per_bout_empty_bouts() -> None:
    """Empty bouts list returns empty result."""
    rng = np.random.default_rng(0)
    assert sample_frames_per_bout([], frames_per_bout=5, rng=rng) == []


def test_sample_frames_per_bout_single_frame_bout() -> None:
    """A single-frame bout with frames_per_bout=1 returns that one frame."""
    rng = np.random.default_rng(0)
    result = sample_frames_per_bout([("v.mp4", 42, 42)], frames_per_bout=1, rng=rng)
    assert result == [("v.mp4", 42)]


def test_sample_frames_per_bout_result_is_sorted() -> None:
    """Returned list is sorted by (video_name, frame_index)."""
    rng = np.random.default_rng(0)
    bouts = [("b.mp4", 0, 9), ("a.mp4", 0, 9)]
    result = sample_frames_per_bout(bouts, frames_per_bout=10, rng=rng)
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# sample_num_frames_total — additional edge cases
# ---------------------------------------------------------------------------


def test_sample_num_frames_total_empty_bouts() -> None:
    """Empty bouts list returns empty result."""
    rng = np.random.default_rng(0)
    assert sample_num_frames_total([], num_frames=10, rng=rng) == []


def test_sample_num_frames_total_frames_from_multiple_videos_stay_separate() -> None:
    """Frame indices that are identical but from different videos are not deduplicated."""
    rng = np.random.default_rng(0)
    # Both videos have frame 0 — they should each appear in the pool independently.
    bouts = [("a.mp4", 0, 0), ("b.mp4", 0, 0)]
    result = sample_num_frames_total(bouts, num_frames=2, rng=rng)
    assert len(result) == 2
    assert ("a.mp4", 0) in result
    assert ("b.mp4", 0) in result


def test_sample_num_frames_total_result_is_sorted() -> None:
    """Returned list is sorted by (video_name, frame_index)."""
    rng = np.random.default_rng(0)
    bouts = [("b.mp4", 0, 4), ("a.mp4", 0, 4)]
    result = sample_num_frames_total(bouts, num_frames=10, rng=rng)
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# write_frames
# ---------------------------------------------------------------------------


def _make_cv2_mock(grabbed: bool = True, imwrite_ok: bool = True) -> MagicMock:
    """Return a mock cv2 module with a VideoCapture that yields one frame."""
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_POS_FRAMES = 1  # arbitrary constant value

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (
        grabbed,
        np.zeros((4, 4, 3), dtype=np.uint8) if grabbed else None,
    )
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.imwrite.return_value = imwrite_ok
    return mock_cv2


def test_write_frames_output_filename_convention(tmp_path: Path) -> None:
    """Output filename matches {stem}_frame{N:06d}.png."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mock_cv2 = _make_cv2_mock()

    with patch("jabs.scripts.cli.sample_frames.cv2", mock_cv2):
        write_frames(tmp_path, [("session_2024.mp4", 1234)], out_dir)

    written_path = Path(mock_cv2.imwrite.call_args.args[0])
    assert written_path.name == "session_2024_frame001234.png"


def test_write_frames_zero_pads_frame_index(tmp_path: Path) -> None:
    """Frame index is zero-padded to 6 digits."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with patch("jabs.scripts.cli.sample_frames.cv2", _make_cv2_mock()) as mock_cv2:
        write_frames(tmp_path, [("v.mp4", 0), ("v.mp4", 999999)], out_dir)
        written_paths = [Path(c.args[0]).name for c in mock_cv2.imwrite.call_args_list]

    assert "v_frame000000.png" in written_paths
    assert "v_frame999999.png" in written_paths


def test_write_frames_groups_by_video(tmp_path: Path) -> None:
    """Multiple frames from the same video open VideoCapture only once."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mock_cv2 = _make_cv2_mock()

    with patch("jabs.scripts.cli.sample_frames.cv2", mock_cv2):
        write_frames(tmp_path, [("v.mp4", 1), ("v.mp4", 2), ("v.mp4", 3)], out_dir)

    assert mock_cv2.VideoCapture.call_count == 1


def test_write_frames_unreadable_frame_is_skipped(tmp_path: Path) -> None:
    """A frame that cannot be read (grabbed=False) is skipped without raising."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mock_cv2 = _make_cv2_mock(grabbed=False)

    with patch("jabs.scripts.cli.sample_frames.cv2", mock_cv2):
        write_frames(tmp_path, [("v.mp4", 5)], out_dir)  # must not raise

    mock_cv2.imwrite.assert_not_called()


def test_write_frames_unopenable_video_raises(tmp_path: Path) -> None:
    """An unopenable video file raises OSError."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mock_cv2 = _make_cv2_mock()
    mock_cv2.VideoCapture.return_value.isOpened.return_value = False

    with (
        patch("jabs.scripts.cli.sample_frames.cv2", mock_cv2),
        pytest.raises(OSError, match="Unable to open video"),
    ):
        write_frames(tmp_path, [("missing.mp4", 0)], out_dir)


def test_write_frames_imwrite_failure_raises(tmp_path: Path) -> None:
    """A failed cv2.imwrite raises RuntimeError."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mock_cv2 = _make_cv2_mock(imwrite_ok=False)

    with (
        patch("jabs.scripts.cli.sample_frames.cv2", mock_cv2),
        pytest.raises(RuntimeError, match="Failed to write PNG"),
    ):
        write_frames(tmp_path, [("v.mp4", 0)], out_dir)


# ---------------------------------------------------------------------------
# CLI — additional cases
# ---------------------------------------------------------------------------


def test_cli_no_bouts_found_error(tmp_path: Path) -> None:
    """A clear error is shown when no frames are labeled with the behavior."""
    runner = CliRunner()
    with (
        patch("jabs.scripts.cli.sample_frames.Project") as MockProject,
        patch("jabs.scripts.cli.sample_frames.collect_behavior_bouts") as mock_collect,
    ):
        MockProject.is_valid_project_directory.return_value = True
        mock_project = MagicMock()
        mock_project.settings = {"behavior": {"walking": {}}}
        mock_project.video_manager.videos = []
        MockProject.return_value = mock_project
        mock_collect.return_value = []

        result = runner.invoke(
            cli,
            ["sample-frames", "--behavior", "walking", "--num-frames", "10", str(tmp_path)],
        )

    assert result.exit_code != 0
    assert "walking" in result.output


def test_cli_unknown_behavior_lists_available(tmp_path: Path) -> None:
    """The error for an unknown behavior includes the available behavior names."""
    runner = CliRunner()
    with patch("jabs.scripts.cli.sample_frames.Project") as MockProject:
        MockProject.is_valid_project_directory.return_value = True
        mock_project = MagicMock()
        mock_project.settings = {"behavior": {"rearing": {}, "grooming": {}}}
        MockProject.return_value = mock_project

        result = runner.invoke(
            cli,
            ["sample-frames", "--behavior", "walking", "--num-frames", "5", str(tmp_path)],
        )

    assert result.exit_code != 0
    assert "rearing" in result.output
    assert "grooming" in result.output


def test_cli_default_out_dir_is_cwd(tmp_path: Path) -> None:
    """When --out-dir is omitted, frames are written to the current working directory."""
    runner = CliRunner()
    captured_out_dir: list[Path] = []

    def _capture_write_frames(project_dir, sampled, out_dir, **kwargs):
        captured_out_dir.append(out_dir)

    with (
        patch("jabs.scripts.cli.sample_frames.Project") as MockProject,
        patch("jabs.scripts.cli.sample_frames.collect_behavior_bouts") as mock_collect,
        patch("jabs.scripts.cli.sample_frames.sample_num_frames_total") as mock_sample,
        patch("jabs.scripts.cli.sample_frames.write_frames", side_effect=_capture_write_frames),
    ):
        MockProject.is_valid_project_directory.return_value = True
        mock_project = MagicMock()
        mock_project.settings = {"behavior": {"walking": {}}}
        mock_project.video_manager.videos = []
        MockProject.return_value = mock_project
        mock_collect.return_value = [("v.mp4", 0, 9)]
        mock_sample.return_value = [("v.mp4", 5)]

        result = runner.invoke(
            cli,
            ["sample-frames", "--behavior", "walking", "--num-frames", "1", str(tmp_path)],
        )

    assert result.exit_code == 0, result.output
    assert len(captured_out_dir) == 1
    assert captured_out_dir[0] == Path.cwd()
