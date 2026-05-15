"""Tests for the compute-features CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from jabs.core.enums import CacheFormat, ProjectDistanceUnit
from jabs.scripts.cli.cli import cli

MODULE = "jabs.scripts.cli.compute_features"


def _make_pose_file(tmp_path: Path, name: str = "session_pose_est_v6.h5") -> Path:
    pose_file = tmp_path / name
    pose_file.touch()
    return pose_file


def _fake_pose(cm_per_pixel: float | None = None, identities: tuple[int, ...] = (0,)) -> MagicMock:
    pose = MagicMock()
    pose.cm_per_pixel = cm_per_pixel
    pose.identities = identities
    return pose


def test_invalid_filename_rejected(tmp_path: Path) -> None:
    """Filename without a recognizable _v<N>.h5 produces a clear error."""
    bad_pose = tmp_path / "poses.h5"
    bad_pose.touch()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["compute-features", "--pose-file", str(bad_pose), "--feature-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "pose version" in result.output


def test_window_size_rejects_zero(tmp_path: Path) -> None:
    """--window-size 0 fails argument validation."""
    pose_file = _make_pose_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compute-features",
            "--pose-file",
            str(pose_file),
            "--feature-dir",
            str(tmp_path),
            "--window-size",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "x>=1" in result.output or "Invalid value" in result.output


def test_window_size_rejects_negative(tmp_path: Path) -> None:
    """--window-size -1 fails argument validation."""
    pose_file = _make_pose_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compute-features",
            "--pose-file",
            str(pose_file),
            "--feature-dir",
            str(tmp_path),
            "-w",
            "-1",
        ],
    )
    assert result.exit_code != 0


def test_fps_rejects_zero(tmp_path: Path) -> None:
    """--fps 0 fails argument validation."""
    pose_file = _make_pose_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compute-features",
            "--pose-file",
            str(pose_file),
            "--feature-dir",
            str(tmp_path),
            "--fps",
            "0",
        ],
    )
    assert result.exit_code != 0


def test_default_cm_when_pose_has_scale(tmp_path: Path) -> None:
    """When pose file provides cm_per_pixel, distance unit defaults to CM."""
    pose_file = _make_pose_file(tmp_path)
    pose = _fake_pose(cm_per_pixel=0.05)
    with (
        patch(f"{MODULE}.open_pose_file", return_value=pose),
        patch(f"{MODULE}.Project.settings_by_pose_version") as mock_settings,
        patch(f"{MODULE}.IdentityFeatures") as mock_features_cls,
    ):
        mock_settings.return_value = {"social": False}
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compute-features",
                "--pose-file",
                str(pose_file),
                "--feature-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_settings.assert_called_once_with(6, ProjectDistanceUnit.CM)
    mock_features_cls.assert_called_once()


def test_default_pixel_when_pose_lacks_scale(tmp_path: Path) -> None:
    """When pose file lacks cm_per_pixel, distance unit defaults to PIXEL."""
    pose_file = _make_pose_file(tmp_path)
    pose = _fake_pose(cm_per_pixel=None)
    with (
        patch(f"{MODULE}.open_pose_file", return_value=pose),
        patch(f"{MODULE}.Project.settings_by_pose_version") as mock_settings,
        patch(f"{MODULE}.IdentityFeatures"),
    ):
        mock_settings.return_value = {"social": False}
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compute-features",
                "--pose-file",
                str(pose_file),
                "--feature-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_settings.assert_called_once_with(6, ProjectDistanceUnit.PIXEL)


def test_use_pixel_distances_overrides_cm_default(tmp_path: Path) -> None:
    """--use-pixel-distances forces PIXEL even when the pose file has scale."""
    pose_file = _make_pose_file(tmp_path)
    pose = _fake_pose(cm_per_pixel=0.05)
    with (
        patch(f"{MODULE}.open_pose_file", return_value=pose),
        patch(f"{MODULE}.Project.settings_by_pose_version") as mock_settings,
        patch(f"{MODULE}.IdentityFeatures"),
    ):
        mock_settings.return_value = {"social": False}
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compute-features",
                "--pose-file",
                str(pose_file),
                "--feature-dir",
                str(tmp_path),
                "--use-pixel-distances",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_settings.assert_called_once_with(6, ProjectDistanceUnit.PIXEL)


def test_identity_features_invoked_with_expected_args(tmp_path: Path) -> None:
    """IdentityFeatures is constructed once per identity with the right kwargs."""
    pose_file = _make_pose_file(tmp_path, "rec_pose_est_v8.h5")
    feature_dir = tmp_path / "feats"
    pose = _fake_pose(cm_per_pixel=0.04, identities=(0, 1))
    settings = {"social": True}

    fake_features = MagicMock()
    with (
        patch(f"{MODULE}.open_pose_file", return_value=pose),
        patch(f"{MODULE}.Project.settings_by_pose_version", return_value=settings),
        patch(f"{MODULE}.IdentityFeatures", return_value=fake_features) as mock_features_cls,
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compute-features",
                "--pose-file",
                str(pose_file),
                "--feature-dir",
                str(feature_dir),
                "--fps",
                "60",
                "--cache-format",
                "hdf5",
                "--use-pose-hash",
            ],
        )

    assert result.exit_code == 0, result.output
    assert mock_features_cls.call_count == 2
    for call, expected_id in zip(mock_features_cls.call_args_list, (0, 1), strict=True):
        args, kwargs = call
        assert args == (pose_file, expected_id, feature_dir, pose)
        assert kwargs == {
            "force": False,
            "fps": 60,
            "op_settings": settings,
            "cache_window": False,
            "cache_format": CacheFormat.HDF5,
            "include_pose_hash": True,
        }
    fake_features.get_window_features.assert_not_called()


def test_window_sizes_iterated_dedup_sorted(tmp_path: Path) -> None:
    """Repeated -w values are deduplicated, sorted, and each triggers a window feature call."""
    pose_file = _make_pose_file(tmp_path)
    pose = _fake_pose(cm_per_pixel=0.05, identities=(0,))

    fake_features = MagicMock()
    with (
        patch(f"{MODULE}.open_pose_file", return_value=pose),
        patch(f"{MODULE}.Project.settings_by_pose_version", return_value={"social": False}),
        patch(f"{MODULE}.IdentityFeatures", return_value=fake_features) as mock_features_cls,
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compute-features",
                "--pose-file",
                str(pose_file),
                "--feature-dir",
                str(tmp_path),
                "-w",
                "10",
                "-w",
                "5",
                "-w",
                "10",
            ],
        )

    assert result.exit_code == 0, result.output
    _, kwargs = mock_features_cls.call_args
    assert kwargs["cache_window"] is True
    assert fake_features.get_window_features.call_count == 2
    called_sizes = [call.args[0] for call in fake_features.get_window_features.call_args_list]
    assert called_sizes == [5, 10]
    for call in fake_features.get_window_features.call_args_list:
        assert call.kwargs == {"force": False}


def test_force_flag_forwarded(tmp_path: Path) -> None:
    """--force is forwarded to IdentityFeatures and get_window_features."""
    pose_file = _make_pose_file(tmp_path)
    pose = _fake_pose(cm_per_pixel=0.05, identities=(0,))

    fake_features = MagicMock()
    with (
        patch(f"{MODULE}.open_pose_file", return_value=pose),
        patch(f"{MODULE}.Project.settings_by_pose_version", return_value={"social": False}),
        patch(f"{MODULE}.IdentityFeatures", return_value=fake_features) as mock_features_cls,
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compute-features",
                "--pose-file",
                str(pose_file),
                "--feature-dir",
                str(tmp_path),
                "-w",
                "5",
                "--force",
            ],
        )

    assert result.exit_code == 0, result.output
    _, kwargs = mock_features_cls.call_args
    assert kwargs["force"] is True
    fake_features.get_window_features.assert_called_once_with(5, force=True)
