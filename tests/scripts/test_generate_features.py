"""Tests for the deprecated ``jabs-features`` shim."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import pytest

from jabs.scripts import generate_features

MODULE = "jabs.scripts.generate_features"


def _legacy_args(**overrides) -> argparse.Namespace:
    """Build a legacy argparse namespace with sensible defaults."""
    defaults = {
        "pose_file": Path("session_pose_est_v6.h5"),
        "pose_version": None,
        "feature_dir": Path("features"),
        "cm_units": False,
        "window_size": None,
        "fps": 30,
        "use_pose_hash": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_translation_defaults_to_pixel_distances() -> None:
    """Without --use-cm-distances the legacy pixel default is preserved."""
    result = generate_features._build_compute_features_args(_legacy_args())

    assert result == [
        "--pose-file",
        "session_pose_est_v6.h5",
        "--feature-dir",
        "features",
        "--fps",
        "30",
        "--use-pixel-distances",
    ]


def test_translation_cm_distances_omits_pixel_flag() -> None:
    """--use-cm-distances leaves compute-features on its cm-when-available default."""
    result = generate_features._build_compute_features_args(_legacy_args(cm_units=True))

    assert "--use-pixel-distances" not in result


def test_translation_forwards_window_size_fps_and_pose_hash() -> None:
    """Legacy window size, fps, and pose hash options map onto the new command."""
    result = generate_features._build_compute_features_args(
        _legacy_args(window_size=5, fps=60, use_pose_hash=True)
    )

    assert result[result.index("-w") + 1] == "5"
    assert result[result.index("--fps") + 1] == "60"
    assert "--use-pose-hash" in result


def test_translation_drops_pose_version() -> None:
    """--pose-version is not forwarded, the new command infers it from the filename."""
    result = generate_features._build_compute_features_args(_legacy_args(pose_version=3))

    assert "--pose-version" not in result


def test_main_prints_deprecation_and_runs_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The shim warns on stderr and then invokes compute-features."""
    pose_file = tmp_path / "session_pose_est_v6.h5"
    pose_file.touch()
    feature_dir = tmp_path / "features"

    monkeypatch.setattr(
        "sys.argv",
        [
            "jabs-features",
            "--pose-file",
            str(pose_file),
            "--feature-dir",
            str(feature_dir),
            "--window-size",
            "5",
        ],
    )
    with mock.patch(f"{MODULE}.compute_features_command") as mock_command:
        generate_features.main()

    captured = capsys.readouterr()
    assert "jabs-features is deprecated" in captured.err
    assert "jabs-cli compute-features" in captured.err

    mock_command.main.assert_called_once()
    forwarded = mock_command.main.call_args.args[0]
    assert forwarded == [
        "--pose-file",
        str(pose_file),
        "--feature-dir",
        str(feature_dir),
        "--fps",
        "30",
        "--use-pixel-distances",
        "-w",
        "5",
    ]


def test_main_warns_that_pose_version_is_ignored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Passing the legacy --pose-version produces a warning and is not forwarded."""
    pose_file = tmp_path / "session_pose_est_v6.h5"
    pose_file.touch()

    monkeypatch.setattr(
        "sys.argv",
        [
            "jabs-features",
            "--pose-file",
            str(pose_file),
            "--pose-version",
            "3",
            "--feature-dir",
            str(tmp_path / "features"),
        ],
    )
    with mock.patch(f"{MODULE}.compute_features_command") as mock_command:
        generate_features.main()

    captured = capsys.readouterr()
    assert "--pose-version is ignored" in captured.err
    assert "--pose-version" not in mock_command.main.call_args.args[0]


def test_main_still_requires_pose_file_and_feature_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy required options are still enforced by the shim."""
    monkeypatch.setattr("sys.argv", ["jabs-features", "--pose-file", "session_pose_est_v6.h5"])

    with pytest.raises(SystemExit) as exc_info:
        generate_features.main()

    assert exc_info.value.code != 0
