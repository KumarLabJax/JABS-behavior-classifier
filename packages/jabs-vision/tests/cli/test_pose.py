"""Tests for the jabs-pose CLI core (CPU, no real weights)."""

from unittest import mock

import h5py
import numpy as np
import pytest
import torch

import jabs.vision.cli.pose as pose_cli
from jabs.core.abstract.pose_est import MINIMUM_CONFIDENCE
from jabs.vision.cli.pose import _build_pose_data, run_pose_inference


class _DummyPoseModel(torch.nn.Module):
    """Returns a fixed (B, 12, 4, 4) heatmap so argmax is deterministic."""

    def forward(self, x):
        """Return a heatmap with a peak at (row=1, col=2) for every keypoint."""
        batch = x.shape[0]
        hm = torch.zeros(batch, 12, 4, 4)
        hm[:, :, 1, 2] = 1.0
        return hm


def test_run_pose_inference_writes_v2(monkeypatch, tmp_path):
    """run_pose_inference writes a valid v2 file with on-disk (y, x) coordinates."""
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(
        pose_cli, "load_pose_model", mock.Mock(return_value=(_DummyPoseModel(), None))
    )
    monkeypatch.setattr(pose_cli, "read_frames", mock.Mock(return_value=iter(frames)))
    monkeypatch.setattr(pose_cli, "video_fps", mock.Mock(return_value=30))

    out = tmp_path / "clip_pose_est_v2.h5"
    run_pose_inference(
        video=tmp_path / "clip.mp4",
        out=out,
        config=tmp_path / "gait.yaml",
        checkpoint=tmp_path / "gait.pth",
    )

    with h5py.File(out, "r") as h5:
        assert h5["poseest"].attrs["version"].tolist() == [2, 0]
        assert h5["poseest/points"].shape == (3, 12, 2)
        assert h5["poseest/confidence"].shape == (3, 12)
        # dummy peak (row=1, col=2) -> stored on-disk as (y, x) = (1, 2)
        assert h5["poseest/points"][0, 0].tolist() == [1, 2]


def test_build_pose_data_derives_point_mask_from_confidence(tmp_path):
    """point_mask reflects the confidence threshold, matching the v2 reader semantics."""
    pose_yx = np.zeros((1, 12, 2), dtype=np.uint16)
    confidence = np.full((1, 12), 0.9, dtype=np.float32)
    confidence[0, 3] = 0.1  # below MINIMUM_CONFIDENCE
    pose_data = _build_pose_data(
        pose_yx, confidence, fps=30, config=tmp_path / "g.yaml", checkpoint=tmp_path / "g.pth"
    )
    assert bool(pose_data.point_mask[0, 0, 3]) is False
    assert pose_data.point_mask[0, 0, 0]  # high-confidence keypoint stays valid
    # sanity: threshold used is the shared jabs-core constant
    assert 0.1 < MINIMUM_CONFIDENCE < 0.9


def test_run_pose_inference_loads_strict(monkeypatch, tmp_path):
    """The CLI enforces strict checkpoint loading (guards silent partial loads)."""
    loader = mock.Mock(return_value=(_DummyPoseModel(), None))
    monkeypatch.setattr(pose_cli, "load_pose_model", loader)
    monkeypatch.setattr(
        pose_cli, "read_frames", mock.Mock(return_value=iter([np.zeros((8, 8, 3), np.uint8)]))
    )
    monkeypatch.setattr(pose_cli, "video_fps", mock.Mock(return_value=30))

    run_pose_inference(
        video=tmp_path / "c.mp4",
        out=tmp_path / "c_pose_est_v2.h5",
        config=tmp_path / "g.yaml",
        checkpoint=tmp_path / "g.pth",
    )
    assert loader.call_args.kwargs["strict"] is True


def test_run_pose_inference_empty_video_raises(monkeypatch, tmp_path):
    """An empty video fails loudly instead of writing a zero-frame file."""
    monkeypatch.setattr(
        pose_cli, "load_pose_model", mock.Mock(return_value=(_DummyPoseModel(), None))
    )
    monkeypatch.setattr(pose_cli, "read_frames", mock.Mock(return_value=iter([])))
    monkeypatch.setattr(pose_cli, "video_fps", mock.Mock(return_value=30))

    with pytest.raises(ValueError, match="no frames"):
        run_pose_inference(
            video=tmp_path / "c.mp4",
            out=tmp_path / "c_pose_est_v2.h5",
            config=tmp_path / "g.yaml",
            checkpoint=tmp_path / "g.pth",
        )
