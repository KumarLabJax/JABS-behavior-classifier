import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import jabs.scripts.update_pose as update_pose


def test_preflight_selects_only_latest_replacement_pose(tmp_path, monkeypatch):
    """Preflight should keep only the latest replacement pose file for each video."""
    project_dir = tmp_path / "project"
    new_pose_dir = tmp_path / "new_pose"
    annotations_dir = project_dir / "jabs" / "annotations"
    annotations_dir.mkdir(parents=True)
    new_pose_dir.mkdir()

    (project_dir / "jabs" / "project.json").write_text("{}")
    (project_dir / "video1.avi").touch()
    (project_dir / "video1_pose_est_v8.h5").touch()
    (annotations_dir / "video1.json").write_text("{}")

    (new_pose_dir / "video1_pose_est_v8.h5").touch()
    (new_pose_dir / "video1_pose_est_v7.h5").touch()

    def fake_open_pose_file(path, _cache_dir):
        return SimpleNamespace(format_major_version=8, has_bounding_boxes=True, num_frames=10)

    monkeypatch.setattr(update_pose, "open_pose_file", fake_open_pose_file)
    monkeypatch.setattr(
        update_pose.VideoReader,
        "get_nframes_from_file",
        staticmethod(lambda _path: 10),
    )

    videos, replacement_pose_files, live_annotations = update_pose._preflight_update_inputs(
        project_dir, new_pose_dir
    )

    assert videos == ["video1.avi"]
    assert replacement_pose_files == {"video1.avi": new_pose_dir / "video1_pose_est_v8.h5"}
    assert live_annotations == {"video1.avi"}


def test_preflight_rejects_nonwritable_live_annotation_file(tmp_path, monkeypatch):
    """Preflight should fail before backup if an existing annotation target is not writable."""
    project_dir = tmp_path / "project"
    new_pose_dir = tmp_path / "new_pose"
    annotations_dir = project_dir / "jabs" / "annotations"
    annotations_dir.mkdir(parents=True)
    new_pose_dir.mkdir()

    (project_dir / "jabs" / "project.json").write_text("{}")
    (project_dir / "video1.avi").touch()
    (project_dir / "video1_pose_est_v8.h5").touch()
    annotation_path = annotations_dir / "video1.json"
    annotation_path.write_text("{}")

    (new_pose_dir / "video1_pose_est_v8.h5").touch()

    def fake_open_pose_file(path, _cache_dir):
        return SimpleNamespace(format_major_version=8, has_bounding_boxes=True, num_frames=10)

    def fake_access(path, mode):
        return not (Path(path) == annotation_path and mode & os.W_OK)

    monkeypatch.setattr(update_pose, "open_pose_file", fake_open_pose_file)
    monkeypatch.setattr(
        update_pose.VideoReader,
        "get_nframes_from_file",
        staticmethod(lambda _path: 10),
    )
    monkeypatch.setattr(update_pose.os, "access", fake_access)

    with pytest.raises(PermissionError, match="live annotation file is not writable"):
        update_pose._preflight_update_inputs(project_dir, new_pose_dir)


def test_preflight_rejects_nonwritable_annotations_directory(tmp_path, monkeypatch):
    """Preflight should fail before backup if the annotations target directory is not writable."""
    project_dir = tmp_path / "project"
    new_pose_dir = tmp_path / "new_pose"
    annotations_dir = project_dir / "jabs" / "annotations"
    annotations_dir.mkdir(parents=True)
    new_pose_dir.mkdir()

    (project_dir / "jabs" / "project.json").write_text("{}")
    (project_dir / "video1.avi").touch()
    (project_dir / "video1_pose_est_v8.h5").touch()
    (annotations_dir / "video1.json").write_text("{}")

    (new_pose_dir / "video1_pose_est_v8.h5").touch()

    def fake_open_pose_file(path, _cache_dir):
        return SimpleNamespace(format_major_version=8, has_bounding_boxes=True, num_frames=10)

    def fake_access(path, mode):
        return not (Path(path) == annotations_dir and mode & os.W_OK)

    monkeypatch.setattr(update_pose, "open_pose_file", fake_open_pose_file)
    monkeypatch.setattr(
        update_pose.VideoReader,
        "get_nframes_from_file",
        staticmethod(lambda _path: 10),
    )
    monkeypatch.setattr(update_pose.os, "access", fake_access)

    with pytest.raises(PermissionError, match="live annotations directory is not writable"):
        update_pose._preflight_update_inputs(project_dir, new_pose_dir)


def test_preflight_rejects_timeline_annotations_without_force(tmp_path, monkeypatch):
    """Preflight should fail fast if source annotations contain timeline annotations."""
    project_dir = tmp_path / "project"
    new_pose_dir = tmp_path / "new_pose"
    annotations_dir = project_dir / "jabs" / "annotations"
    annotations_dir.mkdir(parents=True)
    new_pose_dir.mkdir()

    (project_dir / "jabs" / "project.json").write_text("{}")
    (project_dir / "video1.avi").touch()
    (project_dir / "video1_pose_est_v8.h5").touch()
    (new_pose_dir / "video1_pose_est_v8.h5").touch()
    (annotations_dir / "video1.json").write_text(
        """
        {
          "version": 1,
          "file": "video1.avi",
          "num_frames": 10,
          "labels": {},
          "unfragmented_labels": {},
          "metadata": {"project": {}, "video": {}},
          "annotations": [
            {"start": 1, "end": 2, "tag": "note", "color": "#ffffff"}
          ]
        }
        """
    )

    def fake_open_pose_file(path, _cache_dir):
        return SimpleNamespace(format_major_version=8, has_bounding_boxes=True, num_frames=10)

    monkeypatch.setattr(update_pose, "open_pose_file", fake_open_pose_file)
    monkeypatch.setattr(
        update_pose.VideoReader,
        "get_nframes_from_file",
        staticmethod(lambda _path: 10),
    )

    with pytest.raises(ValueError, match="timeline annotations"):
        update_pose._preflight_update_inputs(project_dir, new_pose_dir)


def test_preflight_allows_timeline_annotations_with_force(tmp_path, monkeypatch):
    """Force mode should allow preflight to continue despite source timeline annotations."""
    project_dir = tmp_path / "project"
    new_pose_dir = tmp_path / "new_pose"
    annotations_dir = project_dir / "jabs" / "annotations"
    annotations_dir.mkdir(parents=True)
    new_pose_dir.mkdir()

    (project_dir / "jabs" / "project.json").write_text("{}")
    (project_dir / "video1.avi").touch()
    (project_dir / "video1_pose_est_v8.h5").touch()
    (new_pose_dir / "video1_pose_est_v8.h5").touch()
    (annotations_dir / "video1.json").write_text(
        """
        {
          "version": 1,
          "file": "video1.avi",
          "num_frames": 10,
          "labels": {},
          "unfragmented_labels": {},
          "metadata": {"project": {}, "video": {}},
          "annotations": [
            {"start": 1, "end": 2, "tag": "note", "color": "#ffffff"}
          ]
        }
        """
    )

    def fake_open_pose_file(path, _cache_dir):
        return SimpleNamespace(format_major_version=8, has_bounding_boxes=True, num_frames=10)

    monkeypatch.setattr(update_pose, "open_pose_file", fake_open_pose_file)
    monkeypatch.setattr(
        update_pose.VideoReader,
        "get_nframes_from_file",
        staticmethod(lambda _path: 10),
    )

    videos, replacement_pose_files, live_annotations = update_pose._preflight_update_inputs(
        project_dir, new_pose_dir, force=True
    )

    assert videos == ["video1.avi"]
    assert replacement_pose_files == {"video1.avi": new_pose_dir / "video1_pose_est_v8.h5"}
    assert live_annotations == {"video1.avi"}


def test_apply_live_update_replaces_pose_set_and_clears_derived_files(tmp_path):
    """Applying a staged pose update should replace annotations/project metadata and swap the pose set."""
    project_dir = tmp_path / "project"
    live_annotations_dir = project_dir / "jabs" / "annotations"
    live_predictions_dir = project_dir / "jabs" / "predictions"
    live_cache_dir = project_dir / "jabs" / "cache"
    live_annotations_dir.mkdir(parents=True)
    live_predictions_dir.mkdir(parents=True)
    live_cache_dir.mkdir(parents=True)

    live_annotation = live_annotations_dir / "video1.json"
    live_annotation.write_text("live-annotation")
    live_project_file = project_dir / "jabs" / "project.json"
    live_project_file.write_text("live-project")
    (project_dir / "video1_pose_est_v8.h5").write_text("old-v8")
    (project_dir / "video1_pose_est_v6.h5").write_text("old-v6")
    (live_predictions_dir / "video1.h5").write_text("prediction")
    (live_cache_dir / "cache.bin").write_text("cache")

    stage_root = tmp_path / "stage"
    staged_annotations_dir = stage_root / "jabs" / "annotations"
    staged_annotations_dir.mkdir(parents=True)
    staged_annotation = staged_annotations_dir / "video1.json"
    staged_annotation.write_text("staged-annotation")
    staged_project_file = stage_root / "jabs" / "project.json"
    staged_project_file.write_text("staged-project")

    new_pose_dir = tmp_path / "new_pose"
    new_pose_dir.mkdir()
    replacement_pose = new_pose_dir / "video1_pose_est_v7.h5"
    replacement_pose.write_text("new-v7")

    dest_project = SimpleNamespace(
        project_paths=SimpleNamespace(
            annotations_dir=staged_annotations_dir,
            project_file=staged_project_file,
        )
    )

    update_pose._apply_live_update(
        project_dir,
        dest_project,
        ["video1.avi"],
        {"video1.avi": replacement_pose},
        {"video1.avi"},
        project_dir / ".backup" / "update_pose_test.zip",
    )

    assert live_annotation.read_text() == "staged-annotation"
    assert live_project_file.read_text() == "staged-project"
    assert not (project_dir / "video1_pose_est_v8.h5").exists()
    assert not (project_dir / "video1_pose_est_v6.h5").exists()
    assert (project_dir / "video1_pose_est_v7.h5").read_text() == "new-v7"
    assert not live_predictions_dir.exists()
    assert not live_cache_dir.exists()


def test_restore_cleanup_paths_includes_created_pose_and_annotation(tmp_path):
    """Cleanup path calculation should include files the failed apply may have created."""
    project_dir = tmp_path / "project"
    staged_annotations_dir = tmp_path / "stage" / "jabs" / "annotations"
    staged_annotations_dir.mkdir(parents=True)
    (staged_annotations_dir / "video1.json").write_text("staged-annotation")

    replacement_pose = tmp_path / "new_pose" / "video1_pose_est_v7.h5"
    replacement_pose.parent.mkdir(parents=True)
    replacement_pose.write_text("new-v7")

    cleanup_paths = update_pose._restore_cleanup_paths(
        project_dir,
        ["video1.avi"],
        {"video1.avi": replacement_pose},
        staged_annotations_dir,
    )

    assert cleanup_paths == [
        project_dir / "jabs" / "annotations" / "video1.json",
        project_dir / "video1_pose_est_v7.h5",
    ]


def test_apply_live_update_failure_prints_cleanup_and_restore_instructions(
    tmp_path, monkeypatch, capsys
):
    """A failed live apply should print cleanup steps for newly created files before restore."""
    project_dir = tmp_path / "project"
    live_annotations_dir = project_dir / "jabs" / "annotations"
    live_predictions_dir = project_dir / "jabs" / "predictions"
    live_annotations_dir.mkdir(parents=True)
    live_predictions_dir.mkdir(parents=True)

    (project_dir / "jabs" / "project.json").write_text("live-project")
    (project_dir / "video1_pose_est_v8.h5").write_text("old-v8")
    (live_predictions_dir / "video1.h5").write_text("prediction")

    stage_root = tmp_path / "stage"
    staged_annotations_dir = stage_root / "jabs" / "annotations"
    staged_annotations_dir.mkdir(parents=True)
    (staged_annotations_dir / "video1.json").write_text("staged-annotation")
    staged_project_file = stage_root / "jabs" / "project.json"
    staged_project_file.write_text("staged-project")

    new_pose_dir = tmp_path / "new_pose"
    new_pose_dir.mkdir()
    replacement_pose = new_pose_dir / "video1_pose_est_v7.h5"
    replacement_pose.write_text("new-v7")

    dest_project = SimpleNamespace(
        project_paths=SimpleNamespace(
            annotations_dir=staged_annotations_dir,
            project_file=staged_project_file,
        )
    )

    def fail_rmtree(_path, ignore_errors=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(update_pose.shutil, "rmtree", fail_rmtree)

    with pytest.raises(SystemExit):
        update_pose._apply_live_update(
            project_dir,
            dest_project,
            ["video1.avi"],
            {"video1.avi": replacement_pose},
            set(),
            project_dir / ".backup" / "update_pose_test.zip",
        )

    stderr = capsys.readouterr().err
    assert "boom" in stderr
    assert "rm -f" in stderr
    assert "video1_pose_est_v7.h5" in stderr
    assert "jabs/annotations/video1.json" in stderr
    assert "unzip -o .backup/update_pose_test.zip" in stderr
