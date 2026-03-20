import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

import jabs.scripts.cli.update_pose as update_pose
from jabs.project import TimelineAnnotations, VideoLabels
from jabs.scripts.cli.cli import cli


def _make_pose(identities, boxes_by_identity):
    """Build a simple bbox-capable pose object for remap tests."""
    num_frames = next(iter(boxes_by_identity.values())).shape[0]
    return SimpleNamespace(
        format_major_version=8,
        has_bounding_boxes=True,
        num_frames=num_frames,
        identities=list(identities),
        get_bounding_boxes=lambda identity: boxes_by_identity[identity],
        identity_index_to_display=lambda identity: f"id-{identity}",
    )


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


def test_preflight_allows_timeline_annotations(tmp_path, monkeypatch):
    """Preflight should allow source timeline annotations because they are remapped."""
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
        project_dir, new_pose_dir
    )

    assert videos == ["video1.avi"]
    assert replacement_pose_files == {"video1.avi": new_pose_dir / "video1_pose_est_v8.h5"}
    assert live_annotations == {"video1.avi"}


def test_update_pose_subcommand_invokes_update_project_pose_in_place(tmp_path, monkeypatch):
    """The jabs-cli update-pose subcommand should forward parsed options to the core update helper."""
    project_dir = tmp_path / "project"
    new_pose_dir = tmp_path / "new_pose"
    project_dir.mkdir()
    new_pose_dir.mkdir()

    captured = {}

    def fake_update_project_pose_in_place(
        project_dir_arg,
        new_pose_dir_arg,
        min_iou,
        *,
        verbose,
        annotate_failures,
        drop_timeline_annotations,
        skip_feature_gen,
    ):
        captured.update(
            {
                "project_dir": project_dir_arg,
                "new_pose_dir": new_pose_dir_arg,
                "min_iou": min_iou,
                "verbose": verbose,
                "annotate_failures": annotate_failures,
                "drop_timeline_annotations": drop_timeline_annotations,
                "skip_feature_gen": skip_feature_gen,
            }
        )
        return 3, 1, project_dir / ".backup" / "update_pose_test.zip", "skipped_by_option"

    monkeypatch.setattr(
        update_pose,
        "update_project_pose_in_place",
        fake_update_project_pose_in_place,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "update-pose",
            str(project_dir),
            str(new_pose_dir),
            "--min-iou-thresh",
            "0.75",
            "--verbose",
            "--annotate-failures",
            "--drop-timeline-annotations",
            "--skip-feature-gen",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "project_dir": project_dir,
        "new_pose_dir": new_pose_dir,
        "min_iou": 0.75,
        "verbose": True,
        "annotate_failures": True,
        "drop_timeline_annotations": True,
        "skip_feature_gen": True,
    }
    assert "Backup archive:" in result.output
    assert "Pose update summary: 3 label blocks assigned, 1 label blocks skipped" in result.output
    assert "Feature regeneration: skipped (--skip-feature-gen)." in result.output


def test_load_preexisting_window_sizes_returns_explicit_project_values(tmp_path):
    """Feature regeneration should use only explicit window sizes already stored on disk."""
    project_dir = tmp_path / "project"
    project_file = project_dir / "jabs" / "project.json"
    project_file.parent.mkdir(parents=True)
    project_file.write_text('{"window_sizes": [2, 5, 10]}')

    assert update_pose._load_preexisting_window_sizes(project_dir) == (2, 5, 10)


def test_load_preexisting_window_sizes_returns_empty_when_missing(tmp_path):
    """Missing window sizes in project.json should mean no automatic feature regeneration."""
    project_dir = tmp_path / "project"
    project_file = project_dir / "jabs" / "project.json"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("{}")

    assert update_pose._load_preexisting_window_sizes(project_dir) == ()


def test_regenerate_features_after_update_calls_run_initialize_project(tmp_path, monkeypatch):
    """Automatic regeneration should call the shared jabs-init implementation directly."""
    project_dir = tmp_path / "project"
    backup_path = project_dir / ".backup" / "update_pose_test.zip"

    captured = {}

    def fake_run_initialize_project(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        update_pose,
        "run_initialize_project",
        fake_run_initialize_project,
    )

    status = update_pose._regenerate_features_after_update(
        project_dir,
        (2, 5),
        backup_path,
        skip_feature_gen=False,
    )

    assert status == "regenerated"
    assert captured == {
        "force": False,
        "processes": update_pose.DEFAULT_PROCESSES,
        "window_sizes": (2, 5),
        "force_pixel_distances": False,
        "metadata_path": None,
        "skip_feature_generation": False,
        "project_dir": project_dir,
    }


def test_update_project_pose_in_place_uses_preexisting_window_sizes_for_feature_regen(
    tmp_path, monkeypatch
):
    """The live update flow should pass raw on-disk window sizes into feature regeneration."""
    project_dir = tmp_path / "project"
    new_pose_dir = tmp_path / "new_pose"
    project_file = project_dir / "jabs" / "project.json"
    project_file.parent.mkdir(parents=True)
    project_file.write_text('{"window_sizes": [2, 5]}')
    new_pose_dir.mkdir()

    captured = {}

    monkeypatch.setattr(
        update_pose,
        "_preflight_update_inputs",
        lambda *_args: (
            ["video1.avi"],
            {"video1.avi": new_pose_dir / "video1_pose_est_v8.h5"},
            set(),
        ),
    )

    def fake_create_backup_archive(project_dir_arg, videos_arg):
        captured["backup_project_dir"] = project_dir_arg
        captured["backup_videos"] = videos_arg
        return project_dir_arg / ".backup" / "update_pose_test.zip"

    monkeypatch.setattr(update_pose, "_create_backup_archive", fake_create_backup_archive)
    monkeypatch.setattr(update_pose, "_seed_stage_project", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        update_pose,
        "Project",
        lambda *args, **kwargs: SimpleNamespace(project_paths=SimpleNamespace()),
    )
    monkeypatch.setattr(update_pose, "_run_staged_label_remap", lambda *_args, **_kwargs: (3, 1))
    monkeypatch.setattr(
        update_pose, "_refresh_project_identity_counts", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(update_pose, "_apply_live_update", lambda *_args, **_kwargs: None)

    def fake_regenerate_features_after_update(
        project_dir_arg,
        window_sizes_arg,
        backup_path_arg,
        skip_feature_gen_arg,
    ):
        captured["regen_project_dir"] = project_dir_arg
        captured["regen_window_sizes"] = window_sizes_arg
        captured["regen_backup_path"] = backup_path_arg
        captured["regen_skip_feature_gen"] = skip_feature_gen_arg
        return "regenerated"

    monkeypatch.setattr(
        update_pose,
        "_regenerate_features_after_update",
        fake_regenerate_features_after_update,
    )

    total_success, total_skipped, backup_path, feature_regen_status = (
        update_pose.update_project_pose_in_place(
            project_dir,
            new_pose_dir,
            min_iou=0.5,
        )
    )

    assert (total_success, total_skipped) == (3, 1)
    assert backup_path == project_dir / ".backup" / "update_pose_test.zip"
    assert feature_regen_status == "regenerated"
    assert captured == {
        "backup_project_dir": project_dir.resolve(),
        "backup_videos": ["video1.avi"],
        "regen_project_dir": project_dir.resolve(),
        "regen_window_sizes": (2, 5),
        "regen_backup_path": project_dir.resolve() / ".backup" / "update_pose_test.zip",
        "regen_skip_feature_gen": False,
    }


def test_remap_labels_for_video_remaps_timeline_annotations():
    """Timeline annotations should be preserved and identity-scoped annotations remapped."""
    src_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)
    dst_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)

    source_pose = _make_pose([0], {0: src_boxes})
    dest_pose = _make_pose([1], {1: dst_boxes})

    source_labels = VideoLabels("video1.avi", 10)
    source_labels.add_annotation(
        TimelineAnnotations.Annotation(
            start=1,
            end=2,
            tag="global_note",
            color="#ffffff",
            description="global",
            identity_index=None,
        )
    )
    source_labels.add_annotation(
        TimelineAnnotations.Annotation(
            start=3,
            end=4,
            tag="identity_note",
            color="#ff0000",
            description="identity",
            identity_index=0,
            display_identity="id-0",
        )
    )

    label_source_project = MagicMock()
    label_source_project.video_manager.video_path.return_value = Path("video1.avi")
    label_source_project.video_manager.load_video_labels.return_value = source_labels
    label_source_project.load_pose_est.return_value = source_pose

    label_dest_project = MagicMock()
    label_dest_project.video_manager.video_path.return_value = Path("video1.avi")
    label_dest_project.load_pose_est.return_value = dest_pose
    label_dest_project.project_paths.annotations_dir = Path("/tmp/stage-annotations")

    success_count, skipped_count = update_pose._remap_labels_for_video(
        "video1.avi",
        label_source_project,
        label_dest_project,
        min_iou=0.5,
    )

    assert success_count == 0
    assert skipped_count == 0
    saved_labels = label_dest_project.save_annotations.call_args[0][0]
    assert saved_labels.timeline_annotations.serialize() == [
        {
            "start": 1,
            "end": 2,
            "tag": "global_note",
            "color": "#ffffff",
            "description": "global",
        },
        {
            "start": 3,
            "end": 4,
            "tag": "identity_note",
            "color": "#ff0000",
            "description": "identity",
            "identity": 1,
        },
    ]


def test_remap_labels_for_video_skips_unmatched_identity_annotation(capsys):
    """Identity-scoped timeline annotations should warn and be skipped when no match meets IoU."""
    src_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)
    dst_boxes = np.array([[[50.0, 50.0], [60.0, 60.0]]] * 10)

    source_pose = _make_pose([0], {0: src_boxes})
    dest_pose = _make_pose([1], {1: dst_boxes})

    source_labels = VideoLabels("video1.avi", 10)
    source_labels.add_annotation(
        TimelineAnnotations.Annotation(
            start=3,
            end=4,
            tag="identity_note",
            color="#ff0000",
            description="identity",
            identity_index=0,
            display_identity="id-0",
        )
    )

    label_source_project = MagicMock()
    label_source_project.video_manager.video_path.return_value = Path("video1.avi")
    label_source_project.video_manager.load_video_labels.return_value = source_labels
    label_source_project.load_pose_est.return_value = source_pose

    label_dest_project = MagicMock()
    label_dest_project.video_manager.video_path.return_value = Path("video1.avi")
    label_dest_project.load_pose_est.return_value = dest_pose
    label_dest_project.project_paths.annotations_dir = Path("/tmp/stage-annotations")

    success_count, skipped_count = update_pose._remap_labels_for_video(
        "video1.avi",
        label_source_project,
        label_dest_project,
        min_iou=0.5,
    )

    assert success_count == 0
    assert skipped_count == 0
    saved_labels = label_dest_project.save_annotations.call_args[0][0]
    assert saved_labels.timeline_annotations.serialize() == []

    stderr = capsys.readouterr().err
    assert "annotation tag=identity_note src_id=0 frames=3-4" in stderr
    assert "Skipping annotation." in stderr


def test_remap_labels_for_video_suppresses_duplicate_annotations():
    """Duplicate remapped annotations should only be written once."""
    src_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)
    dst_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)

    source_pose = _make_pose([0, 1], {0: src_boxes, 1: src_boxes})
    dest_pose = _make_pose([2], {2: dst_boxes})

    source_labels = VideoLabels("video1.avi", 10)
    source_labels.add_annotation(
        TimelineAnnotations.Annotation(
            start=3,
            end=4,
            tag="identity_note",
            color="#ff0000",
            description="identity",
            identity_index=0,
            display_identity="id-0",
        )
    )
    source_labels.add_annotation(
        TimelineAnnotations.Annotation(
            start=3,
            end=4,
            tag="identity_note",
            color="#ff0000",
            description="identity",
            identity_index=1,
            display_identity="id-1",
        )
    )

    label_source_project = MagicMock()
    label_source_project.video_manager.video_path.return_value = Path("video1.avi")
    label_source_project.video_manager.load_video_labels.return_value = source_labels
    label_source_project.load_pose_est.return_value = source_pose

    label_dest_project = MagicMock()
    label_dest_project.video_manager.video_path.return_value = Path("video1.avi")
    label_dest_project.load_pose_est.return_value = dest_pose
    label_dest_project.project_paths.annotations_dir = Path("/tmp/stage-annotations")

    success_count, skipped_count = update_pose._remap_labels_for_video(
        "video1.avi",
        label_source_project,
        label_dest_project,
        min_iou=0.5,
    )

    assert success_count == 0
    assert skipped_count == 0
    saved_labels = label_dest_project.save_annotations.call_args[0][0]
    assert saved_labels.timeline_annotations.serialize() == [
        {
            "start": 3,
            "end": 4,
            "tag": "identity_note",
            "color": "#ff0000",
            "description": "identity",
            "identity": 2,
        }
    ]


def test_remap_labels_for_video_drops_source_timeline_annotations():
    """Existing source timeline annotations should be discarded when requested."""
    src_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)
    dst_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)

    source_pose = _make_pose([0], {0: src_boxes})
    dest_pose = _make_pose([1], {1: dst_boxes})

    source_labels = VideoLabels("video1.avi", 10)
    source_labels.add_annotation(
        TimelineAnnotations.Annotation(
            start=1,
            end=2,
            tag="global_note",
            color="#ffffff",
            description="global",
            identity_index=None,
        )
    )
    source_labels.add_annotation(
        TimelineAnnotations.Annotation(
            start=3,
            end=4,
            tag="identity_note",
            color="#ff0000",
            description="identity",
            identity_index=0,
            display_identity="id-0",
        )
    )

    label_source_project = MagicMock()
    label_source_project.video_manager.video_path.return_value = Path("video1.avi")
    label_source_project.video_manager.load_video_labels.return_value = source_labels
    label_source_project.load_pose_est.return_value = source_pose

    label_dest_project = MagicMock()
    label_dest_project.video_manager.video_path.return_value = Path("video1.avi")
    label_dest_project.load_pose_est.return_value = dest_pose
    label_dest_project.project_paths.annotations_dir = Path("/tmp/stage-annotations")

    success_count, skipped_count = update_pose._remap_labels_for_video(
        "video1.avi",
        label_source_project,
        label_dest_project,
        min_iou=0.5,
        drop_timeline_annotations=True,
    )

    assert success_count == 0
    assert skipped_count == 0
    saved_labels = label_dest_project.save_annotations.call_args[0][0]
    assert saved_labels.timeline_annotations.serialize() == []


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

    label_dest_project = SimpleNamespace(
        project_paths=SimpleNamespace(
            annotations_dir=staged_annotations_dir,
            project_file=staged_project_file,
        )
    )

    update_pose._apply_live_update(
        project_dir,
        label_dest_project,
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

    label_dest_project = SimpleNamespace(
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
            label_dest_project,
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
