import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import jabs.scripts.cli.update_pose as update_pose
from jabs.project import TimelineAnnotations, VideoLabels
from jabs.project.timeline_annotations import MAX_TAG_LEN
from jabs.scripts.cli.cli import cli


def _make_pose(identities, boxes_by_identity, *, num_identities=None):
    """Build a simple bbox-capable pose object for remap tests.

    ``num_identities`` defaults to ``len(identities)`` (the consistent case). Pass an
    explicit value to simulate a pose file whose total identity count diverges from
    the iterable returned by ``.identities`` — e.g. annotations that reference an
    identity index the pose can no longer supply.
    """
    num_frames = next(iter(boxes_by_identity.values())).shape[0]
    identity_list = list(identities)
    return SimpleNamespace(
        format_major_version=8,
        has_bounding_boxes=True,
        num_frames=num_frames,
        num_identities=num_identities if num_identities is not None else len(identity_list),
        identities=identity_list,
        get_bounding_boxes=lambda identity: boxes_by_identity[identity],
        identity_index_to_display=lambda identity: f"id-{identity}",
    )


def _write_pose_file_with_metadata(
    pose_path: Path,
    metadata: dict[str, object] | None = None,
    *,
    raw_json: str | None = None,
) -> None:
    """Create a minimal pose H5 file with optional model metadata."""
    with h5py.File(pose_path, "w") as pose_h5:
        pose_grp = pose_h5.create_group("poseest")
        if raw_json is not None:
            pose_grp.attrs["model_metadata_json"] = raw_json
        elif metadata is not None:
            pose_grp.attrs["model_metadata_json"] = json.dumps(metadata)


def _make_metadata_project(pose_paths: dict[str, Path]) -> SimpleNamespace:
    """Build a minimal project stub for metadata injection tests."""
    return SimpleNamespace(
        video_manager=SimpleNamespace(
            videos=list(pose_paths),
            get_cached_pose_path=lambda video: pose_paths[video],
        ),
        settings_manager=MagicMock(),
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
        return SimpleNamespace(
            format_major_version=8, has_bounding_boxes=True, num_frames=10, num_identities=2
        )

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
        return SimpleNamespace(
            format_major_version=8, has_bounding_boxes=True, num_frames=10, num_identities=2
        )

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
        return SimpleNamespace(
            format_major_version=8, has_bounding_boxes=True, num_frames=10, num_identities=2
        )

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
        return SimpleNamespace(
            format_major_version=8, has_bounding_boxes=True, num_frames=10, num_identities=2
        )

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
        tolerate_orphan_identities,
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
                "tolerate_orphan_identities": tolerate_orphan_identities,
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
            "--tolerate-orphan-identities",
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
        "tolerate_orphan_identities": True,
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


def test_inject_consistent_pose_model_metadata_merges_into_project(tmp_path):
    """Consistent replacement pose metadata should be merged into project metadata."""
    metadata = {
        "detection_model_name": "detector:backbone",
        "detection_max_instances": 4,
        "pose_model_name": "pose:backbone",
    }
    pose_a = tmp_path / "video1_pose_est_v8.h5"
    pose_b = tmp_path / "video2_pose_est_v8.h5"
    _write_pose_file_with_metadata(pose_a, metadata)
    _write_pose_file_with_metadata(pose_b, metadata)
    project = _make_metadata_project({"video1.avi": pose_a, "video2.avi": pose_b})

    returned = update_pose._inject_consistent_pose_model_metadata(project)

    assert returned == metadata
    project.settings_manager.set_project_metadata.assert_called_once_with(
        {"project": metadata},
        replace=False,
    )


def test_inject_consistent_pose_model_metadata_warns_when_missing(tmp_path, capsys):
    """Missing metadata across the whole replacement pose set should only warn."""
    pose_a = tmp_path / "video1_pose_est_v8.h5"
    pose_b = tmp_path / "video2_pose_est_v8.h5"
    _write_pose_file_with_metadata(pose_a)
    _write_pose_file_with_metadata(pose_b)
    project = _make_metadata_project({"video1.avi": pose_a, "video2.avi": pose_b})

    returned = update_pose._inject_consistent_pose_model_metadata(project)

    assert returned is None
    project.settings_manager.set_project_metadata.assert_not_called()
    assert "missing model_metadata_json" in capsys.readouterr().err


def test_inject_consistent_pose_model_metadata_rejects_mixed_presence(tmp_path):
    """Replacement pose metadata must be present for every video or none of them."""
    metadata = {"pose_model_name": "pose:backbone"}
    pose_a = tmp_path / "video1_pose_est_v8.h5"
    pose_b = tmp_path / "video2_pose_est_v8.h5"
    _write_pose_file_with_metadata(pose_a, metadata)
    _write_pose_file_with_metadata(pose_b)
    project = _make_metadata_project({"video1.avi": pose_a, "video2.avi": pose_b})

    with pytest.raises(ValueError, match="inconsistent model metadata presence"):
        update_pose._inject_consistent_pose_model_metadata(project)


def test_inject_consistent_pose_model_metadata_rejects_mismatch(tmp_path):
    """Replacement pose metadata should fail when files disagree."""
    pose_a = tmp_path / "video1_pose_est_v8.h5"
    pose_b = tmp_path / "video2_pose_est_v8.h5"
    _write_pose_file_with_metadata(pose_a, {"pose_model_name": "pose:a"})
    _write_pose_file_with_metadata(pose_b, {"pose_model_name": "pose:b"})
    project = _make_metadata_project({"video1.avi": pose_a, "video2.avi": pose_b})

    with pytest.raises(ValueError, match="inconsistent model metadata"):
        update_pose._inject_consistent_pose_model_metadata(project)


def test_inject_consistent_pose_model_metadata_allows_non_whitelist_differences(tmp_path):
    """Differences outside the whitelist should not block metadata injection."""
    pose_a = tmp_path / "video1_pose_est_v8.h5"
    pose_b = tmp_path / "video2_pose_est_v8.h5"
    metadata_a = {
        "pose_model_name": "pose:a",
        "pose_model_version": "1.0.0",
        "detection_model_name": "detector:a",
        "detection_model_version": "2.0.0",
        "config_files": ["a.yaml"],
    }
    metadata_b = {
        "pose_model_name": "pose:a",
        "pose_model_version": "1.0.0",
        "detection_model_name": "detector:a",
        "detection_model_version": "2.0.0",
        "config_files": ["b.yaml"],
    }
    _write_pose_file_with_metadata(pose_a, metadata_a)
    _write_pose_file_with_metadata(pose_b, metadata_b)
    project = _make_metadata_project({"video1.avi": pose_a, "video2.avi": pose_b})

    returned = update_pose._inject_consistent_pose_model_metadata(project)

    assert returned == metadata_a
    project.settings_manager.set_project_metadata.assert_called_once_with(
        {"project": metadata_a},
        replace=False,
    )


def test_inject_consistent_pose_model_metadata_rejects_inconsistent_missing_whitelist_key(
    tmp_path,
):
    """A whitelisted key must be either present for every file or absent for every file."""
    pose_a = tmp_path / "video1_pose_est_v8.h5"
    pose_b = tmp_path / "video2_pose_est_v8.h5"
    _write_pose_file_with_metadata(
        pose_a,
        {
            "pose_model_name": "pose:a",
            "detection_model_name": "detector:a",
            "detection_model_version": "2.0.0",
        },
    )
    _write_pose_file_with_metadata(
        pose_b,
        {
            "pose_model_name": "pose:a",
            "pose_model_version": "1.0.0",
            "detection_model_name": "detector:a",
            "detection_model_version": "2.0.0",
        },
    )
    project = _make_metadata_project({"video1.avi": pose_a, "video2.avi": pose_b})

    with pytest.raises(ValueError, match="pose_model_version"):
        update_pose._inject_consistent_pose_model_metadata(project)


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
        lambda *_args, **_kwargs: (
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
    monkeypatch.setattr(
        update_pose, "_inject_consistent_pose_model_metadata", lambda *_args, **_kwargs: None
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


def test_run_staged_label_remap_uses_explicit_videos_list_when_provided(monkeypatch):
    """When ``videos`` is passed, ``_run_staged_label_remap`` should only iterate that list."""
    source_videos = ["a.avi", "b.avi"]
    captured = []

    def fake_remap(video, *_args, **_kwargs):
        captured.append(video)
        return (1, 0)

    monkeypatch.setattr(update_pose, "_remap_labels_for_video", fake_remap)

    label_source_project = MagicMock()
    label_source_project.video_manager.videos = source_videos
    label_dest_project = MagicMock()

    success, skipped = update_pose._run_staged_label_remap(
        label_source_project,
        label_dest_project,
        min_iou=0.5,
        verbose=False,
        annotate_failures=False,
        drop_timeline_annotations=False,
        videos=["a.avi"],
    )

    assert captured == ["a.avi"]
    assert (success, skipped) == (1, 0)


def test_run_staged_label_remap_falls_back_to_source_videos(monkeypatch):
    """When ``videos`` is None, iteration should default to ``label_source_project``'s videos."""
    captured = []

    def fake_remap(video, *_args, **_kwargs):
        captured.append(video)
        return (1, 0)

    monkeypatch.setattr(update_pose, "_remap_labels_for_video", fake_remap)

    label_source_project = MagicMock()
    label_source_project.video_manager.videos = ["a.avi", "b.avi"]
    label_dest_project = MagicMock()

    update_pose._run_staged_label_remap(
        label_source_project,
        label_dest_project,
        min_iou=0.5,
        verbose=False,
        annotate_failures=False,
        drop_timeline_annotations=False,
    )

    assert captured == ["a.avi", "b.avi"]


def test_remap_labels_for_video_failure_tag_fits_within_max_tag_len():
    """Failure tags must fit within the timeline-annotation MAX_TAG_LEN limit.

    Without this constraint the failure annotation would be written to disk but
    silently dropped on the next load (``TimelineAnnotations.load`` rejects tags
    longer than ``MAX_TAG_LEN``). The description phrase customization should not
    influence the tag itself.
    """
    src_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)
    dst_boxes = np.array([[[50.0, 50.0], [60.0, 60.0]]] * 10)

    source_pose = _make_pose([0], {0: src_boxes})
    dest_pose = _make_pose([1], {1: dst_boxes})

    source_labels = VideoLabels("video1.avi", 10)
    track_labels = source_labels.get_track_labels("0", "Grooming")
    track_labels.label_behavior(3, 4)

    label_source_project = MagicMock()
    label_source_project.video_manager.video_path.return_value = Path("video1.avi")
    label_source_project.video_manager.load_video_labels.return_value = source_labels
    label_source_project.load_pose_est.return_value = source_pose

    label_dest_project = MagicMock()
    label_dest_project.video_manager.video_path.return_value = Path("video1.avi")
    label_dest_project.load_pose_est.return_value = dest_pose
    label_dest_project.project_paths.annotations_dir = Path("/tmp/stage-annotations")

    update_pose._remap_labels_for_video(
        "video1.avi",
        label_source_project,
        label_dest_project,
        min_iou=0.5,
        annotate_failures=True,
        failure_description_phrase="label update",
    )

    saved_labels = label_dest_project.save_annotations.call_args[0][0]
    annotations = saved_labels.timeline_annotations.serialize()
    assert len(annotations) == 1
    assert annotations[0]["tag"] == "behavior-remap-failed"
    assert len(annotations[0]["tag"]) <= MAX_TAG_LEN
    assert "label remap failed during label update" in annotations[0]["description"]


def test_orphan_identities_in_annotation_detects_label_track_keys(tmp_path):
    """Identity keys in labels/unfragmented_labels above num_identities are reported."""
    annotation_path = tmp_path / "video1.json"
    annotation_path.write_text(
        json.dumps(
            {
                "file": "video1.avi",
                "num_frames": 10,
                "unfragmented_labels": {
                    "0": {"Grooming": []},
                    "2": {"Grooming": []},
                    "3": {"Grooming": []},
                },
                "labels": {},
            }
        )
    )

    assert update_pose._orphan_identities_in_annotation(annotation_path, 2) == [2, 3]


def test_orphan_identities_in_annotation_detects_timeline_annotation_identity(tmp_path):
    """Identity-scoped timeline annotation entries are also checked."""
    annotation_path = tmp_path / "video1.json"
    annotation_path.write_text(
        json.dumps(
            {
                "file": "video1.avi",
                "num_frames": 10,
                "labels": {},
                "annotations": [
                    {"start": 1, "end": 2, "tag": "note", "color": "#fff"},
                    {"start": 3, "end": 4, "tag": "n", "color": "#fff", "identity": 0},
                    {"start": 5, "end": 6, "tag": "n", "color": "#fff", "identity": 5},
                ],
            }
        )
    )

    assert update_pose._orphan_identities_in_annotation(annotation_path, 2) == [5]


def test_orphan_identities_in_annotation_handles_clean_files(tmp_path):
    """A consistent annotation file yields an empty list."""
    annotation_path = tmp_path / "video1.json"
    annotation_path.write_text(
        json.dumps(
            {
                "file": "video1.avi",
                "num_frames": 10,
                "unfragmented_labels": {"0": {"Grooming": []}, "1": {"Grooming": []}},
                "labels": {},
            }
        )
    )

    assert update_pose._orphan_identities_in_annotation(annotation_path, 2) == []


def test_orphan_identities_in_annotation_missing_file_returns_empty(tmp_path):
    """A missing annotation file is treated as nothing to check rather than an error."""
    assert update_pose._orphan_identities_in_annotation(tmp_path / "nope.json", 2) == []


def test_handle_orphan_identities_raises_by_default_listing_every_offender():
    """The composite error must list every video and identity, not just the first."""
    issues = [
        ("video1.avi", 2, [2, 3]),
        ("video2.avi", 3, [4]),
    ]

    with pytest.raises(ValueError) as excinfo:
        update_pose._handle_orphan_identities(issues, source_label="source")

    msg = str(excinfo.value)
    assert "video1.avi" in msg
    assert "video2.avi" in msg
    assert "identity 2, 3" in msg
    assert "identity 4" in msg
    # Should describe the remediation, not just the problem.
    assert "Clean up" in msg or "clean up" in msg
    # Should mention the escape-hatch flag so users discover it from the error.
    assert "--tolerate-orphan-identities" in msg


def test_handle_orphan_identities_warns_when_tolerated(capsys):
    """With ``tolerate=True``, issues are reported via stderr and execution continues."""
    issues = [("video1.avi", 2, [2, 3])]

    update_pose._handle_orphan_identities(issues, source_label="source", tolerate=True)

    stderr = capsys.readouterr().err
    assert "WARNING" in stderr
    assert "video1.avi" in stderr
    assert "identity 2, 3" in stderr
    assert "--tolerate-orphan-identities" in stderr


def test_handle_orphan_identities_noop_when_empty():
    """No exception, no output, regardless of tolerate."""
    update_pose._handle_orphan_identities([])
    update_pose._handle_orphan_identities([], tolerate=True)


def test_handle_orphan_identities_message_handles_zero_identities():
    """A pose with ``num_identities == 0`` must not render an inverted range."""
    issues = [("video1.avi", 0, [0, 1])]

    with pytest.raises(ValueError) as excinfo:
        update_pose._handle_orphan_identities(issues, source_label="source")

    msg = str(excinfo.value)
    assert "pose has 0 identities" in msg
    assert "no valid indices" in msg
    # Guard against a regression to the "0 to -1" phrasing.
    assert "-1" not in msg


def test_preflight_update_inputs_raises_on_orphan_source_identities(tmp_path, monkeypatch):
    """Preflight must abort before any mutation when source labels reference orphans."""
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
        json.dumps(
            {
                "file": "video1.avi",
                "num_frames": 10,
                "unfragmented_labels": {"0": {"Grooming": []}, "2": {"Grooming": []}},
                "labels": {},
            }
        )
    )

    def fake_open_pose_file(path, _cache_dir):
        return SimpleNamespace(
            format_major_version=8, has_bounding_boxes=True, num_frames=10, num_identities=2
        )

    monkeypatch.setattr(update_pose, "open_pose_file", fake_open_pose_file)
    monkeypatch.setattr(
        update_pose.VideoReader,
        "get_nframes_from_file",
        staticmethod(lambda _path: 10),
    )

    with pytest.raises(ValueError, match=r"video1\.avi.*identity 2"):
        update_pose._preflight_update_inputs(project_dir, new_pose_dir)


def test_preflight_update_inputs_tolerates_orphans_and_warns(tmp_path, monkeypatch, capsys):
    """``tolerate_orphan_identities=True`` swaps the preflight error for a warning."""
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
        json.dumps(
            {
                "file": "video1.avi",
                "num_frames": 10,
                "unfragmented_labels": {"0": {"Grooming": []}, "2": {"Grooming": []}},
                "labels": {},
            }
        )
    )

    def fake_open_pose_file(path, _cache_dir):
        return SimpleNamespace(
            format_major_version=8, has_bounding_boxes=True, num_frames=10, num_identities=2
        )

    monkeypatch.setattr(update_pose, "open_pose_file", fake_open_pose_file)
    monkeypatch.setattr(
        update_pose.VideoReader,
        "get_nframes_from_file",
        staticmethod(lambda _path: 10),
    )

    videos, _, live_annotations = update_pose._preflight_update_inputs(
        project_dir, new_pose_dir, tolerate_orphan_identities=True
    )

    assert videos == ["video1.avi"]
    assert "video1.avi" in live_annotations

    stderr = capsys.readouterr().err
    assert "WARNING" in stderr
    assert "identity 2" in stderr


def test_remap_labels_for_video_inline_guard_skips_orphan_blocks(capsys):
    """When preflight is tolerated, the inline guard skips orphan (identity, behavior) pairs.

    Without the guard, ``_find_best_identity`` would crash on
    ``_bboxes_for_identity(src_pose, src_identity)`` for any out-of-range source
    identity. The guard preserves all valid identities' blocks and reports a
    single warning per skipped pair.
    """
    src_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)
    dst_boxes = np.array([[[0.0, 0.0], [10.0, 10.0]]] * 10)

    source_pose = _make_pose([0, 1], {0: src_boxes, 1: src_boxes}, num_identities=2)
    dest_pose = _make_pose([0, 1, 2], {0: dst_boxes, 1: dst_boxes, 2: dst_boxes})

    source_labels = VideoLabels("video1.avi", 10)
    source_labels.get_track_labels("0", "Grooming").label_behavior(1, 2)
    orphan_track = source_labels.get_track_labels("2", "Grooming")
    orphan_track.label_behavior(3, 4)
    orphan_track.label_not_behavior(5, 6)

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
        annotate_failures=True,
    )

    assert success_count == 1
    assert skipped_count == 2

    stderr = capsys.readouterr().err
    assert "src_id=2 not present in source pose" in stderr
    assert "behavior=Grooming" in stderr

    # No failure annotation is written for the orphan blocks even with
    # --annotate-failures, since the root cause is in source.
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

    def fail_rmtree(_path):
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
