import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import jabs.scripts.cli.update_labels as update_labels
import jabs.scripts.cli.update_pose as update_pose
from jabs.scripts.cli.cli import cli


def _make_valid_project_dir(
    project_dir: Path,
    video_name: str = "video1.avi",
    pose_version: int = 8,
    behaviors: dict | None = None,
    write_annotation: bool = False,
) -> Path:
    """Create a minimal-on-disk JABS project for preflight tests."""
    annotations_dir = project_dir / "jabs" / "annotations"
    annotations_dir.mkdir(parents=True)
    project_file = project_dir / "jabs" / "project.json"
    project_file.write_text(json.dumps({"behavior": behaviors or {}}))
    (project_dir / video_name).touch()
    (project_dir / f"{Path(video_name).stem}_pose_est_v{pose_version}.h5").touch()
    if write_annotation:
        (annotations_dir / f"{Path(video_name).stem}.json").write_text("{}")
    return project_dir


def _stub_pose_reader(monkeypatch, num_frames: int = 10, version: int = 8) -> None:
    """Replace pose-file readers used by the preflight with a permissive stub."""
    monkeypatch.setattr(
        update_pose,
        "open_pose_file",
        lambda *_args, **_kwargs: SimpleNamespace(
            format_major_version=version,
            has_bounding_boxes=True,
            num_frames=num_frames,
        ),
    )
    monkeypatch.setattr(
        update_pose.VideoReader,
        "get_nframes_from_file",
        staticmethod(lambda _path: num_frames),
    )


def test_scan_target_for_init_returns_sorted_videos(tmp_path):
    """The scan should return the sorted list of videos when every video has a paired pose."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "b.mp4").touch()
    (target_dir / "b_pose_est_v8.h5").touch()
    (target_dir / "a.avi").touch()
    (target_dir / "a_pose_est_v8.h5").touch()

    assert update_labels._scan_target_for_init(target_dir) == ["a.avi", "b.mp4"]


def test_scan_target_for_init_rejects_empty_directory(tmp_path):
    """The scan should reject a directory with no video files."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    with pytest.raises(ValueError, match="no video files"):
        update_labels._scan_target_for_init(target_dir)


def test_scan_target_for_init_rejects_video_without_pose(tmp_path):
    """The scan should reject a directory where a video has no paired pose file."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "video1.avi").touch()
    # no pose file
    (target_dir / "video2.avi").touch()
    (target_dir / "video2_pose_est_v8.h5").touch()

    with pytest.raises(ValueError, match="missing pose files for: video1.avi"):
        update_labels._scan_target_for_init(target_dir)


def test_scaffold_target_project_if_missing_creates_jabs_layout(tmp_path, monkeypatch):
    """A bare videos+pose directory should be auto-scaffolded into a JABS project."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "video1.avi").touch()
    (target_dir / "video1_pose_est_v8.h5").touch()

    captured: dict[str, object] = {}

    class FakeProject:
        def __init__(self, project_path, **kwargs):
            captured["project_path"] = project_path
            captured["kwargs"] = kwargs
            (project_path / "jabs").mkdir(parents=True, exist_ok=True)
            (project_path / "jabs" / "project.json").write_text("{}")

    monkeypatch.setattr(update_labels, "Project", FakeProject)
    # The real Project.is_valid_project_directory is a staticmethod we still need.
    monkeypatch.setattr(
        update_labels.Project,
        "is_valid_project_directory",
        staticmethod(lambda d: (d / "jabs" / "project.json").exists()),
        raising=False,
    )

    scaffolded = update_labels._scaffold_target_project_if_missing(target_dir)

    assert scaffolded is True
    assert captured["project_path"] == target_dir
    assert captured["kwargs"]["enable_session_tracker"] is False
    assert (target_dir / "jabs" / "project.json").exists()


def test_scaffold_target_project_if_missing_skips_existing_project(tmp_path, monkeypatch):
    """An already-initialized target should not be re-scaffolded."""
    target_dir = tmp_path / "target"
    (target_dir / "jabs").mkdir(parents=True)
    (target_dir / "jabs" / "project.json").write_text("{}")

    def fail_project_init(*_args, **_kwargs):  # pragma: no cover - guarded by assertion
        raise AssertionError("Project should not be re-constructed when jabs/ exists")

    monkeypatch.setattr(update_labels, "Project", fail_project_init)
    monkeypatch.setattr(
        update_labels.Project,
        "is_valid_project_directory",
        staticmethod(lambda _d: True),
        raising=False,
    )

    assert update_labels._scaffold_target_project_if_missing(target_dir) is False


def test_scaffold_target_project_if_missing_refuses_unpaired_dir(tmp_path):
    """Auto-scaffold must refuse a directory with mismatched video and pose files."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "video1.avi").touch()
    # intentional: no pose file for video1

    with pytest.raises(ValueError, match="missing pose files for: video1.avi"):
        update_labels._scaffold_target_project_if_missing(target_dir)


def test_preflight_returns_labeled_videos_and_live_annotation_set(tmp_path, monkeypatch):
    """The preflight should return source labeled videos and the live-annotated subset."""
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    _make_valid_project_dir(target_dir, write_annotation=True)
    _make_valid_project_dir(source_dir, behaviors={"Grooming": {}}, write_annotation=True)
    _stub_pose_reader(monkeypatch)

    videos, live_annotation_videos = update_labels._preflight_label_update_inputs(
        target_dir, source_dir
    )

    assert videos == ["video1.avi"]
    assert live_annotation_videos == {"video1.avi"}


def test_preflight_rejects_empty_target_dir(tmp_path):
    """The preflight should fail when the target dir has no videos and no jabs/ to scaffold from."""
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    target_dir.mkdir()
    _make_valid_project_dir(source_dir, write_annotation=True)

    with pytest.raises(ValueError, match="no video files"):
        update_labels._preflight_label_update_inputs(target_dir, source_dir)


def test_preflight_auto_scaffolds_target_when_jabs_missing(tmp_path, monkeypatch):
    """A target with videos+pose but no jabs/ should be scaffolded automatically by preflight."""
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    target_dir.mkdir()
    (target_dir / "video1.avi").touch()
    (target_dir / "video1_pose_est_v8.h5").touch()
    _make_valid_project_dir(source_dir, write_annotation=True)
    _stub_pose_reader(monkeypatch)

    real_project = update_labels.Project

    class FakeProject:
        is_valid_project_directory = staticmethod(
            lambda d: (Path(d) / "jabs" / "project.json").exists()
        )

        def __init__(self, project_path, **_kwargs):
            (Path(project_path) / "jabs" / "annotations").mkdir(parents=True, exist_ok=True)
            (Path(project_path) / "jabs" / "project.json").write_text("{}")

    monkeypatch.setattr(update_labels, "Project", FakeProject)
    try:
        videos, _ = update_labels._preflight_label_update_inputs(target_dir, source_dir)
    finally:
        monkeypatch.setattr(update_labels, "Project", real_project)

    assert videos == ["video1.avi"]
    assert (target_dir / "jabs" / "project.json").exists()


def test_preflight_rejects_invalid_source(tmp_path):
    """The preflight should fail when the source is not a valid JABS project."""
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    _make_valid_project_dir(target_dir, write_annotation=True)
    source_dir.mkdir()

    with pytest.raises(ValueError, match="source labels"):
        update_labels._preflight_label_update_inputs(target_dir, source_dir)


def test_preflight_rejects_source_video_missing_in_target(tmp_path, monkeypatch):
    """The preflight should fail when the source has labels for a video not in the target."""
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    _make_valid_project_dir(target_dir, video_name="video1.avi", write_annotation=False)
    _make_valid_project_dir(source_dir, video_name="other.avi", write_annotation=True)
    _stub_pose_reader(monkeypatch)

    with pytest.raises(ValueError, match="missing from target project"):
        update_labels._preflight_label_update_inputs(target_dir, source_dir)


def test_preflight_requires_source_labels(tmp_path, monkeypatch):
    """The preflight should fail when the source project has no annotation files."""
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    _make_valid_project_dir(target_dir, write_annotation=True)
    _make_valid_project_dir(source_dir, write_annotation=False)
    _stub_pose_reader(monkeypatch)

    with pytest.raises(ValueError, match="No labeled videos found"):
        update_labels._preflight_label_update_inputs(target_dir, source_dir)


def test_preflight_skips_unmatched_annotation_files(tmp_path, monkeypatch, capsys):
    """Source annotation files with no matching video file should be skipped with a warning."""
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    _make_valid_project_dir(target_dir, write_annotation=False)
    _make_valid_project_dir(source_dir, write_annotation=True)
    orphan = source_dir / "jabs" / "annotations" / "orphan.json"
    orphan.write_text("{}")
    _stub_pose_reader(monkeypatch)

    videos, live_annotation_videos = update_labels._preflight_label_update_inputs(
        target_dir, source_dir
    )

    assert videos == ["video1.avi"]
    assert live_annotation_videos == set()
    err = capsys.readouterr().err
    assert "orphan.json" in err and "no matching video" in err


def test_preflight_does_not_require_pose_writability(tmp_path, monkeypatch):
    """Preflight should succeed when live pose files are read-only because pose stays put."""
    target_dir = tmp_path / "target"
    source_dir = tmp_path / "source"
    _make_valid_project_dir(target_dir, write_annotation=True)
    _make_valid_project_dir(source_dir, write_annotation=True)
    _stub_pose_reader(monkeypatch)

    pose_path = target_dir / "video1_pose_est_v8.h5"

    def fake_access(path, mode):
        return not (Path(path) == pose_path and mode & os.W_OK)

    monkeypatch.setattr(update_pose.os, "access", fake_access)

    videos, _ = update_labels._preflight_label_update_inputs(target_dir, source_dir)
    assert videos == ["video1.avi"]


def test_merge_source_behaviors_adds_only_missing_entries(tmp_path):
    """Source behaviors should only be added if the target does not already have them."""
    source_dir = tmp_path / "source"
    dest_stage = tmp_path / "dest_stage"
    (source_dir / "jabs").mkdir(parents=True)
    (dest_stage / "jabs").mkdir(parents=True)

    (source_dir / "jabs" / "project.json").write_text(
        json.dumps(
            {
                "behavior": {
                    "Grooming": {"window_size": 5, "source_only": True},
                    "Climbing": {"window_size": 7},
                }
            }
        )
    )
    (dest_stage / "jabs" / "project.json").write_text(
        json.dumps(
            {
                "behavior": {
                    "Grooming": {"window_size": 99, "target_only": True},
                }
            }
        )
    )

    newly_added = update_labels._merge_source_behaviors_into_staged_project(source_dir, dest_stage)

    assert newly_added == ["Climbing"]
    updated = json.loads((dest_stage / "jabs" / "project.json").read_text())
    # Grooming kept target's settings; Climbing added from source.
    assert updated["behavior"]["Grooming"] == {"window_size": 99, "target_only": True}
    assert updated["behavior"]["Climbing"] == {"window_size": 7}


def test_merge_source_behaviors_no_op_when_all_present(tmp_path):
    """No behaviors should be added when the target already has every source behavior."""
    source_dir = tmp_path / "source"
    dest_stage = tmp_path / "dest_stage"
    (source_dir / "jabs").mkdir(parents=True)
    (dest_stage / "jabs").mkdir(parents=True)

    (source_dir / "jabs" / "project.json").write_text(
        json.dumps({"behavior": {"Grooming": {"window_size": 5}}})
    )
    original = json.dumps({"behavior": {"Grooming": {"window_size": 99}}}, indent=2)
    (dest_stage / "jabs" / "project.json").write_text(original)

    newly_added = update_labels._merge_source_behaviors_into_staged_project(source_dir, dest_stage)

    assert newly_added == []
    assert (dest_stage / "jabs" / "project.json").read_text() == original


def test_merge_source_behaviors_creates_behavior_section_when_missing(tmp_path):
    """The merge should add a behavior section to the staged target if missing."""
    source_dir = tmp_path / "source"
    dest_stage = tmp_path / "dest_stage"
    (source_dir / "jabs").mkdir(parents=True)
    (dest_stage / "jabs").mkdir(parents=True)

    (source_dir / "jabs" / "project.json").write_text(
        json.dumps({"behavior": {"Grooming": {"window_size": 5}}})
    )
    (dest_stage / "jabs" / "project.json").write_text(json.dumps({}))

    newly_added = update_labels._merge_source_behaviors_into_staged_project(source_dir, dest_stage)

    assert newly_added == ["Grooming"]
    updated = json.loads((dest_stage / "jabs" / "project.json").read_text())
    assert updated["behavior"] == {"Grooming": {"window_size": 5}}


def test_apply_live_label_update_replaces_annotations_and_clears_predictions(tmp_path):
    """Applying a staged label update should rewrite annotations and clear predictions only."""
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
    pose_file = project_dir / "video1_pose_est_v8.h5"
    pose_file.write_text("pose-v8")
    (live_predictions_dir / "video1.h5").write_text("prediction")
    (live_cache_dir / "cache.bin").write_text("cache")

    stage_root = tmp_path / "stage"
    staged_annotations_dir = stage_root / "jabs" / "annotations"
    staged_annotations_dir.mkdir(parents=True)
    staged_annotation = staged_annotations_dir / "video1.json"
    staged_annotation.write_text("staged-annotation")
    staged_project_file = stage_root / "jabs" / "project.json"
    staged_project_file.write_text("staged-project")

    label_dest_project = SimpleNamespace(
        project_paths=SimpleNamespace(
            annotations_dir=staged_annotations_dir,
            project_file=staged_project_file,
        )
    )

    update_labels._apply_live_label_update(
        project_dir,
        label_dest_project,
        ["video1.avi"],
        project_dir / ".backup" / "update_labels_test.zip",
    )

    assert live_annotation.read_text() == "staged-annotation"
    assert live_project_file.read_text() == "staged-project"
    assert pose_file.read_text() == "pose-v8"
    assert not live_predictions_dir.exists()
    assert (live_cache_dir / "cache.bin").read_text() == "cache"


def test_apply_live_label_update_failure_prints_cleanup_instructions(
    tmp_path, monkeypatch, capsys
):
    """A failed apply should print restore instructions and exit non-zero."""
    project_dir = tmp_path / "project"
    live_annotations_dir = project_dir / "jabs" / "annotations"
    live_annotations_dir.mkdir(parents=True)
    (project_dir / "jabs" / "project.json").write_text("live-project")

    stage_root = tmp_path / "stage"
    staged_annotations_dir = stage_root / "jabs" / "annotations"
    staged_annotations_dir.mkdir(parents=True)
    (staged_annotations_dir / "video1.json").write_text("staged-annotation")
    staged_project_file = stage_root / "jabs" / "project.json"
    staged_project_file.write_text("staged-project")

    label_dest_project = SimpleNamespace(
        project_paths=SimpleNamespace(
            annotations_dir=staged_annotations_dir,
            project_file=staged_project_file,
        )
    )

    def fail_rmtree(_path, ignore_errors=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(update_labels.shutil, "rmtree", fail_rmtree)

    with pytest.raises(SystemExit):
        update_labels._apply_live_label_update(
            project_dir,
            label_dest_project,
            ["video1.avi"],
            project_dir / ".backup" / "update_labels_test.zip",
        )

    stderr = capsys.readouterr().err
    assert "boom" in stderr
    assert "rm -f" in stderr
    assert "jabs/annotations/video1.json" in stderr
    assert "unzip -o .backup/update_labels_test.zip" in stderr


def test_apply_live_label_update_raises_when_staged_annotation_missing(tmp_path):
    """Missing staged output for a labeled video should be a hard error before any writes."""
    project_dir = tmp_path / "project"
    (project_dir / "jabs" / "annotations").mkdir(parents=True)
    stage_root = tmp_path / "stage"
    staged_annotations_dir = stage_root / "jabs" / "annotations"
    staged_annotations_dir.mkdir(parents=True)
    staged_project_file = stage_root / "jabs" / "project.json"
    staged_project_file.write_text("staged-project")

    label_dest_project = SimpleNamespace(
        project_paths=SimpleNamespace(
            annotations_dir=staged_annotations_dir,
            project_file=staged_project_file,
        )
    )

    with pytest.raises(RuntimeError, match="did not produce annotations"):
        update_labels._apply_live_label_update(
            project_dir,
            label_dest_project,
            ["video1.avi"],
            project_dir / ".backup" / "update_labels_test.zip",
        )


def test_update_labels_subcommand_forwards_options(tmp_path, monkeypatch):
    """The jabs-cli update-labels subcommand should forward parsed options to the helper."""
    project_dir = tmp_path / "project"
    source_dir = tmp_path / "source"
    project_dir.mkdir()
    source_dir.mkdir()

    captured = {}

    def fake_update_project_labels_in_place(
        project_dir_arg,
        source_project_dir_arg,
        min_iou,
        *,
        verbose,
        annotate_failures,
        drop_timeline_annotations,
    ):
        captured.update(
            {
                "project_dir": project_dir_arg,
                "source_project_dir": source_project_dir_arg,
                "min_iou": min_iou,
                "verbose": verbose,
                "annotate_failures": annotate_failures,
                "drop_timeline_annotations": drop_timeline_annotations,
            }
        )
        return 4, 2, project_dir / ".backup" / "update_labels_test.zip", ["Climbing"]

    monkeypatch.setattr(
        update_labels,
        "update_project_labels_in_place",
        fake_update_project_labels_in_place,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "update-labels",
            str(project_dir),
            str(source_dir),
            "--min-iou-thresh",
            "0.75",
            "--verbose",
            "--annotate-failures",
            "--drop-timeline-annotations",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "project_dir": project_dir,
        "source_project_dir": source_dir,
        "min_iou": 0.75,
        "verbose": True,
        "annotate_failures": True,
        "drop_timeline_annotations": True,
    }
    assert "Backup archive:" in result.output
    assert "Label update summary: 4 label blocks assigned, 2 label blocks skipped" in result.output
    assert "Behaviors added to project.json from source: Climbing" in result.output


def test_update_labels_subcommand_reports_no_new_behaviors(tmp_path, monkeypatch):
    """The CLI summary should mention when no behaviors were newly added."""
    project_dir = tmp_path / "project"
    source_dir = tmp_path / "source"
    project_dir.mkdir()
    source_dir.mkdir()

    monkeypatch.setattr(
        update_labels,
        "update_project_labels_in_place",
        lambda *_args, **_kwargs: (
            0,
            0,
            project_dir / ".backup" / "update_labels_test.zip",
            [],
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["update-labels", str(project_dir), str(source_dir)],
    )

    assert result.exit_code == 0
    assert "No new behaviors added to project.json." in result.output


def test_update_project_labels_in_place_invokes_pipeline(tmp_path, monkeypatch):
    """The orchestrator should run preflight, backup, staged remap, and apply in order."""
    target_dir = tmp_path / "project"
    source_dir = tmp_path / "source"
    target_dir.mkdir()
    source_dir.mkdir()

    sequence = []

    monkeypatch.setattr(
        update_labels,
        "_preflight_label_update_inputs",
        lambda *_args: (sequence.append("preflight") or (["video1.avi"], set())),
    )

    def fake_create_backup_archive(project_dir_arg, videos_arg, *, include_pose_files, prefix):
        sequence.append("backup")
        assert include_pose_files is False
        assert prefix == "update_labels"
        assert videos_arg == ["video1.avi"]
        return project_dir_arg / ".backup" / "update_labels_test.zip"

    monkeypatch.setattr(update_labels, "_create_backup_archive", fake_create_backup_archive)
    monkeypatch.setattr(
        update_labels,
        "_seed_stage_project",
        lambda *_args, **_kwargs: sequence.append("seed"),
    )
    monkeypatch.setattr(
        update_labels,
        "_merge_source_behaviors_into_staged_project",
        lambda *_args, **_kwargs: (sequence.append("merge_behaviors") or ["Grooming"]),
    )
    monkeypatch.setattr(
        update_labels,
        "Project",
        lambda *args, **kwargs: SimpleNamespace(project_paths=SimpleNamespace()),
    )
    monkeypatch.setattr(
        update_labels,
        "_run_staged_label_remap",
        lambda *_args, **_kwargs: (sequence.append("remap") or (3, 1)),
    )
    monkeypatch.setattr(
        update_labels,
        "_apply_live_label_update",
        lambda *_args, **_kwargs: sequence.append("apply"),
    )

    total_success, total_skipped, backup_path, newly_added = (
        update_labels.update_project_labels_in_place(
            target_dir,
            source_dir,
            min_iou=0.5,
        )
    )

    assert (total_success, total_skipped) == (3, 1)
    assert backup_path == target_dir.resolve() / ".backup" / "update_labels_test.zip"
    assert newly_added == ["Grooming"]
    # Preflight must run before backup, seed before remap, remap before apply.
    assert sequence.index("preflight") < sequence.index("backup")
    assert sequence.index("seed") < sequence.index("remap")
    assert sequence.index("remap") < sequence.index("apply")


def test_update_project_labels_in_place_rejects_identical_dirs(tmp_path):
    """The orchestrator should refuse to operate when source and target are the same path."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with pytest.raises(ValueError, match="must be different"):
        update_labels.update_project_labels_in_place(project_dir, project_dir, min_iou=0.5)
