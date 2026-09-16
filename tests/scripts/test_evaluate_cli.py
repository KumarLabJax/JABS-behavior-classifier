"""Tests for the ``jabs-cli evaluate`` command.

The pure comparison and reporting logic is tested directly. The heavy
``run_evaluation`` implementation, which needs a real project and pose files on
disk, is replaced with a spy for the tests that exercise the Click wiring.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner

import jabs.scripts.cli.evaluate as evaluate_module
from jabs.behavior.evaluation import IoUCriterion, OverlapCriterion
from jabs.classifier import MultiClassClassifier
from jabs.scripts.cli.cli import cli
from jabs.scripts.cli.evaluate import (
    GROUND_TRUTH_SOURCE,
    PREDICTED_SOURCE,
    _resolve_pipeline_config,
    build_bout_records,
    compare_identity,
    load_binary_classifier,
    resolve_behavior,
)
from jabs.scripts.cli.evaluate_report import (
    build_summary,
    render_markdown,
    write_csv,
    write_json,
    write_markdown,
)
from jabs.scripts.cli.evaluate_results import (
    POSTPROCESSED_STAGE,
    RAW_STAGE,
    EvaluationResult,
    IdentityResult,
    aggregate_bout_metrics,
    aggregate_frame_metrics,
    format_rate,
)

OVERLAP = OverlapCriterion(1)
IOU = IoUCriterion(0.5)
CRITERIA = [OVERLAP, IOU]

# -----------------------------------------------------------------------------
# compare_identity
# -----------------------------------------------------------------------------


def test_compare_identity_reports_both_criteria() -> None:
    """Compare identity reports both criteria."""
    truth = np.array([0, 1, 1, 1, 1, 0])
    predicted = np.array([0, 0, 1, 1, 0, 0])
    comparison = compare_identity(truth, predicted, CRITERIA)

    assert set(comparison.bout_metrics) == {OVERLAP.label, IOU.label}
    assert comparison.frame_metrics.true_positive == 2
    assert comparison.truth_bouts == [evaluate_module.Bout(1, 4)]
    assert comparison.predicted_bouts == [evaluate_module.Bout(2, 3)]


def test_compare_identity_criteria_can_disagree() -> None:
    """A 2-frame prediction inside an 8-frame bout: found, but poorly bounded."""
    truth = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    predicted = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    comparison = compare_identity(truth, predicted, CRITERIA)

    assert comparison.bout_metrics[OVERLAP.label].detection_rate == pytest.approx(1.0)
    assert comparison.bout_metrics[IOU.label].detection_rate == pytest.approx(0.0)


def test_compare_identity_shares_one_overlap_sweep_across_criteria() -> None:
    """Compare identity shares one overlap sweep across criteria."""
    truth = np.array([1, 1, 0, 1, 1])
    predicted = np.array([1, 1, 1, 1, 1])
    comparison = compare_identity(truth, predicted, CRITERIA)
    # one prediction spanning both truth bouts -> two overlapping pairs
    assert len(comparison.overlaps) == 2


def test_compare_identity_with_no_bouts_on_either_side() -> None:
    """Compare identity with no bouts on either side."""
    zeros = np.zeros(10, dtype=np.int8)
    comparison = compare_identity(zeros, zeros, CRITERIA)
    assert comparison.truth_bouts == []
    assert comparison.bout_metrics[OVERLAP.label].detection_rate is None
    assert comparison.frame_metrics.true_negative == 10


# -----------------------------------------------------------------------------
# build_bout_records
# -----------------------------------------------------------------------------


def _records(truth, predicted):
    """Build bout records for one identity from two vectors."""
    comparison = compare_identity(np.asarray(truth), np.asarray(predicted), CRITERIA)
    return build_bout_records(comparison, "v.mp4", 0, RAW_STAGE, OVERLAP.label, IOU.label)


def test_bout_records_cover_both_sides() -> None:
    """Bout records cover both sides."""
    records = _records([1, 1, 0, 0], [0, 1, 1, 0])
    sources = [r.source for r in records]
    assert sources == [GROUND_TRUTH_SOURCE, PREDICTED_SOURCE]
    assert all(r.video == "v.mp4" and r.identity == 0 and r.stage == RAW_STAGE for r in records)


def test_bout_record_carries_best_overlap_and_iou() -> None:
    """Bout record carries best overlap and iou."""
    #  truth 0-3 (4 frames), prediction 2-5 (4 frames) -> 2 shared, union 6
    (truth_record, predicted_record) = _records([1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1])
    assert truth_record.best_overlap_frames == 2
    assert truth_record.best_iou == pytest.approx(1 / 3, abs=1e-4)
    assert predicted_record.best_overlap_frames == 2
    assert truth_record.overlapping_bouts == 1


def test_bout_record_match_flags_follow_the_two_criteria() -> None:
    """Bout record match flags follow the two criteria."""
    truth = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    predicted = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    truth_record = _records(truth, predicted)[0]
    assert truth_record.matched_overlap is True
    assert truth_record.matched_iou is False


def test_bout_record_marks_an_unevaluable_truth_bout() -> None:
    """Bout record marks an unevaluable truth bout."""
    records = _records([1, 1, 0, 0], [-1, -1, 0, 0])
    truth_record = next(r for r in records if r.source == GROUND_TRUTH_SOURCE)
    assert truth_record.evaluable is False


def test_bout_record_for_a_missed_bout_has_no_overlap() -> None:
    """Bout record for a missed bout has no overlap."""
    truth_record = _records([1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])[0]
    assert truth_record.overlapping_bouts == 0
    assert truth_record.best_overlap_frames == 0
    assert truth_record.best_iou == 0.0
    assert truth_record.matched_overlap is False


def test_bout_records_index_each_side_independently() -> None:
    """Best-overlap bookkeeping must not mix up truth index 1 with predicted index 1."""
    # truth bouts at 0-1 and 6-9; predictions at 6-9 and 20-21
    records = _records(
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1] + [0] * 12,
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1] + [0] * 10 + [1, 1],
    )
    truth_records = [r for r in records if r.source == GROUND_TRUTH_SOURCE]
    predicted_records = [r for r in records if r.source == PREDICTED_SOURCE]

    assert [r.matched_overlap for r in truth_records] == [False, True]
    assert [r.matched_overlap for r in predicted_records] == [True, False]
    assert truth_records[1].best_overlap_frames == 4


# -----------------------------------------------------------------------------
# classifier / behavior resolution
# -----------------------------------------------------------------------------


def test_load_binary_classifier_rejects_multiclass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load binary classifier rejects multiclass."""
    monkeypatch.setattr(
        evaluate_module,
        "load_classifier_from_pickle",
        lambda _p: mock.Mock(spec=MultiClassClassifier),
    )
    with pytest.raises(Exception, match="binary classifiers only"):
        load_binary_classifier(Path("model.pickle"))


def test_load_binary_classifier_wraps_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load binary classifier wraps load failure."""

    def boom(_p):
        raise RuntimeError("corrupt pickle")

    monkeypatch.setattr(evaluate_module, "load_classifier_from_pickle", boom)
    with pytest.raises(Exception, match="Unable to load classifier"):
        load_binary_classifier(Path("model.pickle"))


def test_resolve_behavior_prefers_the_explicit_name() -> None:
    """Resolve behavior prefers the explicit name."""
    classifier = mock.Mock(behavior_name="Grooming")
    assert resolve_behavior(classifier, "Rearing") == "Rearing"


def test_resolve_behavior_falls_back_to_the_classifier() -> None:
    """Resolve behavior falls back to the classifier."""
    classifier = mock.Mock(behavior_name="Grooming")
    assert resolve_behavior(classifier, None) == "Grooming"


def test_resolve_behavior_errors_when_neither_is_available() -> None:
    """Resolve behavior errors when neither is available."""
    classifier = mock.Mock(behavior_name=None)
    with pytest.raises(Exception, match="does not record a behavior name"):
        resolve_behavior(classifier, None)


# -----------------------------------------------------------------------------
# postprocessing config resolution
# -----------------------------------------------------------------------------


def test_resolve_pipeline_config_accepts_a_list() -> None:
    """Resolve pipeline config accepts a list."""
    config = [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 5}}]
    assert _resolve_pipeline_config(config, "Grooming") == config


def test_resolve_pipeline_config_selects_the_behavior_from_a_dict() -> None:
    """Resolve pipeline config selects the behavior from a dict."""
    stages = [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 5}}]
    assert _resolve_pipeline_config({"Grooming": stages, "Rearing": []}, "Grooming") == stages


def test_resolve_pipeline_config_errors_on_missing_behavior() -> None:
    """Resolve pipeline config errors on missing behavior."""
    with pytest.raises(Exception, match="not found in the postprocessing config"):
        _resolve_pipeline_config({"Rearing": []}, "Grooming")


def test_resolve_pipeline_config_rejects_a_scalar() -> None:
    """Resolve pipeline config rejects a scalar."""
    with pytest.raises(Exception, match="list or object at the top level"):
        _resolve_pipeline_config("nope", "Grooming")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# results aggregation
# -----------------------------------------------------------------------------


def _identity_result(video: str, identity: int, stage: str, truth, predicted) -> IdentityResult:
    """Build an IdentityResult from two vectors."""
    comparison = compare_identity(np.asarray(truth), np.asarray(predicted), CRITERIA)
    return IdentityResult(
        video=video,
        identity=identity,
        stage=stage,
        frame_metrics=comparison.frame_metrics,
        bout_metrics=comparison.bout_metrics,
    )


@pytest.fixture
def sample_result() -> EvaluationResult:
    """A two-video, two-stage evaluation result."""
    return EvaluationResult(
        project_dir=Path("/proj"),
        behavior="Grooming",
        classifier_path=Path("/proj/grooming.pickle"),
        classifier_type="XGBoost",
        window_size=5,
        stages=(RAW_STAGE, POSTPROCESSED_STAGE),
        criteria=(OVERLAP.label, IOU.label),
        identity_results=[
            _identity_result("a.mp4", 0, RAW_STAGE, [1, 1, 0, 0], [1, 1, 0, 0]),
            _identity_result("a.mp4", 1, RAW_STAGE, [1, 1, 0, 0], [0, 0, 0, 0]),
            _identity_result("b.mp4", 0, RAW_STAGE, [0, 1, 1, 0], [0, 1, 0, 0]),
            _identity_result("a.mp4", 0, POSTPROCESSED_STAGE, [1, 1, 0, 0], [1, 1, 0, 0]),
            _identity_result("a.mp4", 1, POSTPROCESSED_STAGE, [1, 1, 0, 0], [0, 0, 0, 0]),
            _identity_result("b.mp4", 0, POSTPROCESSED_STAGE, [0, 1, 1, 0], [0, 1, 1, 0]),
        ],
        skipped_videos=[("c.mp4", "no annotations in the project")],
        unlabeled_identities=[("b.mp4", 1)],
        postprocess_stages=("BoutStitchingStage",),
    )


def test_videos_are_listed_in_first_seen_order(sample_result: EvaluationResult) -> None:
    """Videos are listed in first seen order."""
    assert sample_result.videos == ["a.mp4", "b.mp4"]


def test_for_stage_and_for_video_filter(sample_result: EvaluationResult) -> None:
    """For stage and for video filter."""
    assert len(sample_result.for_stage(RAW_STAGE)) == 3
    assert len(sample_result.for_video(RAW_STAGE, "a.mp4")) == 2


def test_aggregate_frame_metrics_sums_across_identities(sample_result: EvaluationResult) -> None:
    """Aggregate frame metrics sums across identities."""
    metrics = aggregate_frame_metrics(sample_result.for_stage(RAW_STAGE))
    # 2 TP from a.mp4/0, 1 TP from b.mp4/0
    assert metrics.true_positive == 3
    assert metrics.false_negative == 3
    assert metrics.evaluated_frames == 12


def test_aggregate_bout_metrics_sums_across_identities(sample_result: EvaluationResult) -> None:
    """Aggregate bout metrics sums across identities."""
    metrics = aggregate_bout_metrics(sample_result.for_stage(RAW_STAGE), OVERLAP.label)
    assert metrics.truth_bouts == 3
    assert metrics.detected_truth_bouts == 2
    assert metrics.detection_rate == pytest.approx(2 / 3)


def test_aggregate_bout_metrics_ignores_an_unknown_criterion(
    sample_result: EvaluationResult,
) -> None:
    """A criterion no result carries aggregates to zero rather than raising."""
    metrics = aggregate_bout_metrics(sample_result.for_stage(RAW_STAGE), "IoU >= 0.99")
    assert metrics.truth_bouts == 0


def test_format_rate_renders_undefined_as_na() -> None:
    """Format rate renders undefined as na."""
    assert format_rate(None) == "n/a"
    assert format_rate(0.5) == "0.500"


# -----------------------------------------------------------------------------
# report rendering
# -----------------------------------------------------------------------------


def test_json_summary_is_serializable_and_nested(sample_result: EvaluationResult) -> None:
    """Json summary is serializable and nested."""
    summary = build_summary(sample_result, datetime(2026, 9, 16, 12, 0, 0))
    text = json.dumps(summary)  # would raise on a numpy scalar or NaN
    assert "NaN" not in text

    assert summary["behavior"] == "Grooming"
    assert set(summary["stages"]) == {RAW_STAGE, POSTPROCESSED_STAGE}
    raw = summary["stages"][RAW_STAGE]
    assert raw["overall"]["bouts"][OVERLAP.label]["detected_truth_bouts"] == 2
    assert set(raw["videos"]) == {"a.mp4", "b.mp4"}
    assert set(raw["videos"]["a.mp4"]["identities"]) == {"0", "1"}


def test_json_summary_uses_null_for_undefined_rates() -> None:
    """Json summary uses null for undefined rates."""
    result = EvaluationResult(
        project_dir=Path("/proj"),
        behavior="Grooming",
        classifier_path=Path("/m.pickle"),
        classifier_type="XGBoost",
        window_size=5,
        stages=(RAW_STAGE,),
        criteria=(OVERLAP.label,),
        identity_results=[_identity_result("a.mp4", 0, RAW_STAGE, [0, 0], [0, 0])],
    )
    summary = build_summary(result, datetime(2026, 9, 16))
    assert (
        summary["stages"][RAW_STAGE]["overall"]["bouts"][OVERLAP.label]["detection_rate"] is None
    )


def test_write_json_round_trips(sample_result: EvaluationResult, tmp_path: Path) -> None:
    """Write json round trips."""
    path = tmp_path / "summary.json"
    write_json(sample_result, path, datetime(2026, 9, 16))
    assert json.loads(path.read_text())["behavior"] == "Grooming"


def test_write_csv_emits_a_header_even_with_no_rows(tmp_path: Path) -> None:
    """Write csv emits a header even with no rows."""
    path = tmp_path / "bouts.csv"
    write_csv([], path)
    rows = list(csv.reader(path.open()))
    assert len(rows) == 1
    assert "best_iou" in rows[0]


def test_write_csv_writes_one_row_per_bout(tmp_path: Path) -> None:
    """Write csv writes one row per bout."""
    path = tmp_path / "bouts.csv"
    write_csv(_records([1, 1, 0, 0], [0, 1, 1, 0]), path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert {r["source"] for r in rows} == {GROUND_TRUTH_SOURCE, PREDICTED_SOURCE}
    assert rows[0]["video"] == "v.mp4"


def test_markdown_report_contains_the_key_sections(sample_result: EvaluationResult) -> None:
    """Markdown report contains the key sections."""
    text = render_markdown(sample_result, datetime(2026, 9, 16, 12, 0, 0))
    assert "# Classifier Evaluation: Grooming" in text
    assert "## Frame-level agreement" in text
    assert "## Bout-level agreement" in text
    assert f"### {OVERLAP.label}" in text
    assert f"### {IOU.label}" in text
    assert "## Per-video breakdown" in text
    assert "## Skipped videos" in text
    assert "c.mp4" in text
    assert "BoutStitchingStage" in text


def test_write_markdown_round_trips(sample_result: EvaluationResult, tmp_path: Path) -> None:
    """Write markdown round trips."""
    path = tmp_path / "report.md"
    write_markdown(sample_result, path, datetime(2026, 9, 16))
    assert path.read_text().startswith("# Classifier Evaluation: Grooming")


# -----------------------------------------------------------------------------
# Click wiring
# -----------------------------------------------------------------------------


@pytest.fixture
def wired(
    monkeypatch: pytest.MonkeyPatch, sample_result: EvaluationResult
) -> tuple[mock.Mock, EvaluationResult]:
    """Stub out classifier loading, project scanning, and the evaluation run."""
    classifier = mock.Mock(behavior_name="Grooming")
    monkeypatch.setattr(evaluate_module, "load_binary_classifier", lambda _p: classifier)

    project = mock.Mock()
    project.video_manager.videos = []
    monkeypatch.setattr(evaluate_module, "Project", lambda *a, **k: project)

    spy = mock.Mock(return_value=sample_result)
    monkeypatch.setattr(evaluate_module, "run_evaluation", spy)
    return spy, sample_result


def _invoke(tmp_path: Path, *extra: str):
    """Invoke the evaluate command against a temporary project directory."""
    classifier = tmp_path / "model.pickle"
    classifier.touch()
    return CliRunner().invoke(
        cli, ["evaluate", str(tmp_path), "--classifier", str(classifier), *extra]
    )


def test_command_runs_and_prints_tables(wired, tmp_path: Path) -> None:
    """Command runs and prints tables."""
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert "Frame-level agreement" in result.output
    assert "Bout-level agreement" in result.output


def test_command_passes_both_criteria_in_overlap_then_iou_order(wired, tmp_path: Path) -> None:
    """run_evaluation documents this order; the CSV match columns depend on it."""
    spy, _ = wired
    assert _invoke(tmp_path).exit_code == 0
    criteria = spy.call_args.kwargs["criteria"]
    assert [c.label for c in criteria] == ["overlap >= 1 frame", "IoU >= 0.5"]


def test_command_threshold_options_reach_the_criteria(wired, tmp_path: Path) -> None:
    """Command threshold options reach the criteria."""
    spy, _ = wired
    assert _invoke(tmp_path, "--min-overlap", "5", "--iou-threshold", "0.25").exit_code == 0
    criteria = spy.call_args.kwargs["criteria"]
    assert [c.label for c in criteria] == ["overlap >= 5 frames", "IoU >= 0.25"]


def test_command_rejects_an_out_of_range_iou(wired, tmp_path: Path) -> None:
    """Command rejects an out of range iou."""
    assert _invoke(tmp_path, "--iou-threshold", "0").exit_code != 0
    assert _invoke(tmp_path, "--iou-threshold", "1.5").exit_code != 0


def test_command_rejects_a_zero_min_overlap(wired, tmp_path: Path) -> None:
    """Command rejects a zero min overlap."""
    assert _invoke(tmp_path, "--min-overlap", "0").exit_code != 0


def test_out_dir_writes_all_three_files(wired, tmp_path: Path) -> None:
    """Out dir writes all three files."""
    out = tmp_path / "results"
    result = _invoke(tmp_path, "--out-dir", str(out))
    assert result.exit_code == 0, result.output

    written = sorted(p.name.rsplit("_", 1)[-1] for p in out.iterdir())
    assert written == ["bouts.csv", "evaluation.json", "evaluation.md"]
    assert all(p.name.startswith("Grooming_") for p in out.iterdir())


def test_out_dir_filenames_use_the_resolved_behavior(wired, tmp_path: Path) -> None:
    """The behavior comes from the classifier when --behavior is not given."""
    out = tmp_path / "results"
    assert _invoke(tmp_path, "--out-dir", str(out)).exit_code == 0
    assert all("Grooming" in p.name for p in out.iterdir())


def test_explicit_output_paths_are_honored(wired, tmp_path: Path) -> None:
    """Explicit output paths are honored."""
    json_path = tmp_path / "m.json"
    csv_path = tmp_path / "b.csv"
    report_path = tmp_path / "r.md"
    result = _invoke(
        tmp_path,
        "--json-out",
        str(json_path),
        "--csv-out",
        str(csv_path),
        "--report-out",
        str(report_path),
    )
    assert result.exit_code == 0, result.output
    assert json_path.exists() and csv_path.exists() and report_path.exists()


def test_no_output_options_writes_nothing(wired, tmp_path: Path) -> None:
    """No output options writes nothing."""
    result = _invoke(tmp_path)
    assert result.exit_code == 0
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.md"))


def test_bout_records_are_only_collected_when_a_csv_is_requested(wired, tmp_path: Path) -> None:
    """Building per-bout rows for a long project is not free; skip it when unused."""
    spy, _ = wired

    assert _invoke(tmp_path).exit_code == 0
    assert spy.call_args.kwargs["collect_bout_records"] is False

    assert _invoke(tmp_path, "--csv-out", str(tmp_path / "b.csv")).exit_code == 0
    assert spy.call_args.kwargs["collect_bout_records"] is True


def test_per_video_flag_adds_the_breakdown(wired, tmp_path: Path) -> None:
    """Per video flag adds the breakdown."""
    without = _invoke(tmp_path)
    with_flag = _invoke(tmp_path, "--per-video")
    assert "by video" not in without.output
    assert "by video" in with_flag.output


def test_behavior_option_overrides_the_classifier(wired, tmp_path: Path) -> None:
    """Behavior option overrides the classifier."""
    spy, _ = wired
    assert _invoke(tmp_path, "--behavior", "Rearing").exit_code == 0
    assert spy.call_args.kwargs["behavior"] == "Rearing"


# -----------------------------------------------------------------------------
# run_evaluation orchestration
#
# The prediction call and the project are stubbed, but the loop itself - label
# lookup, alignment, staging, postprocessing, record collection and skip
# handling - runs for real.
# -----------------------------------------------------------------------------


def _fake_project(
    monkeypatch: pytest.MonkeyPatch,
    videos: dict[str, dict],
    behavior: str = "Grooming",
) -> mock.Mock:
    """Install a stub Project whose videos yield canned labels and predictions.

    Args:
        monkeypatch: Patching fixture.
        videos: Maps video name to a dict with ``truth`` and ``predicted``, each
            mapping identity index to a per-frame vector.
        behavior: Behavior the stub project reports having labels for.

    Returns:
        The stub project instance.
    """
    project = mock.Mock()
    project.settings_manager.behavior_names = [behavior]
    project.video_manager.videos = list(videos)
    project.video_manager.get_video_identity_count.return_value = 1
    project.video_manager.get_cached_pose_path.side_effect = lambda v: Path(f"/{v}_pose.h5")
    project.video_manager.video_path.side_effect = lambda v: Path(f"/{v}")

    def load_pose_est(video_path):
        spec = videos[Path(video_path).name]
        pose = mock.Mock()
        pose.identities = list(spec["truth"])
        pose.num_identities = len(spec["truth"])
        pose.num_frames = len(next(iter(spec["predicted"].values())))
        return pose

    def load_video_labels(video, _pose):
        spec = videos[video]
        labels = mock.Mock()
        labels.get_track_labels.side_effect = lambda ident, _beh: mock.Mock(
            get_labels=lambda: np.asarray(spec["truth"][int(ident)])
        )
        return labels

    project.load_pose_est.side_effect = load_pose_est
    project.video_manager.load_video_labels.side_effect = load_video_labels

    def predict(_clf, pose_path, _pose, identity, *_a, **_k):
        video = Path(pose_path).name.removesuffix("_pose.h5")
        predicted = np.asarray(videos[video]["predicted"][identity], dtype=np.int8)
        return predicted, np.full(len(predicted), 0.9, dtype=np.float32)

    monkeypatch.setattr(evaluate_module, "Project", mock.Mock(return_value=project))
    monkeypatch.setattr(evaluate_module, "_predict_identity", predict)
    monkeypatch.setattr(evaluate_module, "get_fps_and_nframes", lambda _p: (30, 100))
    return project


def _run(**kwargs):
    """Invoke run_evaluation with the standard arguments."""
    defaults = {
        "project_dir": Path("/proj"),
        "classifier": mock.Mock(classifier_name="XGBoost", project_settings={"window_size": 5}),
        "classifier_path": Path("/m.pickle"),
        "behavior": "Grooming",
        "criteria": CRITERIA,
        "pipeline": None,
        "feature_dir": None,
        "fps_override": 30,
        "collect_bout_records": False,
    }
    return evaluate_module.run_evaluation(**{**defaults, **kwargs})


def test_run_evaluation_compares_every_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each identity of each video yields one result for the raw stage."""
    _fake_project(
        monkeypatch,
        {
            "a.mp4": {
                "truth": {0: [1, 1, 0, 0], 1: [0, 0, 1, 1]},
                "predicted": {0: [1, 1, 0, 0], 1: [0, 0, 1, 1]},
            },
            "b.mp4": {"truth": {0: [1, 1, 0, 0]}, "predicted": {0: [0, 0, 0, 0]}},
        },
    )
    result = _run()

    assert result.stages == (RAW_STAGE,)
    assert len(result.identity_results) == 3
    assert result.videos == ["a.mp4", "b.mp4"]
    metrics = aggregate_bout_metrics(result.for_stage(RAW_STAGE), OVERLAP.label)
    assert metrics.truth_bouts == 3
    assert metrics.detected_truth_bouts == 2


def test_run_evaluation_records_the_classifier_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report metadata comes from the passed-in classifier."""
    _fake_project(monkeypatch, {"a.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}}})
    result = _run()
    assert result.classifier_type == "XGBoost"
    assert result.window_size == 5
    assert result.behavior == "Grooming"


def test_run_evaluation_adds_a_postprocessed_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a pipeline, both stages are evaluated and the filter actually bites."""
    from jabs.behavior.postprocessing import PostprocessingPipeline

    # a 2-frame prediction blip against an all-not-behavior ground truth
    _fake_project(
        monkeypatch,
        {"a.mp4": {"truth": {0: [0] * 10}, "predicted": {0: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]}}},
    )
    pipeline = PostprocessingPipeline(
        [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 5}}]
    )
    result = _run(pipeline=pipeline)

    assert result.stages == (RAW_STAGE, POSTPROCESSED_STAGE)
    assert result.postprocess_stages == ("BoutDurationFilterStage",)

    raw = aggregate_bout_metrics(result.for_stage(RAW_STAGE), OVERLAP.label)
    post = aggregate_bout_metrics(result.for_stage(POSTPROCESSED_STAGE), OVERLAP.label)
    assert raw.predicted_bouts == 1  # the blip
    assert post.predicted_bouts == 0  # filtered away


def test_run_evaluation_skips_a_video_with_no_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A video without a label document is reported, not fatal."""
    project = _fake_project(
        monkeypatch,
        {
            "a.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}},
            "b.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}},
        },
    )
    real = project.video_manager.load_video_labels.side_effect
    project.video_manager.load_video_labels.side_effect = (
        lambda v, p: None if v == "b.mp4" else real(v, p)
    )

    result = _run()
    assert result.videos == ["a.mp4"]
    assert result.skipped_videos == [("b.mp4", "no annotations in the project")]


def test_run_evaluation_skips_an_identity_whose_prediction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad identity does not abort the run."""
    _fake_project(
        monkeypatch,
        {"a.mp4": {"truth": {0: [1, 0], 1: [1, 0]}, "predicted": {0: [1, 0], 1: [1, 0]}}},
    )

    def flaky(_clf, _path, _pose, identity, *a, **k):
        if identity == 1:
            raise RuntimeError("no features")
        return np.array([1, 0], dtype=np.int8), np.array([0.9, 0.9], dtype=np.float32)

    monkeypatch.setattr(evaluate_module, "_predict_identity", flaky)

    result = _run()
    assert len(result.identity_results) == 1
    assert result.skipped_videos == [("a.mp4 (identity 1)", "no features")]


def test_run_evaluation_notes_an_identity_with_no_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-unlabeled identity is recorded and contributes nothing."""
    _fake_project(
        monkeypatch,
        {"a.mp4": {"truth": {0: [-1, -1, -1]}, "predicted": {0: [1, 1, 0]}}},
    )
    result = _run()
    assert result.unlabeled_identities == [("a.mp4", 0)]
    assert aggregate_frame_metrics(result.for_stage(RAW_STAGE)).evaluated_frames == 0


def test_run_evaluation_truncates_to_the_shorter_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    """A label/pose frame-count disagreement is survivable."""
    _fake_project(
        monkeypatch,
        {"a.mp4": {"truth": {0: [1, 1, 0, 0, 0, 0]}, "predicted": {0: [1, 1, 0, 0]}}},
    )
    result = _run()
    assert aggregate_frame_metrics(result.for_stage(RAW_STAGE)).evaluated_frames == 4


def test_run_evaluation_collects_bout_records_only_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-bout rows are built only when asked for."""
    spec = {"a.mp4": {"truth": {0: [1, 1, 0, 0]}, "predicted": {0: [1, 1, 0, 0]}}}

    _fake_project(monkeypatch, spec)
    assert _run(collect_bout_records=False).bout_records == []

    _fake_project(monkeypatch, spec)
    records = _run(collect_bout_records=True).bout_records
    assert len(records) == 2
    assert {r.source for r in records} == {GROUND_TRUTH_SOURCE, PREDICTED_SOURCE}


def test_run_evaluation_advances_progress_past_a_skipped_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bar must not stall on a video that is never opened."""
    project = _fake_project(
        monkeypatch,
        {
            "a.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}},
            "b.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}},
        },
    )
    real = project.video_manager.load_video_labels.side_effect
    project.video_manager.load_video_labels.side_effect = (
        lambda v, p: None if v == "b.mp4" else real(v, p)
    )

    advanced: list[int] = []
    _run(progress_callback=lambda advance=1: advanced.append(advance))
    assert sum(advanced) == 2  # one real identity plus the skipped video's one


def test_run_evaluation_requires_two_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-bout CSV columns depend on criteria[0] and criteria[1]."""
    _fake_project(monkeypatch, {"a.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}}})
    with pytest.raises(ValueError, match="at least two criteria"):
        _run(criteria=[OVERLAP])


def test_run_evaluation_errors_when_nothing_was_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty project is an error, not a report of zeros."""
    _fake_project(monkeypatch, {})
    with pytest.raises(Exception, match="Nothing was evaluated"):
        _run()


# -----------------------------------------------------------------------------
# --save-predictions
# -----------------------------------------------------------------------------


def test_save_predictions_writes_one_file_per_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One prediction record per video, named from the pose file stem."""
    _fake_project(
        monkeypatch,
        {
            "a.mp4": {"truth": {0: [1, 1, 0, 0]}, "predicted": {0: [1, 1, 0, 0]}},
            "b.mp4": {"truth": {0: [1, 0, 0, 0]}, "predicted": {0: [0, 0, 0, 0]}},
        },
    )
    written = {}
    monkeypatch.setattr(
        evaluate_module.PredictionManager,
        "write_predictions",
        staticmethod(
            lambda beh, path, pred, prob, *a, **k: written.__setitem__(path, (beh, pred))
        ),
    )

    result = _run(save_predictions_dir=tmp_path)

    assert sorted(p.name for p in result.prediction_files) == [
        "a.mp4_pose_behavior.h5",
        "b.mp4_pose_behavior.h5",
    ]
    assert all(beh == "Grooming" for beh, _ in written.values())


def test_save_predictions_accumulates_every_identity_into_one_array(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The saved array is (n_identities, n_frames), not one row per call."""
    _fake_project(
        monkeypatch,
        {
            "a.mp4": {
                "truth": {0: [1, 1, 0, 0], 1: [0, 0, 1, 1]},
                "predicted": {0: [1, 1, 0, 0], 1: [0, 0, 1, 1]},
            }
        },
    )
    captured = {}
    monkeypatch.setattr(
        evaluate_module.PredictionManager,
        "write_predictions",
        staticmethod(
            lambda beh, path, pred, prob, *a, **k: captured.update(predictions=pred, probs=prob)
        ),
    )

    _run(save_predictions_dir=tmp_path)

    assert captured["predictions"].shape == (2, 4)
    np.testing.assert_array_equal(captured["predictions"][0], [1, 1, 0, 0])
    np.testing.assert_array_equal(captured["predictions"][1], [0, 0, 1, 1])
    assert captured["probs"].shape == (2, 4)


def test_save_predictions_leaves_a_failed_identity_unscored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An identity that could not be classified stays NONE rather than 0."""
    _fake_project(
        monkeypatch,
        {"a.mp4": {"truth": {0: [1, 1], 1: [1, 1]}, "predicted": {0: [1, 1], 1: [1, 1]}}},
    )

    def flaky(_clf, _path, _pose, identity, *a, **k):
        if identity == 1:
            raise RuntimeError("no features")
        return np.array([1, 1], dtype=np.int8), np.array([0.9, 0.9], dtype=np.float32)

    monkeypatch.setattr(evaluate_module, "_predict_identity", flaky)
    captured = {}
    monkeypatch.setattr(
        evaluate_module.PredictionManager,
        "write_predictions",
        staticmethod(lambda beh, path, pred, *a, **k: captured.update(predictions=pred)),
    )

    _run(save_predictions_dir=tmp_path)

    np.testing.assert_array_equal(captured["predictions"][0], [1, 1])
    np.testing.assert_array_equal(captured["predictions"][1], [-1, -1])


def test_save_predictions_includes_postprocessed_when_a_pipeline_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A saved file carries both raw and postprocessed, like jabs-cli postprocess writes."""
    from jabs.behavior.postprocessing import PostprocessingPipeline

    _fake_project(
        monkeypatch,
        {"a.mp4": {"truth": {0: [0] * 10}, "predicted": {0: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]}}},
    )
    captured = {}
    monkeypatch.setattr(
        evaluate_module.PredictionManager,
        "write_predictions",
        staticmethod(
            lambda beh, path, pred, prob, poses, clf, postprocessed_predictions=None, **k: (
                captured.update(raw=pred, post=postprocessed_predictions)
            )
        ),
    )
    pipeline = PostprocessingPipeline(
        [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 5}}]
    )

    _run(pipeline=pipeline, save_predictions_dir=tmp_path)

    assert captured["post"] is not None
    assert captured["raw"][0].tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert captured["post"][0].tolist() == [0] * 10  # the blip was filtered out


def test_save_predictions_omits_postprocessed_without_a_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No pipeline means no postprocessed dataset in the saved file."""
    _fake_project(monkeypatch, {"a.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}}})
    captured = {}
    monkeypatch.setattr(
        evaluate_module.PredictionManager,
        "write_predictions",
        staticmethod(
            lambda beh, path, pred, prob, poses, clf, postprocessed_predictions=None, **k: (
                captured.update(post=postprocessed_predictions)
            )
        ),
    )

    _run(save_predictions_dir=tmp_path)
    assert captured["post"] is None


def test_save_predictions_saves_full_length_not_truncated_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A label/pose length mismatch truncates the comparison, not the saved file."""
    _fake_project(
        monkeypatch,
        {"a.mp4": {"truth": {0: [1, 1]}, "predicted": {0: [1, 1, 0, 0, 0, 0]}}},
    )
    captured = {}
    monkeypatch.setattr(
        evaluate_module.PredictionManager,
        "write_predictions",
        staticmethod(lambda beh, path, pred, *a, **k: captured.update(predictions=pred)),
    )

    result = _run(save_predictions_dir=tmp_path)

    assert captured["predictions"].shape == (1, 6)  # the pose length
    assert aggregate_frame_metrics(result.for_stage(RAW_STAGE)).evaluated_frames == 2


def test_save_predictions_skips_a_video_with_no_successful_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An all-NONE file would misrepresent a video nothing could be classified in."""
    _fake_project(monkeypatch, {"a.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}}})

    def always_fails(*a, **k):
        raise RuntimeError("no features")

    monkeypatch.setattr(evaluate_module, "_predict_identity", always_fails)
    monkeypatch.setattr(
        evaluate_module.PredictionManager,
        "write_predictions",
        staticmethod(lambda *a, **k: pytest.fail("should not write an empty prediction file")),
    )

    with pytest.raises(Exception, match="Nothing was evaluated"):
        _run(save_predictions_dir=tmp_path)


def test_save_predictions_records_a_write_failure_without_losing_the_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full disk must not cost the run its results."""
    _fake_project(
        monkeypatch, {"a.mp4": {"truth": {0: [1, 1, 0, 0]}, "predicted": {0: [1, 1, 0, 0]}}}
    )

    def boom(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(evaluate_module.PredictionManager, "write_predictions", staticmethod(boom))

    result = _run(save_predictions_dir=tmp_path)

    assert result.prediction_files == []
    assert result.prediction_write_errors == [("a.mp4", "No space left on device")]
    assert aggregate_frame_metrics(result.for_stage(RAW_STAGE)).true_positive == 2


def test_no_prediction_files_written_when_not_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saving is opt-in."""
    _fake_project(monkeypatch, {"a.mp4": {"truth": {0: [1, 0]}, "predicted": {0: [1, 0]}}})
    monkeypatch.setattr(
        evaluate_module.PredictionManager,
        "write_predictions",
        staticmethod(lambda *a, **k: pytest.fail("should not write anything")),
    )
    assert _run().prediction_files == []


def test_command_creates_the_prediction_directory(wired, tmp_path: Path) -> None:
    """The directory is created up front so a bad path fails before the slow work."""
    spy, _ = wired
    out = tmp_path / "preds" / "nested"
    assert _invoke(tmp_path, "--save-predictions", str(out)).exit_code == 0
    assert out.is_dir()
    assert spy.call_args.kwargs["save_predictions_dir"] == out


def test_command_does_not_pass_a_directory_when_not_requested(wired, tmp_path: Path) -> None:
    """Without the flag, run_evaluation is told not to save."""
    spy, _ = wired
    assert _invoke(tmp_path).exit_code == 0
    assert spy.call_args.kwargs["save_predictions_dir"] is None


def test_report_mentions_written_prediction_files(sample_result: EvaluationResult) -> None:
    """Saved files and write failures surface in the JSON and markdown outputs."""
    result = EvaluationResult(
        **{
            **sample_result.__dict__,
            "prediction_files": [Path("/out/a.mp4_pose_behavior.h5")],
            "prediction_write_errors": [("b.mp4", "No space left on device")],
        }
    )
    summary = build_summary(result, datetime(2026, 9, 16))
    assert summary["prediction_files"] == ["/out/a.mp4_pose_behavior.h5"]
    assert summary["prediction_write_errors"] == [
        {"video": "b.mp4", "reason": "No space left on device"}
    ]

    text = render_markdown(result, datetime(2026, 9, 16))
    assert "## Saved predictions" in text
    assert "## Prediction files that could not be written" in text
    assert "No space left on device" in text


def test_accumulated_arrays_round_trip_through_a_real_prediction_file(tmp_path: Path) -> None:
    """The arrays the accumulator builds must satisfy the real HDF5 writer.

    Every other --save-predictions test stubs out write_predictions, so this one
    exercises the actual dtype and shape contract: an all-NONE row for a failed
    identity, and a postprocessed array alongside the raw one.
    """
    from jabs import io
    from jabs.core.types.prediction import BehaviorPrediction, ClassifierMetadata

    accumulator = evaluate_module._PredictionAccumulator(2, 6, with_postprocessed=True)
    accumulator.add(
        0,
        np.array([1, 1, 0, 0, 1, 1], dtype=np.int8),
        np.full(6, 0.8, dtype=np.float32),
        np.array([1, 1, 1, 1, 1, 1], dtype=np.int8),
    )
    # identity 1 never classified, so its rows stay NONE

    path = tmp_path / "video_pose_est_v6_behavior.h5"
    io.save(
        BehaviorPrediction(
            behavior="Grooming",
            predicted_class=accumulator.predictions,
            probabilities=accumulator.probabilities,
            classifier=ClassifierMetadata("m.pickle", "abc", "0.0.0", "2026-09-16"),
            pose_file="video_pose_est_v6.h5",
            pose_hash="deadbeef",
            predicted_class_postprocessed=accumulator.postprocessed,
        ),
        path,
    )

    loaded = io.load(path, BehaviorPrediction, behavior="Grooming")
    assert loaded.predicted_class.shape == (2, 6)
    np.testing.assert_array_equal(loaded.predicted_class[0], [1, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(loaded.predicted_class[1], [-1] * 6)
    np.testing.assert_array_equal(loaded.predicted_class_postprocessed[0], [1] * 6)


def test_accumulator_reports_whether_anything_was_added() -> None:
    """has_predictions gates writing a file at all."""
    accumulator = evaluate_module._PredictionAccumulator(1, 3, with_postprocessed=False)
    assert accumulator.has_predictions is False
    assert accumulator.postprocessed is None

    accumulator.add(0, np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.float32), None)
    assert accumulator.has_predictions is True
