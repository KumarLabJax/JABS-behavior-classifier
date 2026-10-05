"""Tests for training report generation."""

from datetime import datetime

import numpy as np
import pytest

from jabs.classifier.training_report import (
    BinaryCVResult,
    MultiClassCVResult,
    PostprocessedMetrics,
    TrainingReportData,
    generate_json_report,
    generate_markdown_report,
    save_training_report,
)
from jabs.core.enums import CrossValidationGroupingStrategy


@pytest.fixture
def sample_cv_results():
    """Create sample cross-validation results.

    Returns:
        List of CrossValidationResult objects.
    """
    return [
        BinaryCVResult(
            iteration=1,
            test_label="video_1.mp4 [0]",
            accuracy=0.9234,
            precision_not_behavior=0.9145,
            precision_behavior=0.9323,
            recall_not_behavior=0.9456,
            recall_behavior=0.9012,
            f1_behavior=0.9163,
            support_behavior=150,
            support_not_behavior=200,
            confusion_matrix=np.array([[180, 20], [15, 135]]),
            top_features=[("nose_speed", 0.16), ("ear_angle", 0.14)],
        ),
        BinaryCVResult(
            iteration=2,
            test_label="video_2.mp4 [1]",
            accuracy=0.8912,
            precision_not_behavior=0.8823,
            precision_behavior=0.9001,
            recall_not_behavior=0.9134,
            recall_behavior=0.8690,
            f1_behavior=0.8842,
            support_behavior=140,
            support_not_behavior=210,
            confusion_matrix=np.array([[192, 18], [18, 122]]),
            top_features=[("nose_speed", 0.15), ("ear_angle", 0.13)],
        ),
    ]


@pytest.fixture
def sample_training_data(sample_cv_results):
    """Create sample training report data.

    Returns:
        TrainingReportData object with sample data.
    """
    return TrainingReportData(
        behavior_name="Grooming",
        classifier_type="Random Forest",
        window_size=5,
        balance_training_labels=True,
        symmetric_behavior=False,
        distance_unit="cm",
        cv_results=sample_cv_results,
        final_top_features=[
            ("nose_speed", 0.156),
            ("left_ear_angle", 0.143),
            ("right_ear_angle", 0.129),
            ("body_length", 0.098),
            ("centroid_speed", 0.087),
        ],
        frames_behavior=1250,
        frames_not_behavior=3840,
        bouts_behavior=42,
        bouts_not_behavior=156,
        training_time_ms=12345,
        timestamp=datetime(2026, 1, 3, 14, 30, 45),
        cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
    )


class TestCrossValidationResult:
    """Tests for BinaryCVResult dataclass."""

    def test_create_cv_result(self):
        """Test creating a BinaryCVResult instance."""
        result = BinaryCVResult(
            iteration=1,
            test_label="test.mp4 [0]",
            accuracy=0.95,
            precision_not_behavior=0.94,
            precision_behavior=0.96,
            recall_not_behavior=0.97,
            recall_behavior=0.93,
            f1_behavior=0.945,
            support_behavior=100,
            support_not_behavior=150,
            confusion_matrix=np.array([[140, 10], [7, 93]]),
            top_features=[("feature1", 0.5), ("feature2", 0.3)],
        )

        assert result.iteration == 1
        assert result.test_label == "test.mp4 [0]"
        assert result.accuracy == 0.95
        assert result.precision_not_behavior == 0.94
        assert result.precision_behavior == 0.96
        assert result.recall_not_behavior == 0.97
        assert result.recall_behavior == 0.93
        assert result.f1_behavior == 0.945
        assert result.support_behavior == 100
        assert result.support_not_behavior == 150
        assert result.confusion_matrix.shape == (2, 2)
        assert result.top_features == [("feature1", 0.5), ("feature2", 0.3)]


class TestTrainingReportData:
    """Tests for TrainingReportData dataclass."""

    def test_create_training_data(self, sample_cv_results):
        """Test creating a TrainingReportData instance."""
        timestamp = datetime.now()
        data = TrainingReportData(
            behavior_name="Rearing",
            classifier_type="XGBoost",
            window_size=7,
            balance_training_labels=False,
            symmetric_behavior=True,
            distance_unit="pixel",
            cv_results=sample_cv_results,
            final_top_features=[("feature1", 0.5), ("feature2", 0.3)],
            frames_behavior=500,
            frames_not_behavior=1500,
            bouts_behavior=20,
            bouts_not_behavior=80,
            training_time_ms=5000,
            timestamp=timestamp,
            cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO,
        )

        assert data.behavior_name == "Rearing"
        assert data.classifier_type == "XGBoost"
        assert data.window_size == 7
        assert data.balance_training_labels is False
        assert data.symmetric_behavior is True
        assert data.distance_unit == "pixel"
        assert len(data.cv_results) == 2
        assert len(data.final_top_features) == 2
        assert data.frames_behavior == 500
        assert data.frames_not_behavior == 1500
        assert data.bouts_behavior == 20
        assert data.bouts_not_behavior == 80
        assert data.training_time_ms == 5000
        assert data.timestamp == timestamp
        assert data.cv_grouping_strategy == CrossValidationGroupingStrategy.VIDEO


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report function."""

    def test_report_contains_header(self, sample_training_data):
        """Test that report contains behavior name in header."""
        report = generate_markdown_report(sample_training_data)

        assert "# Training Report: Grooming" in report

    def test_report_contains_timestamp(self, sample_training_data):
        """Test that report contains formatted timestamp."""
        report = generate_markdown_report(sample_training_data)

        assert "**Date:**" in report
        assert "January 03, 2026" in report
        assert "02:30:45 PM" in report

    def test_report_contains_training_summary(self, sample_training_data):
        """Test that report contains training summary section."""
        report = generate_markdown_report(sample_training_data)

        assert "## Training Summary" in report
        assert "**Behavior:** Grooming" in report
        assert "**Classifier:** Random Forest" in report
        assert "**Balanced Training Labels:** Yes" in report
        assert "**Symmetric Behavior:** No" in report
        assert "**Distance Unit:** cm" in report
        assert "**Training Time:** 12.35 seconds" in report

    def test_report_contains_label_counts(self, sample_training_data):
        """Test that report contains label count information."""
        report = generate_markdown_report(sample_training_data)

        assert "### Label Counts" in report
        assert "**Behavior frames:** 1,250" in report
        assert "**Not-behavior frames:** 3,840" in report
        assert "**Behavior bouts:** 42" in report
        assert "**Not-behavior bouts:** 156" in report

    def test_report_contains_cv_results(self, sample_training_data):
        """Test that report contains cross-validation results."""
        report = generate_markdown_report(sample_training_data)

        assert "## Cross-Validation Results" in report
        assert "### Performance Summary" in report
        assert "**Mean Accuracy:**" in report
        assert "**Mean F1 Score (Behavior):**" in report
        assert "### Iteration Details" in report

    def test_report_contains_cv_table(self, sample_training_data):
        """Test that CV results table is included."""
        report = generate_markdown_report(sample_training_data)

        # Check for table headers
        assert "Iter" in report
        assert "Accuracy" in report
        assert "Precision (Not Behavior)" in report
        assert "Precision (Behavior)" in report
        assert "Recall (Not Behavior)" in report
        assert "Recall (Behavior)" in report
        assert "F1 Score" in report
        assert "Test Group" in report

        assert "video\\_1.mp4 \\[0\\]" in report
        assert "video\\_2.mp4 \\[1\\]" in report
        assert "0.9234" in report  # accuracy from iteration 1

    def test_report_contains_feature_importance(self, sample_training_data):
        """Test that feature importance section is included."""
        report = generate_markdown_report(sample_training_data)

        assert "## Feature Importance" in report
        assert "Top 20 features from final model" in report
        # Note: underscores in feature names are escaped in markdown
        assert "nose\\_speed" in report
        assert "left\\_ear\\_angle" in report
        assert "0.16" in report  # importance value

    def test_report_without_cv_results(self, sample_training_data):
        """Test report generation when no cross-validation was performed."""
        # Create data with empty CV results
        data_no_cv = TrainingReportData(
            behavior_name="Grooming",
            classifier_type="Random Forest",
            window_size=5,
            balance_training_labels=True,
            symmetric_behavior=False,
            distance_unit="cm",
            cv_results=[],  # Empty CV results
            final_top_features=[("feature1", 0.5)],
            frames_behavior=100,
            frames_not_behavior=200,
            bouts_behavior=10,
            bouts_not_behavior=20,
            training_time_ms=1000,
            timestamp=datetime.now(),
            cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
        )

        report = generate_markdown_report(data_no_cv)

        assert "## Cross-Validation" in report
        assert "*No cross-validation was performed for this training.*" in report
        # Should not contain CV performance summary
        assert "### Performance Summary" not in report
        assert "### Iteration Details" not in report

    def test_report_without_cv_results_states_why_when_known(self, sample_training_data):
        """A CV warning replaces the neutral note, so the report says why metrics are missing."""
        data_no_cv = TrainingReportData(
            behavior_name="Grooming",
            classifier_type="Random Forest",
            window_size=5,
            balance_training_labels=True,
            symmetric_behavior=False,
            distance_unit="cm",
            cv_results=[],
            final_top_features=[("feature1", 0.5)],
            training_time_ms=1000,
            timestamp=datetime.now(),
            cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO,
            cv_warning="No cross-validation group could serve as a test split.",
        )

        report = generate_markdown_report(data_no_cv)

        assert "## Cross-Validation" in report
        assert "> **Warning:** No cross-validation group could serve as a test split." in report
        assert "*No cross-validation was performed for this training.*" not in report

    def test_markdown_escaping_in_video_names(self, sample_training_data):
        """Test that special characters in video names are escaped."""
        sample_training_data.cv_results[0].test_label = "test_video_with_underscores.mp4 [0]"
        report = generate_markdown_report(sample_training_data)
        # Tabulate does not preserve markdown escapes, so check for escaped string
        assert "test\\_video\\_with\\_underscores.mp4 \\[0\\]" in report


class TestSaveTrainingReport:
    """Tests for save_training_report function."""

    def test_save_report_creates_file(self, sample_training_data, tmp_path):
        """Test that saving a report creates a file."""
        output_file = tmp_path / "test_report.md"

        save_training_report(sample_training_data, output_file)

        assert output_file.exists()

    def test_saved_report_content(self, sample_training_data, tmp_path):
        """Test that saved report contains expected content."""
        output_file = tmp_path / "test_report.md"

        save_training_report(sample_training_data, output_file)

        content = output_file.read_text(encoding="utf-8")
        assert "# Training Report: Grooming" in content
        assert "## Training Summary" in content
        assert "## Cross-Validation Results" in content
        assert "## Feature Importance" in content

    def test_save_report_utf8_encoding(self, sample_training_data, tmp_path):
        """Test that report is saved with UTF-8 encoding."""
        output_file = tmp_path / "test_report.md"

        save_training_report(sample_training_data, output_file)

        # Should be able to read with UTF-8
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_save_report_overwrites_existing(self, sample_training_data, tmp_path):
        """Test that saving overwrites an existing file."""
        output_file = tmp_path / "test_report.md"

        # Write some initial content
        output_file.write_text("Old content")

        # Save the report
        save_training_report(sample_training_data, output_file)

        # New content should overwrite old
        content = output_file.read_text(encoding="utf-8")
        assert "Old content" not in content
        assert "# Training Report: Grooming" in content


class TestReportFormatting:
    """Tests for report formatting details."""

    def test_numbers_formatted_correctly(self, sample_training_data):
        """Test that numbers are formatted with proper precision."""
        report = generate_markdown_report(sample_training_data)

        # Accuracies should be 4 decimal places
        assert "0.9234" in report
        assert "0.8912" in report

        # Feature importance should be 2 decimal places
        assert "0.16" in report  # nose_speed importance

    def test_comma_separated_counts(self, sample_training_data):
        """Test that large numbers use comma separators."""
        report = generate_markdown_report(sample_training_data)

        assert "1,250" in report  # behavior frames
        assert "3,840" in report  # not-behavior frames

    def test_training_time_in_seconds(self, sample_training_data):
        """Test that training time is converted from ms to seconds."""
        report = generate_markdown_report(sample_training_data)

        # 12345 ms = 12.35 seconds
        assert "12.35 seconds" in report


class TestMulticlassReport:
    """Tests for multiclass CV/report rendering and JSON serialization."""

    def test_multiclass_markdown_contains_multiclass_metrics(self):
        """Markdown report uses multiclass summary/table and class counts."""
        cv_results = [
            MultiClassCVResult(
                iteration=1,
                test_label="video_a.mp4 [0]",
                accuracy=0.82,
                confusion_matrix=np.array([[10, 2, 1], [1, 9, 1], [0, 2, 8]]),
                class_names=["None", "Walk", "Run"],
                class_support=[13, 11, 10],
                precision_macro=0.83,
                recall_macro=0.82,
                f1_macro=0.81,
                precision_micro=0.82,
                recall_micro=0.82,
                f1_micro=0.82,
            )
        ]
        training_data = TrainingReportData(
            behavior_name="Walk",
            classifier_type="catboost",
            window_size=5,
            balance_training_labels=False,
            symmetric_behavior=False,
            distance_unit="pixel",
            cv_results=cv_results,
            final_top_features=[("feat_a", 0.5)],
            training_time_ms=1000,
            timestamp=datetime(2026, 4, 30, 12, 0, 0),
            cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
            class_frame_counts={"None": 100, "Walk": 80, "Run": 60},
            class_bout_counts={"None": 7, "Walk": 5, "Run": 4},
        )

        report = generate_markdown_report(training_data)
        assert "Mean F1 Score (Macro)" in report
        assert "Mean F1 Score (Micro)" in report
        assert "Precision (Macro)" in report
        assert "F1 Score (Micro)" in report
        assert "**None frames:** 100" in report
        assert "**Walk bouts:** 5" in report

    def test_multiclass_json_contains_optional_metrics(self):
        """JSON report serializes multiclass-only CV and class-count fields."""
        cv_results = [
            MultiClassCVResult(
                iteration=1,
                test_label="video_a.mp4",
                accuracy=0.9,
                confusion_matrix=np.array([[5, 1], [1, 6]]),
                class_names=["None", "Walk"],
                class_support=[6, 7],
                precision_macro=0.9,
                recall_macro=0.9,
                f1_macro=0.9,
                precision_micro=0.9,
                recall_micro=0.9,
                f1_micro=0.9,
                per_class_metrics=[
                    {
                        "class_name": "None",
                        "precision": 0.83,
                        "recall": 0.83,
                        "f1": 0.83,
                        "support": 6,
                    },
                    {
                        "class_name": "Walk",
                        "precision": 0.92,
                        "recall": 0.92,
                        "f1": 0.92,
                        "support": 7,
                    },
                ],
            )
        ]
        training_data = TrainingReportData(
            behavior_name="Walk",
            classifier_type="catboost",
            window_size=5,
            balance_training_labels=False,
            symmetric_behavior=False,
            distance_unit="pixel",
            cv_results=cv_results,
            final_top_features=[("feat_a", 0.5)],
            training_time_ms=1000,
            timestamp=datetime(2026, 4, 30, 12, 0, 0),
            cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
            class_frame_counts={"None": 10, "Walk": 20},
            class_bout_counts={"None": 1, "Walk": 2},
        )

        report = generate_json_report(training_data)
        assert report["class_frame_counts"] == {"None": 10, "Walk": 20}
        assert report["class_bout_counts"] == {"None": 1, "Walk": 2}
        assert report["cv_results"][0]["precision_macro"] == pytest.approx(0.9)
        assert report["cv_results"][0]["class_names"] == ["None", "Walk"]
        assert report["cv_results"][0]["class_support"] == [6, 7]

    def test_multiclass_markdown_empty_count_dicts_do_not_fall_back_to_binary(self):
        """Empty multiclass count dicts should not render binary count labels."""
        data = TrainingReportData(
            behavior_name="Walk",
            classifier_type="catboost",
            window_size=5,
            balance_training_labels=False,
            symmetric_behavior=False,
            distance_unit="pixel",
            cv_results=[],
            final_top_features=[("feat_a", 0.5)],
            training_time_ms=1000,
            timestamp=datetime(2026, 4, 30, 12, 0, 0),
            cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
            class_frame_counts={},
            class_bout_counts={},
        )

        report = generate_markdown_report(data)
        assert "**Behavior frames:**" not in report
        assert "**Not-behavior frames:**" not in report


class TestPostprocessedReporting:
    """Tests for reporting cross-validation metrics with postprocessing applied."""

    @staticmethod
    def _postprocessed_data(sample_training_data, stages: list[dict] | None = None):
        """Attach postprocessed metrics to every CV iteration of a report."""
        for offset, result in enumerate(sample_training_data.cv_results):
            result.postprocessed = PostprocessedMetrics(
                accuracy=0.95 + offset * 0.01,
                confusion_matrix=np.array([[190, 10], [8, 142]]),
                precision_not_behavior=0.9601,
                precision_behavior=0.9702,
                recall_not_behavior=0.9803,
                recall_behavior=0.9504,
                f1_behavior=0.9605,
            )
        sample_training_data.postprocessing_stages = (
            stages
            if stages is not None
            else [
                {
                    "stage_name": "BoutStitchingStage",
                    "enabled": True,
                    "parameters": {"max_stitch_gap": 3},
                }
            ]
        )
        return sample_training_data

    def test_markdown_contains_postprocessed_table(self, sample_training_data):
        """A postprocessed iteration table appears alongside the raw one."""
        data = self._postprocessed_data(sample_training_data)

        report = generate_markdown_report(data)

        assert "### Iteration Details" in report
        assert "### Iteration Details (Postprocessed)" in report
        # the postprocessed table carries its own metrics, distinct from the raw ones
        assert "0.9605" in report
        assert "0.9803" in report
        # raw metrics survive alongside them
        assert "0.9163" in report

    def test_markdown_contains_postprocessed_summary(self, sample_training_data):
        """The performance summary reports postprocessed means next to raw means."""
        data = self._postprocessed_data(sample_training_data)

        report = generate_markdown_report(data)

        assert "Mean Accuracy (Postprocessed):" in report
        assert "Mean F1 Score (Behavior, Postprocessed):" in report
        # raw means are still present, so the two can be compared
        assert "**Mean Accuracy:**" in report

    def test_markdown_lists_evaluated_stages(self, sample_training_data):
        """The summary records which stages were evaluated, so the report is self-describing."""
        data = self._postprocessed_data(sample_training_data)

        report = generate_markdown_report(data)

        assert "**Postprocessing Evaluated in Cross-Validation:** Yes" in report
        assert "BoutStitchingStage" in report
        assert "max\\_stitch\\_gap=3" in report  # underscores escaped for markdown

    def test_markdown_notes_when_no_stages_enabled(self, sample_training_data):
        """Requesting evaluation with no enabled stages is stated rather than silent."""
        sample_training_data.postprocessing_stages = []

        report = generate_markdown_report(sample_training_data)

        assert "**Postprocessing Evaluated in Cross-Validation:** No" in report
        assert "no stages are enabled" in report

    def test_markdown_does_not_claim_evaluation_without_metrics(self, sample_training_data):
        """Requesting evaluation that produced nothing must not report "Yes".

        The stage list is set from the saved configuration as soon as the
        evaluation is requested, so a run that found no valid cross-validation
        splits used to print "Yes" directly above a section saying no
        cross-validation was performed.
        """
        sample_training_data.postprocessing_stages = [
            {"stage_name": "BoutStitchingStage", "enabled": True, "parameters": {}}
        ]
        sample_training_data.cv_results = []

        report = generate_markdown_report(sample_training_data)

        assert "**Postprocessing Evaluated in Cross-Validation:** No" in report
        assert "produced no postprocessed metrics" in report
        # and the stages are not listed under a heading claiming they ran
        assert "BoutStitchingStage" not in report

    def test_markdown_flags_a_partial_set_of_evaluated_iterations(self, sample_training_data):
        """Postprocessed means over only some iterations are not passed off as comparable."""
        data = self._postprocessed_data(sample_training_data)
        data.cv_results[0].postprocessed = None
        evaluated = data.cv_results[1:]
        raw_accuracy = np.mean([r.accuracy for r in evaluated])

        summary = generate_markdown_report(data)

        assert f"cover {len(evaluated)} of {len(data.cv_results)} iterations" in summary
        assert f"not evaluated: {data.cv_results[0].iteration}" in summary
        # raw means over the same iterations are given for a like-for-like comparison
        assert f"accuracy {raw_accuracy:.4f}" in summary

    def test_markdown_has_no_partial_warning_when_every_iteration_was_evaluated(
        self, sample_training_data
    ):
        """A fully evaluated run carries no coverage warning."""
        data = self._postprocessed_data(sample_training_data)

        summary = generate_markdown_report(data)

        assert "iterations (not evaluated" not in summary

    def test_markdown_does_not_claim_evaluation_when_every_fold_skipped(
        self, sample_training_data
    ):
        """Folds that each returned no postprocessed metrics do not count as evaluated."""
        sample_training_data.postprocessing_stages = [
            {"stage_name": "BoutStitchingStage", "enabled": True, "parameters": {}}
        ]
        for result in sample_training_data.cv_results:
            result.postprocessed = None

        report = generate_markdown_report(sample_training_data)

        assert "**Postprocessing Evaluated in Cross-Validation:** No" in report

    def test_json_records_whether_postprocessing_was_evaluated(self, sample_training_data):
        """The JSON separates what was requested from what actually happened."""
        requested_only = generate_json_report(sample_training_data)
        assert requested_only["postprocessing_evaluated"] is False

        evaluated = generate_json_report(self._postprocessed_data(sample_training_data))
        assert evaluated["postprocessing_evaluated"] is True
        assert evaluated["postprocessing_stages"][0]["stage_name"] == "BoutStitchingStage"

    def test_markdown_omits_postprocessing_when_not_evaluated(self, sample_training_data):
        """A report for a run without postprocessing evaluation says nothing about it."""
        report = generate_markdown_report(sample_training_data)

        assert "Postprocessing" not in report
        assert "(Postprocessed)" not in report

    def test_json_contains_postprocessed_metrics(self, sample_training_data):
        """Postprocessed metrics are serialized per iteration plus the stage list."""
        data = self._postprocessed_data(sample_training_data)

        report = generate_json_report(data)

        assert report["postprocessing_stages"][0]["stage_name"] == "BoutStitchingStage"
        postprocessed = report["cv_results"][0]["postprocessed"]
        assert postprocessed["accuracy"] == pytest.approx(0.95)
        assert postprocessed["f1_behavior"] == pytest.approx(0.9605)
        assert postprocessed["confusion_matrix"] == [[190, 10], [8, 142]]

    def test_markdown_reports_a_consistency_warning(self, sample_training_data):
        """A fold whose two prediction passes disagreed says so above the table."""
        data = self._postprocessed_data(sample_training_data)
        data.cv_results[0].postprocessed.consistency_warning = (
            "Raw accuracy from the full-sequence postprocessing pass (0.8000) does not "
            "match this iteration's raw accuracy (0.9000), so the postprocessed metrics "
            "may not be comparable with the raw ones."
        )

        report = generate_markdown_report(data)

        assert f"Iteration {data.cv_results[0].iteration}:" in report
        # flagged in the summary, where the headline means are, and again beside
        # the table - a reader who stops at the summary still sees the caveat
        summary, _, details = report.partition("### Iteration Details (Postprocessed)")
        assert "may not be comparable" in summary
        assert f"1 of {len(data.cv_results)} iterations" in summary
        assert "may not be comparable" in details
        # and beside the table, the caveat precedes the numbers it applies to
        assert details.index("may not be comparable") < details.index("0.9605")

    def test_markdown_has_no_warning_block_when_the_passes_agree(self, sample_training_data):
        """A clean run does not clutter the report with an empty warning."""
        data = self._postprocessed_data(sample_training_data)

        report = generate_markdown_report(data)

        assert "may not be comparable" not in report

    def test_markdown_notes_frames_with_no_prediction(self, sample_training_data):
        """A fold with unpredicted frames says so, since the confusion matrix hides it."""
        data = self._postprocessed_data(sample_training_data)
        data.cv_results[0].postprocessed.no_prediction_count = 3

        report = generate_markdown_report(data)

        _, _, details = report.partition("### Iteration Details (Postprocessed)")
        assert "excludes them" in details
        assert f"Iteration {data.cv_results[0].iteration}: 3 frame(s)" in details
        # the unaffected iteration is not listed
        assert f"Iteration {data.cv_results[1].iteration}: " not in details

    def test_markdown_has_no_no_prediction_note_when_fully_predicted(self, sample_training_data):
        """A clean run does not clutter the report with an empty note."""
        data = self._postprocessed_data(sample_training_data)

        report = generate_markdown_report(data)

        assert "excludes them" not in report

    def test_json_contains_the_no_prediction_count(self, sample_training_data):
        """The saved JSON carries the count so a reader can tell the matrix is partial."""
        data = self._postprocessed_data(sample_training_data)
        data.cv_results[0].postprocessed.no_prediction_count = 3

        report = generate_json_report(data)

        assert report["cv_results"][0]["postprocessed"]["no_prediction_count"] == 3
        assert report["cv_results"][1]["postprocessed"]["no_prediction_count"] == 0

    def test_json_contains_the_consistency_warning(self, sample_training_data):
        """The saved JSON carries the warning so the report is self-describing."""
        data = self._postprocessed_data(sample_training_data)
        data.cv_results[0].postprocessed.consistency_warning = "passes disagreed"

        report = generate_json_report(data)

        assert report["cv_results"][0]["postprocessed"]["consistency_warning"] == (
            "passes disagreed"
        )
        assert report["cv_results"][1]["postprocessed"]["consistency_warning"] is None

    def test_json_omits_postprocessed_when_not_evaluated(self, sample_training_data):
        """Iterations without postprocessed metrics carry no postprocessed key."""
        report = generate_json_report(sample_training_data)

        assert report["postprocessing_stages"] is None
        assert "postprocessed" not in report["cv_results"][0]
