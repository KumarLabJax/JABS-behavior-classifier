"""Tests for training report generation."""

from datetime import datetime

import numpy as np
import pytest

from jabs.classifier.training_report import (
    CrossValidationResult,
    TrainingReportData,
    generate_markdown_report,
    save_training_report,
)
from jabs.types import CrossValidationGroupingStrategy


@pytest.fixture
def sample_cv_results():
    """Create sample cross-validation results.

    Returns:
        List of CrossValidationResult objects.
    """
    return [
        CrossValidationResult(
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
        CrossValidationResult(
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
    """Tests for CrossValidationResult dataclass."""

    def test_create_cv_result(self):
        """Test creating a CrossValidationResult instance."""
        result = CrossValidationResult(
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
        print("\n--- Markdown Report ---\n", report, "\n--- End Report ---\n")

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

    def test_markdown_escaping_in_video_names(self, sample_training_data):
        """Test that special characters in video names are escaped."""
        sample_training_data.cv_results[0].test_label = "test_video_with_underscores.mp4 [0]"
        report = generate_markdown_report(sample_training_data)
        print("\n--- Markdown Report ---\n", report, "\n--- End Report ---\n")
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
