"""Tests for the Classifier class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from jabs.classifier.classifier import Classifier
from jabs.core.enums import ClassifierType, CrossValidationGroupingStrategy
from jabs.project import TrackLabels


@pytest.fixture
def sample_features():
    """Create sample feature data for testing.

    Returns:
        DataFrame with sample features for 100 frames.
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "left_ear_angle": np.random.randn(100),
            "right_ear_angle": np.random.randn(100),
        }
    )


@pytest.fixture
def sample_labels():
    """Create sample labels for testing.

    Returns:
        Array of labels (50 BEHAVIOR, 50 NOT_BEHAVIOR).
    """
    labels = np.array([TrackLabels.Label.BEHAVIOR] * 50 + [TrackLabels.Label.NOT_BEHAVIOR] * 50)
    np.random.seed(42)
    np.random.shuffle(labels)
    return labels


@pytest.fixture
def sample_groups():
    """Create sample group IDs for leave-one-group-out testing.

    Returns:
        Array of group IDs (2 groups of 50 samples each).
        This ensures each group has enough labels to meet the threshold.
    """
    return np.repeat([0, 1], 50)


@pytest.fixture
def mock_project():
    """Create a mock project for testing.

    Returns:
        MagicMock configured with necessary project settings.
    """
    project = MagicMock()
    project.get_project_defaults.return_value = {
        "balance_labels": False,
        "symmetric_behavior": False,
    }
    project.settings_manager.get_behavior.return_value = {
        "balance_labels": False,
        "symmetric_behavior": False,
    }
    return project


class TestClassifierInitialization:
    """Test Classifier initialization."""

    def test_default_initialization(self):
        """Test creating a classifier with default parameters."""
        clf = Classifier()
        assert clf.classifier_type == ClassifierType.RANDOM_FOREST
        assert clf._n_jobs == 1
        assert clf._classifier is None
        assert clf._behavior is None

    def test_initialization_with_classifier_type(self):
        """Test creating a classifier with specific type."""
        clf = Classifier(classifier=ClassifierType.RANDOM_FOREST, n_jobs=4)
        assert clf.classifier_type == ClassifierType.RANDOM_FOREST
        assert clf._n_jobs == 4

    @pytest.mark.skipif(
        ClassifierType.XGBOOST not in Classifier()._supported_classifiers,
        reason="XGBoost not available",
    )
    def test_initialization_with_xgboost(self):
        """Test creating a classifier with XGBoost type."""
        clf = Classifier(classifier=ClassifierType.XGBOOST)
        assert clf.classifier_type == ClassifierType.XGBOOST

    def test_initialization_with_catboost(self):
        """Test creating a classifier with CatBoost type."""
        clf = Classifier(classifier=ClassifierType.CATBOOST)
        assert clf.classifier_type == ClassifierType.CATBOOST

    def test_invalid_classifier_type_raises_error(self):
        """Test that invalid classifier type raises ValueError."""
        with (
            pytest.raises(ValueError, match="Invalid classifier type"),
            patch.object(Classifier, "_supported_classifier_choices") as mock,
        ):
            mock.return_value = set()
            Classifier(classifier=ClassifierType.RANDOM_FOREST)


class TestClassifierProperties:
    """Test Classifier properties."""

    def test_classifier_name_property(self):
        """Test classifier_name property returns correct name."""
        clf = Classifier(classifier=ClassifierType.RANDOM_FOREST)
        assert clf.classifier_name == ClassifierType.RANDOM_FOREST.value

    def test_behavior_name_property(self):
        """Test behavior_name getter and setter."""
        clf = Classifier()
        assert clf.behavior_name is None

        clf.behavior_name = "Grooming"
        assert clf.behavior_name == "Grooming"

    def test_version_property(self):
        """Test version property returns correct version."""
        clf = Classifier()
        assert isinstance(clf.version, int)
        assert clf.version > 0

    def test_classifier_file_property_unset(self):
        """Test classifier_file property when not set."""
        clf = Classifier()
        assert clf.classifier_file == "NO SAVED CLASSIFIER"

    def test_classifier_hash_property_unset(self):
        """Test classifier_hash property when not set."""
        clf = Classifier()
        assert clf.classifier_hash == "NO HASH"

    def test_project_settings_property(self):
        """Test project_settings property returns copy."""
        clf = Classifier()
        assert clf.project_settings == {}

        # Set settings and verify we get a copy
        clf._project_settings = {"test": "value"}
        settings = clf.project_settings
        assert settings == {"test": "value"}
        # Modify returned dict shouldn't affect internal settings
        settings["test"] = "modified"
        assert clf._project_settings["test"] == "value"

    def test_feature_names_property(self):
        """Test feature_names property."""
        clf = Classifier()
        assert clf.feature_names is None


class TestDataSplitting:
    """Test data splitting methods."""

    def test_leave_one_group_out(self, sample_features, sample_labels, sample_groups):
        """Test leave_one_group_out splitting."""
        per_frame = sample_features[["feature_1", "feature_2"]]
        window = sample_features[["feature_3"]]

        splits = list(
            Classifier.leave_one_group_out(per_frame, window, sample_labels, sample_groups)
        )

        # Should generate at least one split
        assert len(splits) > 0

        # Check first split structure
        split = splits[0]
        assert "training_data" in split
        assert "test_data" in split
        assert "training_labels" in split
        assert "test_labels" in split
        assert "test_group" in split
        assert "feature_names" in split

    def test_get_leave_one_group_out_max(self, sample_labels, sample_groups):
        """Test counting maximum leave-one-out groups."""
        max_groups = Classifier.get_leave_one_group_out_max(sample_labels, sample_groups)

        assert isinstance(max_groups, int | np.integer)
        assert max_groups >= 0
        assert max_groups <= len(np.unique(sample_groups))


class TestDataAugmentation:
    """Test data augmentation and balancing methods."""

    def test_downsample_balance(self, sample_features, sample_labels):
        """Test downsampling balances label distribution."""
        # Create imbalanced labels
        imbalanced_labels = np.array(
            [TrackLabels.Label.BEHAVIOR] * 80 + [TrackLabels.Label.NOT_BEHAVIOR] * 20
        )

        balanced_features, balanced_labels = Classifier.downsample_balance(
            sample_features, imbalanced_labels, random_seed=42
        )

        # Check that labels are now balanced
        unique, counts = np.unique(balanced_labels, return_counts=True)
        assert len(set(counts)) == 1  # All counts should be equal
        assert counts[0] == 20  # Should match minority class

    def test_augment_symmetric(self, sample_features, sample_labels):
        """Test symmetric augmentation swaps left/right features."""
        augmented_features, augmented_labels = Classifier.augment_symmetric(
            sample_features, sample_labels
        )

        # Should double the data
        assert len(augmented_features) == len(sample_features) * 2
        assert len(augmented_labels) == len(sample_labels) * 2

        # Check that left/right features were swapped in second half
        # Original left_ear_angle should match augmented right_ear_angle
        original_left = sample_features["left_ear_angle"].values
        augmented_right = augmented_features["right_ear_angle"].values[len(sample_features) :]
        np.testing.assert_array_equal(original_left, augmented_right)


class TestClassifierTraining:
    """Test classifier training."""

    def test_train_random_forest(self, sample_features, sample_labels, mock_project):
        """Test training a Random Forest classifier."""
        clf = Classifier(classifier=ClassifierType.RANDOM_FOREST)
        clf.behavior_name = "Test Behavior"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
            "feature_names": sample_features.columns.to_list(),
        }

        clf.train(data, random_seed=42)

        assert clf._classifier is not None
        assert clf.feature_names == sample_features.columns.to_list()

    def test_train_without_project_settings_raises_error(self, sample_features, sample_labels):
        """Test that training without project settings raises ValueError."""
        clf = Classifier()

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
        }

        with pytest.raises(ValueError, match="Project settings for classifier unset"):
            clf.train(data)

    def test_train_with_balancing(self, sample_features, sample_labels, mock_project):
        """Test training with label balancing enabled."""
        mock_project.settings_manager.get_behavior.return_value = {
            "balance_labels": True,
            "symmetric_behavior": False,
        }

        clf = Classifier()
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        # Create imbalanced data
        imbalanced_labels = np.array(
            [TrackLabels.Label.BEHAVIOR] * 80 + [TrackLabels.Label.NOT_BEHAVIOR] * 20
        )

        data = {
            "training_data": sample_features,
            "training_labels": imbalanced_labels,
        }

        clf.train(data, random_seed=42)
        assert clf._classifier is not None

    def test_train_with_symmetric_augmentation(self, sample_features, sample_labels, mock_project):
        """Test training with symmetric augmentation enabled."""
        mock_project.settings_manager.get_behavior.return_value = {
            "balance_labels": False,
            "symmetric_behavior": True,
        }

        clf = Classifier()
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
        }

        clf.train(data, random_seed=42)
        assert clf._classifier is not None

    def test_train_catboost(self, sample_features, sample_labels, mock_project):
        """Test training a CatBoost classifier."""
        clf = Classifier(classifier=ClassifierType.CATBOOST)
        clf.behavior_name = "Test Behavior"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
            "feature_names": sample_features.columns.to_list(),
        }

        clf.train(data, random_seed=42)

        assert clf._classifier is not None
        assert clf.feature_names == sample_features.columns.to_list()

    def test_train_catboost_with_nan_values(self, sample_features, sample_labels, mock_project):
        """Test training CatBoost with NaN values in features.

        CatBoost should handle NaN values natively without imputation.
        """
        clf = Classifier(classifier=ClassifierType.CATBOOST)
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        # Add some NaN values to features
        features_with_nan = sample_features.copy()
        features_with_nan.iloc[10:20, 0] = np.nan
        features_with_nan.iloc[30:35, 2] = np.nan

        data = {
            "training_data": features_with_nan,
            "training_labels": sample_labels,
            "feature_names": features_with_nan.columns.to_list(),
        }

        # Should train successfully without errors
        clf.train(data, random_seed=42)
        assert clf._classifier is not None


class TestClassifierPrediction:
    """Test classifier prediction methods."""

    def test_predict(self, sample_features, sample_labels, mock_project):
        """Test predict method."""
        clf = Classifier()
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        # Train the classifier
        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
        }
        clf.train(data, random_seed=42)

        # Make predictions
        predictions = clf.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(
            p in [TrackLabels.Label.BEHAVIOR, TrackLabels.Label.NOT_BEHAVIOR] for p in predictions
        )

    def test_predict_proba(self, sample_features, sample_labels, mock_project):
        """Test predict_proba method."""
        clf = Classifier()
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
        }
        clf.train(data, random_seed=42)

        probabilities = clf.predict_proba(sample_features)

        assert probabilities.shape == (len(sample_features), 2)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1), np.ones(len(sample_features))
        )

    def test_predict_with_inf_and_nan(self, sample_features, sample_labels, mock_project):
        """Test prediction handles inf and NaN values."""
        clf = Classifier()
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
        }
        clf.train(data, random_seed=42)

        # Create features with inf and NaN
        test_features = sample_features.copy()
        test_features.iloc[0, 0] = np.inf
        test_features.iloc[1, 1] = -np.inf
        test_features.iloc[2, 2] = np.nan

        # Should not raise error
        predictions = clf.predict(test_features)
        assert len(predictions) == len(test_features)

    def test_predict_catboost(self, sample_features, sample_labels, mock_project):
        """Test predict method with CatBoost classifier."""
        clf = Classifier(classifier=ClassifierType.CATBOOST)
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
        }
        clf.train(data, random_seed=42)

        predictions = clf.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(
            p in [TrackLabels.Label.BEHAVIOR, TrackLabels.Label.NOT_BEHAVIOR] for p in predictions
        )

    def test_predict_proba_catboost(self, sample_features, sample_labels, mock_project):
        """Test predict_proba method with CatBoost classifier."""
        clf = Classifier(classifier=ClassifierType.CATBOOST)
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
        }
        clf.train(data, random_seed=42)

        probabilities = clf.predict_proba(sample_features)

        assert probabilities.shape == (len(sample_features), 2)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1), np.ones(len(sample_features))
        )

    def test_catboost_handles_nan_in_prediction(
        self, sample_features, sample_labels, mock_project
    ):
        """Test that CatBoost handles NaN values during prediction.

        CatBoost should handle NaN natively without requiring imputation.
        """
        clf = Classifier(classifier=ClassifierType.CATBOOST)
        clf.behavior_name = "Test"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
        }
        clf.train(data, random_seed=42)

        # Create test features with NaN values
        test_features = sample_features.copy()
        test_features.iloc[5:10, 1] = np.nan

        # Should handle NaN without error
        predictions = clf.predict(test_features)
        assert len(predictions) == len(test_features)


class TestClassifierSaveLoad:
    """Test classifier save and load functionality."""

    def test_save_and_load(self, sample_features, sample_labels, mock_project, tmp_path):
        """Test saving and loading a trained classifier.

        Note: Currently, feature_names is not preserved during save/load.
        If this is changed, we should check that after loading, feature_names match.
        """
        # Train a classifier
        clf = Classifier()
        clf.behavior_name = "Grooming"
        clf.set_project_settings(mock_project)

        data = {
            "training_data": sample_features,
            "training_labels": sample_labels,
            "feature_names": sample_features.columns.to_list(),
        }
        clf.train(data, random_seed=42)

        # Save the classifier
        save_path = tmp_path / "test_classifier.pkl"
        clf.save(save_path)

        assert save_path.exists()
        assert clf.classifier_file == "test_classifier.pkl"
        assert clf.classifier_hash is not None

        # Load the classifier
        clf2 = Classifier()
        clf2.load(save_path)

        assert clf2.behavior_name == "Grooming"
        assert clf2.classifier_type == clf.classifier_type

        # Predictions should still work and match
        pred1 = clf.predict(sample_features)
        pred2 = clf2.predict(sample_features)
        np.testing.assert_array_equal(pred1, pred2)

    def test_load_invalid_file_raises_error(self, tmp_path):
        """Test loading non-classifier file raises ValueError."""
        # Create a file with wrong content
        invalid_file = tmp_path / "invalid.pkl"
        import joblib

        joblib.dump({"not": "a classifier"}, invalid_file)

        clf = Classifier()
        with pytest.raises(ValueError, match="not instance of Classifier"):
            clf.load(invalid_file)


class TestClassifierSettings:
    """Test classifier settings management."""

    def test_set_project_settings_with_behavior(self, mock_project):
        """Test setting project settings for specific behavior."""
        clf = Classifier()
        clf.behavior_name = "Grooming"

        clf.set_project_settings(mock_project)

        assert clf.project_settings is not None
        mock_project.settings_manager.get_behavior.assert_called_with("Grooming")

    def test_set_project_settings_without_behavior(self, mock_project):
        """Test setting project settings without behavior uses defaults."""
        clf = Classifier()

        clf.set_project_settings(mock_project)

        assert clf.project_settings is not None
        mock_project.get_project_defaults.assert_called_once()

    def test_set_dict_settings(self):
        """Test setting project settings via dictionary."""
        clf = Classifier()
        settings = {
            "balance_labels": True,
            "symmetric_behavior": False,
        }

        clf.set_dict_settings(settings)

        assert clf.project_settings == settings


class TestClassifierChoices:
    """Test classifier type choices."""

    def test_classifier_choices(self):
        """Test getting available classifier choices."""
        clf = Classifier()
        choices = clf.classifier_choices()

        assert isinstance(choices, dict)
        assert ClassifierType.RANDOM_FOREST in choices
        assert ClassifierType.CATBOOST in choices
        assert all(isinstance(v, str) for v in choices.values())

    def test_catboost_in_choices(self):
        """Test that CatBoost is available in classifier choices."""
        clf = Classifier()
        choices = clf.classifier_choices()

        assert ClassifierType.CATBOOST in choices
        assert choices[ClassifierType.CATBOOST] == "CatBoost"

    def test_set_classifier(self):
        """Test changing classifier type."""
        clf = Classifier(classifier=ClassifierType.RANDOM_FOREST)

        # Change to same type should work
        clf.set_classifier(ClassifierType.RANDOM_FOREST)
        assert clf.classifier_type == ClassifierType.RANDOM_FOREST

    def test_set_classifier_to_catboost(self):
        """Test switching to CatBoost classifier type."""
        clf = Classifier(classifier=ClassifierType.RANDOM_FOREST)

        # Switch to CatBoost
        clf.set_classifier(ClassifierType.CATBOOST)
        assert clf.classifier_type == ClassifierType.CATBOOST


class TestStaticMethods:
    """Test static utility methods."""

    def test_accuracy_score(self):
        """Test accuracy_score calculation."""
        truth = np.array([0, 0, 1, 1, 0, 1])
        predictions = np.array([0, 1, 1, 1, 0, 0])

        accuracy = Classifier.accuracy_score(truth, predictions)

        assert 0.0 <= accuracy <= 1.0
        assert accuracy == pytest.approx(2 / 3)

    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        truth = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 0])

        cm = Classifier.confusion_matrix(truth, predictions)

        assert cm.shape == (2, 2)
        assert cm.sum() == len(truth)

    def test_combine_data(self, sample_features):
        """Test combining per-frame and window features."""
        per_frame = sample_features[["feature_1", "feature_2"]]
        window = sample_features[["feature_3"]]

        combined = Classifier.combine_data(per_frame, window)

        assert len(combined.columns) == 3
        assert list(combined.columns) == ["feature_1", "feature_2", "feature_3"]
        assert len(combined) == len(sample_features)

    def test_count_label_threshold(self):
        """Test counting groups meeting label threshold."""
        all_counts = {
            "video1.avi": {
                0: {
                    "fragmented_frame_counts": (25, 25),
                    "fragmented_bout_counts": (5, 5),
                    "unfragmented_frame_counts": (25, 25),
                    "unfragmented_bout_counts": (5, 5),
                },
                1: {
                    "fragmented_frame_counts": (10, 10),  # Below threshold
                    "fragmented_bout_counts": (2, 2),
                    "unfragmented_frame_counts": (10, 10),
                    "unfragmented_bout_counts": (2, 2),
                },
            }
        }

        count = Classifier.count_label_threshold(all_counts)

        # Only first identity meets threshold (>= 20)
        assert count == 1

    def test_label_threshold_met(self):
        """Test checking if label threshold is met."""
        all_counts = {
            "video1.avi": {
                0: {
                    "fragmented_frame_counts": (25, 25),
                    "fragmented_bout_counts": (5, 5),
                    "unfragmented_frame_counts": (25, 25),
                    "unfragmented_bout_counts": (5, 5),
                },
                1: {
                    "fragmented_frame_counts": (30, 30),
                    "fragmented_bout_counts": (6, 6),
                    "unfragmented_frame_counts": (30, 30),
                    "unfragmented_bout_counts": (6, 6),
                },
            }
        }

        # INDIVIDUAL: Two groups meet threshold, need at least 2
        assert Classifier.label_threshold_met(
            all_counts,
            min_groups=2,
            cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
        )
        # INDIVIDUAL: Two groups meet threshold, but need at least 3
        assert not Classifier.label_threshold_met(
            all_counts,
            min_groups=3,
            cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
        )

        # VIDEO: Only one video, should fail for min_groups=1 because we can't split into a train and test set
        assert not Classifier.label_threshold_met(
            all_counts, min_groups=1, cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO
        )

        # Add a second video for VIDEO grouping strategy
        multi_video_counts = {
            "video1.avi": {
                0: {
                    "fragmented_frame_counts": (25, 25),
                    "fragmented_bout_counts": (5, 5),
                    "unfragmented_frame_counts": (25, 25),
                    "unfragmented_bout_counts": (5, 5),
                }
            },
            "video2.avi": {
                0: {
                    "fragmented_frame_counts": (30, 30),
                    "fragmented_bout_counts": (6, 6),
                    "unfragmented_frame_counts": (30, 30),
                    "unfragmented_bout_counts": (6, 6),
                }
            },
        }
        # VIDEO: Two videos, both meet threshold, min_groups=2 should pass
        assert Classifier.label_threshold_met(
            multi_video_counts,
            min_groups=2,
            cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO,
        )
        # VIDEO: Two videos, min_groups=3 should fail
        assert not Classifier.label_threshold_met(
            multi_video_counts,
            min_groups=3,
            cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO,
        )
        # VIDEO: One video below threshold, one above
        multi_video_counts_below = {
            "video1.avi": {
                0: {
                    "fragmented_frame_counts": (10, 10),
                    "fragmented_bout_counts": (2, 2),
                    "unfragmented_frame_counts": (10, 10),
                    "unfragmented_bout_counts": (2, 2),
                }
            },
            "video2.avi": {
                0: {
                    "fragmented_frame_counts": (30, 30),
                    "fragmented_bout_counts": (6, 6),
                    "unfragmented_frame_counts": (30, 30),
                    "unfragmented_bout_counts": (6, 6),
                }
            },
            "video3.avi": {
                0: {
                    "fragmented_frame_counts": (30, 30),
                    "fragmented_bout_counts": (6, 6),
                    "unfragmented_frame_counts": (30, 30),
                    "unfragmented_bout_counts": (6, 6),
                }
            },
        }
        # video2.avi and video3 meet threshold, min_groups=2 should pass
        assert Classifier.label_threshold_met(
            multi_video_counts_below,
            min_groups=2,
            cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO,
        )
        assert not Classifier.label_threshold_met(
            multi_video_counts_below,
            min_groups=3,
            cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO,
        )


class TestFromTrainingFile:
    """Test creating classifier from training file."""

    def test_from_training_file(self, tmp_path, sample_features, sample_labels):
        """Test creating classifier from training file."""
        training_file_path = tmp_path / "training_data.h5"

        # Create mock training data
        training_data = {
            "behavior": "Grooming",
            "classifier_type": ClassifierType.RANDOM_FOREST.value,
            "settings": {
                "balance_labels": False,
                "symmetric_behavior": False,
            },
            "per_frame": sample_features[["feature_1", "feature_2"]],
            "window": sample_features[["feature_3"]],
            "labels": sample_labels,
            "training_seed": 42,
        }

        # Mock both load_training_data and hash_file
        with (
            patch("jabs.classifier.classifier.load_training_data") as mock_load,
            patch("jabs.classifier.classifier.hash_file") as mock_hash,
        ):
            mock_load.return_value = (training_data, None)
            mock_hash.return_value = "mock_hash_value"

            clf = Classifier.from_training_file(training_file_path)

            assert clf.behavior_name == "Grooming"
            assert clf.classifier_type == ClassifierType.RANDOM_FOREST
            assert clf._classifier is not None
            assert clf.classifier_file == training_file_path.name
            assert clf._classifier_source == "training_file"
