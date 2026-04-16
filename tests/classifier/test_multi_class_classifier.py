"""Tests for MultiClassClassifier."""

import numpy as np
import pandas as pd
import pytest

from jabs.classifier.multi_class_classifier import MultiClassClassifier
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierType
from jabs.project import TrackLabels

# Shorthand for label enum values used throughout tests
_B = TrackLabels.Label.BEHAVIOR
_N = TrackLabels.Label.NOT_BEHAVIOR
_X = TrackLabels.Label.NONE

BEHAVIOR_NAMES = ["running", "grooming"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_behavior_labels() -> dict[str, np.ndarray]:
    """Label arrays for two behaviors across 12 frames.

    Layout:
        Frames 0-2:  running  (class 1)
        Frames 3-5:  grooming (class 2)
        Frames 6-8:  "None" explicit background (class 0)
        Frames 9-11: unlabeled (NONE — excluded)
    """
    n = 12
    running = np.full(n, _X, dtype=np.int8)
    grooming = np.full(n, _X, dtype=np.int8)
    none_beh = np.full(n, _X, dtype=np.int8)

    running[0:3] = _B
    grooming[3:6] = _B
    none_beh[6:9] = _B

    return {
        "running": running,
        "grooming": grooming,
        MULTICLASS_NONE_BEHAVIOR: none_beh,
    }


@pytest.fixture
def synthetic_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """12-row per-frame and window feature DataFrames."""
    rng = np.random.default_rng(0)
    per_frame = pd.DataFrame(
        {
            "feat_a": rng.standard_normal(12),
            "feat_b": rng.standard_normal(12),
        }
    )
    window = pd.DataFrame({"feat_c": rng.standard_normal(12)})
    return per_frame, window


@pytest.fixture
def trained_clf(two_behavior_labels, synthetic_features) -> MultiClassClassifier:
    """A MultiClassClassifier trained on the synthetic fixture data."""
    per_frame, window = synthetic_features
    clf = MultiClassClassifier(BEHAVIOR_NAMES)
    clf.train(
        {
            "per_frame": per_frame,
            "window": window,
            "labels_by_behavior": two_behavior_labels,
        },
        random_seed=42,
    )
    return clf


@pytest.fixture
def combined_features(synthetic_features) -> pd.DataFrame:
    """Combined per-frame + window feature DataFrame (as passed to predict)."""
    per_frame, window = synthetic_features
    return pd.concat([per_frame, window], axis=1)


# ---------------------------------------------------------------------------
# merge_labels — label merging logic
# ---------------------------------------------------------------------------


class TestMergeLabels:
    """Tests for MultiClassClassifier.merge_labels."""

    def test_behavior_frames_map_to_correct_class_index(self, two_behavior_labels):
        """Frames labeled BEHAVIOR for each behavior map to class 1 and 2."""
        labels, mask = MultiClassClassifier.merge_labels(two_behavior_labels, BEHAVIOR_NAMES)

        # 9 frames included: 3 running (class 1) + 3 grooming (class 2) + 3 none (class 0)
        assert mask.sum() == 9
        assert np.count_nonzero(labels == 1) == 3  # running
        assert np.count_nonzero(labels == 2) == 3  # grooming
        assert np.count_nonzero(labels == 0) == 3  # background

    def test_none_label_frames_excluded(self, two_behavior_labels):
        """Frames with NONE label on all behaviors are excluded from training."""
        _, mask = MultiClassClassifier.merge_labels(two_behavior_labels, BEHAVIOR_NAMES)

        # frames 9-11 are unlabeled → excluded
        assert not mask[9]
        assert not mask[10]
        assert not mask[11]

    def test_not_behavior_frames_excluded(self):
        """NOT_BEHAVIOR frames (without a matching "None" BEHAVIOR label) are excluded."""
        labels_by_behavior = {
            "walking": np.array([_B, _B, _N, _N, _X, _X], dtype=np.int8),
        }
        _, mask = MultiClassClassifier.merge_labels(labels_by_behavior, ["walking"])

        # only the first two BEHAVIOR frames are included
        assert mask.sum() == 2
        assert mask[0] and mask[1]
        assert not mask[2] and not mask[3]

    def test_none_behavior_maps_to_class_zero(self):
        """BEHAVIOR in MULTICLASS_NONE_BEHAVIOR TrackLabels maps to class 0."""
        labels_by_behavior = {
            "walking": np.array([_B, _B, _X, _X], dtype=np.int8),
            MULTICLASS_NONE_BEHAVIOR: np.array([_X, _X, _B, _B], dtype=np.int8),
        }
        labels, mask = MultiClassClassifier.merge_labels(labels_by_behavior, ["walking"])

        assert mask.sum() == 4
        np.testing.assert_array_equal(labels, [1, 1, 0, 0])

    def test_behavior_ordering_respected(self):
        """Class indices follow the order of behavior_names (1-based)."""
        labels_by_behavior = {
            "c": np.array([_B, _X, _X, _X, _X, _X], dtype=np.int8),
            "a": np.array([_X, _X, _B, _X, _X, _X], dtype=np.int8),
            "b": np.array([_X, _X, _X, _X, _B, _X], dtype=np.int8),
        }
        behavior_names = ["a", "b", "c"]  # alphabetical, class 1/2/3
        labels, _ = MultiClassClassifier.merge_labels(labels_by_behavior, behavior_names)

        # "a" → class 1, "b" → class 2, "c" → class 3
        assert 3 in labels  # "c" at frame 0
        assert 1 in labels  # "a" at frame 2
        assert 2 in labels  # "b" at frame 4

    def test_empty_labels_by_behavior_raises(self):
        """Empty labels_by_behavior raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            MultiClassClassifier.merge_labels({}, ["running"])

    def test_missing_behavior_in_dict_is_skipped(self):
        """Behavior names not present in labels_by_behavior are silently skipped."""
        labels_by_behavior = {
            "running": np.array([_B, _B, _X, _X], dtype=np.int8),
            # "grooming" intentionally absent
        }
        labels, mask = MultiClassClassifier.merge_labels(
            labels_by_behavior, ["running", "grooming"]
        )
        assert mask.sum() == 2
        np.testing.assert_array_equal(labels, [1, 1])


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestMultiClassClassifierInit:
    """Tests for MultiClassClassifier.__init__."""

    def test_default_initialization(self):
        """Classifier initializes with expected defaults."""
        clf = MultiClassClassifier(BEHAVIOR_NAMES)
        assert clf.classifier_type == ClassifierType.RANDOM_FOREST
        assert clf.behavior_names == BEHAVIOR_NAMES
        assert clf.feature_names is None

    def test_behavior_names_stored_as_copy(self):
        """Mutating the input list does not affect the stored behavior names."""
        names = ["a", "b"]
        clf = MultiClassClassifier(names)
        names.append("c")
        assert clf.behavior_names == ["a", "b"]

    def test_invalid_classifier_type_raises(self):
        """Unsupported classifier type raises ValueError."""
        from unittest.mock import patch

        with (
            patch.object(
                MultiClassClassifier, "_supported_classifier_choices", return_value=set()
            ),
            pytest.raises(ValueError, match="Invalid classifier type"),
        ):
            MultiClassClassifier(BEHAVIOR_NAMES, ClassifierType.RANDOM_FOREST)

    def test_empty_behavior_names_raises(self):
        """Empty behavior_names raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            MultiClassClassifier([])

    def test_reserved_none_behavior_name_raises(self):
        """Including MULTICLASS_NONE_BEHAVIOR in behavior_names raises ValueError."""
        with pytest.raises(ValueError, match="reserved name"):
            MultiClassClassifier(["running", MULTICLASS_NONE_BEHAVIOR])

    def test_duplicate_behavior_names_raises(self):
        """Duplicate entries in behavior_names raise ValueError."""
        with pytest.raises(ValueError, match="duplicate"):
            MultiClassClassifier(["running", "grooming", "running"])


# ---------------------------------------------------------------------------
# get_class_names
# ---------------------------------------------------------------------------


def test_get_class_names():
    """get_class_names returns background + behavior names in order."""
    clf = MultiClassClassifier(["walking", "rearing"])
    assert clf.get_class_names() == ["background", "walking", "rearing"]


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrain:
    """Tests for MultiClassClassifier.train."""

    def test_train_sets_feature_names(self, trained_clf):
        """Training populates feature_names."""
        assert trained_clf.feature_names is not None
        assert len(trained_clf.feature_names) > 0

    def test_train_missing_key_raises(self, synthetic_features):
        """Missing required key in data raises ValueError."""
        per_frame, window = synthetic_features
        clf = MultiClassClassifier(BEHAVIOR_NAMES)
        with pytest.raises(ValueError, match="Missing required key"):
            clf.train({"per_frame": per_frame, "window": window})

    def test_train_with_balance_labels(self, two_behavior_labels, synthetic_features):
        """Training with balance_labels does not raise."""
        per_frame, window = synthetic_features
        clf = MultiClassClassifier(BEHAVIOR_NAMES)
        clf.train(
            {
                "per_frame": per_frame,
                "window": window,
                "labels_by_behavior": two_behavior_labels,
                "settings": {"balance_labels": True},
            },
            random_seed=42,
        )
        assert clf.feature_names is not None

    def test_train_with_symmetric_augmentation(self, two_behavior_labels):
        """Training with symmetric_behavior does not raise."""
        per_frame = pd.DataFrame(
            {
                "left_angle": np.random.default_rng(1).standard_normal(12),
                "right_angle": np.random.default_rng(2).standard_normal(12),
            }
        )
        window = pd.DataFrame({"w": np.zeros(12)})
        clf = MultiClassClassifier(BEHAVIOR_NAMES)
        clf.train(
            {
                "per_frame": per_frame,
                "window": window,
                "labels_by_behavior": two_behavior_labels,
                "settings": {"symmetric_behavior": True},
            },
            random_seed=42,
        )
        assert clf.feature_names is not None


# ---------------------------------------------------------------------------
# predict / predict_proba
# ---------------------------------------------------------------------------


class TestPrediction:
    """Tests for predict and predict_proba."""

    def test_predict_returns_correct_shape(self, trained_clf, combined_features):
        """predict returns a 1-D array with one entry per row."""
        predictions = trained_clf.predict(combined_features)
        assert predictions.shape == (len(combined_features),)

    def test_predict_values_in_valid_range(self, trained_clf, combined_features):
        """predict outputs class indices in {-1, 0, 1, ..., N}."""
        n_classes = len(BEHAVIOR_NAMES)
        predictions = trained_clf.predict(combined_features)
        assert np.all((predictions >= -1) & (predictions <= n_classes))

    def test_predict_with_frame_indexes_fills_minus_one(self, trained_clf, combined_features):
        """Frames outside frame_indexes are set to -1."""
        frame_indexes = np.array([0, 1, 2], dtype=np.intp)
        predictions = trained_clf.predict(combined_features, frame_indexes=frame_indexes)
        assert np.all(predictions[3:] == -1)

    def test_predict_proba_shape(self, trained_clf, combined_features):
        """predict_proba returns (n_frames, N+1) where N = len(behavior_names)."""
        proba = trained_clf.predict_proba(combined_features)
        assert proba.shape == (len(combined_features), len(BEHAVIOR_NAMES) + 1)

    def test_predict_proba_sums_to_one(self, trained_clf, combined_features):
        """Probabilities across classes sum to 1 for each frame."""
        proba = trained_clf.predict_proba(combined_features)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(combined_features)), atol=1e-5)

    def test_predict_proba_with_frame_indexes_zeros_missing(self, trained_clf, combined_features):
        """Frames outside frame_indexes have zero probability."""
        frame_indexes = np.array([0, 1], dtype=np.intp)
        proba = trained_clf.predict_proba(combined_features, frame_indexes=frame_indexes)
        assert np.all(proba[2:] == 0.0)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for MultiClassClassifier save and load."""

    def test_save_and_load_round_trip(self, trained_clf, combined_features, tmp_path):
        """Saving and loading produces identical predictions."""
        path = tmp_path / "multi.pkl"
        trained_clf.save(path)
        assert path.exists()

        clf2 = MultiClassClassifier(BEHAVIOR_NAMES)
        clf2.load(path)

        assert clf2.behavior_names == trained_clf.behavior_names
        assert clf2.classifier_type == trained_clf.classifier_type
        np.testing.assert_array_equal(
            trained_clf.predict(combined_features),
            clf2.predict(combined_features),
        )

    def test_load_non_multiclass_raises(self, tmp_path):
        """Loading a file that is not a MultiClassClassifier raises ValueError."""
        import joblib

        path = tmp_path / "bad.pkl"
        joblib.dump({"not": "a classifier"}, path)

        clf = MultiClassClassifier(BEHAVIOR_NAMES)
        with pytest.raises(ValueError, match="not an instance of MultiClassClassifier"):
            clf.load(path)


# ---------------------------------------------------------------------------
# leave_one_group_out / get_leave_one_group_out_max
# ---------------------------------------------------------------------------


class TestLeaveOneGroupOut:
    """Tests for MultiClassClassifier LOGO cross-validation helpers."""

    @pytest.fixture
    def multiclass_logo_data(self):
        """Three groups; each group has 2 of 3 behavior classes plus background.

        Group 0: classes 0 and 1 (30 samples each)
        Group 1: classes 0 and 2 (30 samples each)
        Group 2: classes 1 and 2 (30 samples each)

        No single group has all 3 classes, but the training set for any held-out
        group always contains all 3.
        """
        rng = np.random.default_rng(0)
        n = 30
        labels = np.array(
            [0] * n
            + [1] * n  # group 0
            + [0] * n
            + [2] * n  # group 1
            + [1] * n
            + [2] * n,  # group 2
            dtype=np.intp,
        )
        groups = np.array([0] * (2 * n) + [1] * (2 * n) + [2] * (2 * n))
        per_frame = pd.DataFrame({"f": rng.standard_normal(len(labels))})
        window = pd.DataFrame({"w": rng.standard_normal(len(labels))})
        return per_frame, window, labels, groups

    def test_yields_splits_when_no_group_has_all_classes(self, multiclass_logo_data):
        """LOGO succeeds even when no single group contains all classes."""
        per_frame, window, labels, groups = multiclass_logo_data
        splits = list(MultiClassClassifier.leave_one_group_out(per_frame, window, labels, groups))
        assert len(splits) > 0

    def test_split_structure(self, multiclass_logo_data):
        """Each split dict has the expected keys."""
        per_frame, window, labels, groups = multiclass_logo_data
        split = next(MultiClassClassifier.leave_one_group_out(per_frame, window, labels, groups))
        for key in (
            "training_data",
            "training_labels",
            "test_data",
            "test_labels",
            "test_group",
            "feature_names",
        ):
            assert key in split

    def test_training_split_has_all_classes(self, multiclass_logo_data):
        """Every yielded split has all classes in the training portion."""
        per_frame, window, labels, groups = multiclass_logo_data
        all_classes = np.unique(labels)
        for split in MultiClassClassifier.leave_one_group_out(per_frame, window, labels, groups):
            for cls in all_classes:
                assert np.count_nonzero(split["training_labels"] == cls) > 0

    def test_get_leave_one_group_out_max(self, multiclass_logo_data):
        """get_leave_one_group_out_max counts groups that yield valid splits."""
        _, _, labels, groups = multiclass_logo_data
        max_splits = MultiClassClassifier.get_leave_one_group_out_max(labels, groups)
        assert max_splits == 3  # all three groups are valid test sets

    def test_get_leave_one_group_out_max_zero_when_training_incomplete(self):
        """Returns 0 when no split leaves a training set with all classes."""
        # Only 2 groups; each has only one class — training set always missing classes.
        labels = np.array([0] * 30 + [1] * 30, dtype=np.intp)
        groups = np.array([0] * 30 + [1] * 30)
        # group 0 test: training only has class 1; group 1 test: training only has class 0
        assert MultiClassClassifier.get_leave_one_group_out_max(labels, groups) == 0


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_satisfies_classifier_protocol():
    """MultiClassClassifier structurally satisfies ClassifierProtocol."""
    # Runtime check: verify all required attributes are present
    clf = MultiClassClassifier(BEHAVIOR_NAMES)
    protocol_methods = ("train", "predict", "predict_proba", "save", "load")
    for method in protocol_methods:
        assert callable(getattr(clf, method, None)), f"Missing method: {method}"
    assert hasattr(clf, "feature_names"), "Missing property: feature_names"
