import random
import re
import typing
import warnings
from importlib import import_module
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from jabs.constants import DEFAULT_CALIBRATION_CV, DEFAULT_CALIBRATION_METHOD
from jabs.project import Project, TrackLabels, load_training_data
from jabs.types import ClassifierType
from jabs.utils import hash_file

matplotlib.use("Agg")  # use non-GUI backend to avoid thread warnings


_VERSION = 10

_classifier_choices = [ClassifierType.RANDOM_FOREST, ClassifierType.GRADIENT_BOOSTING]

try:
    _xgboost = import_module("xgboost")
    # we were able to import xgboost, make it available as an option:
    _classifier_choices.append(ClassifierType.XGBOOST)
except Exception:
    # we were unable to import the xgboost module -- possibly due to a missing
    # libomp (which is not available by default on macOS). Mac users should
    # install libomp via Homebrew (brew install libomp) to enable XGBoost support (this is
    # detailed in the installation instructions).
    # we won't add it to the available choices and we can otherwise ignore this exception
    _xgboost = None


class Classifier:
    """A machine learning classifier for behavior classification tasks.

    This class supports training, evaluating, saving, and loading classifiers
    for behavioral data using Random Forest, Gradient Boosting, or XGBoost algorithms.
    It provides utilities for data splitting, balancing, augmentation, and feature management.

    Attributes:
        LABEL_THRESHOLD (int): Minimum number of labels required per group.
    """

    LABEL_THRESHOLD = 20
    TRUE_THRESHOLD = 0.5
    CALIBRATION_METHODS: typing.ClassVar[list[str]] = ["auto", "isotonic", "sigmoid"]

    _CLASSIFIER_NAMES: typing.ClassVar[dict] = {
        ClassifierType.RANDOM_FOREST: "Random Forest",
        ClassifierType.GRADIENT_BOOSTING: "Gradient Boosting",
        ClassifierType.XGBOOST: "XGBoost",
    }

    def __init__(self, classifier=ClassifierType.RANDOM_FOREST, n_jobs=1):
        self._classifier_type = classifier
        self._classifier = None
        self._behavior_settings = None
        self._jabs_settings = None
        self._behavior = None
        self._feature_names = None
        self._n_jobs = n_jobs
        self._version = _VERSION

        self._classifier_file = None
        self._classifier_hash = None
        self._classifier_source = None

        # make sure the value passed for the classifier parameter is valid
        if classifier not in _classifier_choices:
            raise ValueError("Invalid classifier type")

    @classmethod
    def from_training_file(cls, path: Path):
        """Initialize a classifier from an exported training data file.

        This method will load the training data and train a classifier.

        Args:
            path: exported training data file

        Returns:
            trained classifier object
        """
        loaded_training_data, _ = load_training_data(path)
        behavior = loaded_training_data["behavior"]

        classifier = cls()
        classifier.behavior_name = behavior
        classifier.set_behavior_settings(loaded_training_data["behavior_settings"])
        classifier._jabs_settings = loaded_training_data["jabs_settings"]

        classifier_type = ClassifierType(loaded_training_data["classifier_type"])
        if classifier_type in classifier.classifier_choices():
            classifier.set_classifier(classifier_type)
        else:
            print(
                f"Specified classifier type {classifier_type.name} is unavailable, using default: {classifier.classifier_type.name}"
            )

        training_features = classifier.combine_data(
            loaded_training_data["per_frame"], loaded_training_data["window"]
        )
        classifier.train(
            {
                "training_data": training_features,
                "training_labels": loaded_training_data["labels"],
            },
            random_seed=loaded_training_data["training_seed"],
        )

        classifier._classifier_file = Path(path).name
        classifier._classifier_hash = hash_file(Path(path))
        classifier._classifier_source = "training_file"

        return classifier

    @property
    def classifier_name(self) -> str:
        """return the name of the classifier used as a string"""
        return self._CLASSIFIER_NAMES[self._classifier_type]

    @property
    def classifier_type(self) -> ClassifierType:
        """return classifier type"""
        return self._classifier_type

    @property
    def classifier_file(self) -> str:
        """return the filename of the saved classifier"""
        if self._classifier_file is not None:
            return self._classifier_file
        return "NO SAVED CLASSIFIER"

    @property
    def classifier_hash(self) -> str:
        """return the hash of the classifier file"""
        if self._classifier_hash is not None:
            return self._classifier_hash
        return "NO HASH"

    @property
    def behavior_settings(self) -> dict:
        """return a copy of dictionary of behavior-specific settings for this classifier"""
        if self._behavior_settings is not None:
            return dict(self._behavior_settings)
        return {}

    @property
    def behavior_name(self) -> str:
        """return the behavior name property"""
        return self._behavior

    @behavior_name.setter
    def behavior_name(self, value) -> None:
        """set the behavior name property"""
        self._behavior = value

    @property
    def version(self) -> int:
        """return the classifier format version"""
        return self._version

    @property
    def feature_names(self) -> list:
        """returns the list of feature names used when training this classifier"""
        return self._feature_names

    @property
    def calibrate_probabilities(self) -> bool:
        """return whether the classifier is set to calibrate probabilities"""
        if self._jabs_settings is not None:
            return self._jabs_settings.get("calibrate_probabilities", False)
        return False

    @staticmethod
    def _choose_auto_calibration_method(
        labels: np.ndarray, calibration_cv: int
    ) -> tuple[str, dict]:
        """Choose 'isotonic' or 'sigmoid' based on data size per calibration fold.

        Heuristic:
          - Compute class counts on the *training set labels* passed in.
          - Estimate per-fold calibration set size as min(pos, neg) / calibration_cv
            (because CalibratedClassifierCV uses 1/cv of the train split for calibration).
          - If per-fold per-class counts >= 500 ➜ 'isotonic', else 'sigmoid'.

        Returns:
          (method, info_dict) where info_dict contains counts used for logging.
        """
        # count positive and negative labels
        pos = int(np.sum(labels == TrackLabels.Label.BEHAVIOR))
        neg = int(np.sum(labels == TrackLabels.Label.NOT_BEHAVIOR))
        min_per_class = min(pos, neg)
        per_fold_per_class = max(0, min_per_class // calibration_cv)

        # Threshold for isotonic safety
        threshold = 500
        method = "isotonic" if per_fold_per_class >= threshold else "sigmoid"
        return method, {
            "pos_total": pos,
            "neg_total": neg,
            "cv": calibration_cv,
            "per_fold_per_class": per_fold_per_class,
            "threshold": threshold,
        }

    @staticmethod
    def train_test_split(per_frame_features, window_features, label_data):
        """split features and labels into training and test datasets

        Args:
            per_frame_features: per frame features as returned from IdentityFeatures object, filtered to only include labeled frames
            window_features: window features as returned from IdentityFeatures object, filtered to only include labeled frames
            label_data: labels that correspond to the features

        Returns:
            dictionary of training and test data and labels:

        {
            'training_data': list of numpy arrays,
            'test_data': list of numpy arrays,
            'training_labels': numpy array,
            'test_labels': numpy_array,
            'feature_names': list of feature names
        }
        """
        # split labeled data and labels
        all_features = pd.concat([per_frame_features, window_features], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(all_features, label_data)

        return {
            "training_data": x_train,
            "training_labels": y_train,
            "test_data": x_test,
            "test_labels": y_test,
            "feature_names": all_features.columns.to_list(),
        }

    @staticmethod
    def get_leave_one_group_out_max(labels, groups):
        """counts the number of possible leave one out groups for k-fold cross validation

        Args:
            labels: labels to check if they were above the threshold
            groups: group id corresponding to the labels

        Returns:
            int of the maximum number of cross validation to use

        Note: labels excludes label for frames with no identity.
        """
        unique_groups = np.unique(groups)
        count_behavior = [
            np.sum(np.asarray(labels)[np.asarray(groups) == x] == TrackLabels.Label.BEHAVIOR)
            for x in unique_groups
        ]
        count_not_behavior = [
            np.sum(np.asarray(labels)[np.asarray(groups) == x] == TrackLabels.Label.NOT_BEHAVIOR)
            for x in unique_groups
        ]
        can_kfold = np.logical_and(
            np.asarray(count_behavior) > Classifier.LABEL_THRESHOLD,
            np.asarray(count_not_behavior) > Classifier.LABEL_THRESHOLD,
        )
        return np.sum(can_kfold)

    @staticmethod
    def leave_one_group_out(per_frame_features, window_features, labels, groups):
        """implements "leave one group out" data splitting strategy

        Args:
            per_frame_features: per frame features for all labeled data
            window_features: window features for all labeled data
            labels: labels corresponding to each feature row
            groups: group id corresponding to each feature row

        Returns:
            dictionary of training and test data and labels:
        {
            'training_data': list of numpy arrays,
            'test_data': list of numpy arrays,
            'training_labels': numpy array,
            'test_labels': numpy_array,
            'feature_names': list of feature names
        }
        """
        logo = LeaveOneGroupOut()
        x = Classifier.combine_data(per_frame_features, window_features)
        splits = list(logo.split(x, labels, groups))

        # pick random split, make sure we pick a split where the test data
        # has sufficient labels of both classes
        random.shuffle(splits)
        count = 0
        for split in splits:
            behavior_count = np.count_nonzero(labels[split[1]] == TrackLabels.Label.BEHAVIOR)
            not_behavior_count = np.count_nonzero(
                labels[split[1]] == TrackLabels.Label.NOT_BEHAVIOR
            )

            if (
                behavior_count >= Classifier.LABEL_THRESHOLD
                and not_behavior_count >= Classifier.LABEL_THRESHOLD
            ):
                count += 1
                yield {
                    "training_data": x.iloc[split[0]],
                    "training_labels": labels[split[0]],
                    "test_data": x.iloc[split[1]],
                    "test_labels": labels[split[1]],
                    "test_group": groups[split[1]][0],
                    "feature_names": x.columns.to_list(),
                }

        # number of splits exhausted without finding at least one that meets
        # criteria
        # the UI won't allow us to reach this case
        if count == 0:
            raise ValueError("unable to split data")
        # If there are no more splits to yield, just let generator end

    @staticmethod
    def downsample_balance(features, labels, random_seed=None):
        """downsamples features and labels such that labels are equally distributed

        Args:
            features: features to downsample
            labels: labels to downsample
            random_seed: optional random seed

        Returns:
            tuple of downsampled features, labels
        """
        label_states, label_counts = np.unique(labels, return_counts=True)
        max_examples_per_class = np.min(label_counts)
        selected_samples = []
        for cur_label in label_states:
            idxs = np.where(labels == cur_label)[0]
            if random_seed is not None:
                np.random.seed(random_seed)
            sampled_idxs = np.random.choice(idxs, max_examples_per_class, replace=False)
            selected_samples.append(sampled_idxs)
        selected_samples = np.sort(np.concatenate(selected_samples))
        features = features.iloc[selected_samples]
        labels = labels[selected_samples]
        return features, labels

    @staticmethod
    def augment_symmetric(features, labels, random_str="ASygRQDZJD"):
        """augments the features to include L-R and R-L duplicates

        This requires 'left' or 'right' to be in the feature name to be swapped
        Features that don't include these terms will not be swapped

        Args:
            features: features to augment
            labels: labels to augment
            random_str: a random string to use as a temporary
                replacement when swapping left/right

        Returns:
            tuple of augmented features, labels
        """
        # Figure out the L-R swapping of features
        lowercase_features = np.array([x.lower() for x in features.columns.to_list()])
        reflected_feature_names = [re.sub(r"left", random_str, x) for x in lowercase_features]
        reflected_feature_names = [re.sub(r"right", "left", x) for x in reflected_feature_names]
        reflected_feature_names = [re.sub(random_str, "right", x) for x in reflected_feature_names]
        reflected_idxs = [
            np.where(lowercase_features == x)[0][0] if x in lowercase_features else i
            for i, x in enumerate(reflected_feature_names)
        ]
        # expand the features with reflections
        features_duplicate = features.copy()
        features_duplicate.columns = features.columns.to_numpy()[np.asarray(reflected_idxs)]
        features = pd.concat([features, features_duplicate])
        labels = np.concatenate([labels, labels])
        # TODO: Add this as a test-case that these features are the complete list that should be swapped.
        # They were manually checked with the full feature set
        # print('Swapping the following features:')
        # swapped_features = np.where(reflected_idxs!=np.arange(len(reflected_idxs)))[0]
        # for idx in swapped_features:
        #     print(str(lowercase_features[idx]) + ' -> ' + str(reflected_feature_names[idx]))
        return features, labels

    def set_classifier(self, classifier):
        """change the type of the classifier being used"""
        if classifier not in _classifier_choices:
            raise ValueError("Invalid Classifier Type")
        self._classifier_type = classifier

    def set_project_settings(self, project: Project):
        """assign project settings to the classifier

        Args:
            project: project to copy classifier-relevant settings from for the current behavior

        if no behavior is currently set will use project defaults
        """
        if self._behavior is None:
            self._behavior_settings = project.get_project_defaults()
        else:
            self._behavior_settings = project.settings_manager.get_behavior(self._behavior)

        # grab other JABS settings from settings manager, some might be used by the classifier
        self._jabs_settings = project.settings_manager.jabs_settings

    def set_behavior_settings(self, settings: dict):
        """assign behavior-specific settings via a dict to the classifier

        Args:
            settings: dict of project settings. Must be same structure as project.settings_manager.get_behavior

        TODO: Add checks to enforce conformity to project settings
        """
        self._behavior_settings = dict(settings)

    def classifier_choices(self):
        """get the available classifier types

        Returns:
            dict where keys are ClassifierType enum values, and the
        values are string names for the classifiers. example:

        {
            <ClassifierType.RANDOM_FOREST: 1>: 'Random Forest',
            <ClassifierType.GRADIENT_BOOSTING: 2>: 'Gradient Boosting',
            <ClassifierType.XGBOOST: 3>: 'XGBoost'
        }
        """
        return {d: self._CLASSIFIER_NAMES[d] for d in _classifier_choices}

    def train(self, data: dict, random_seed: int | None = None) -> None:
        """train the classifier

        Args:
            data: dict returned from train_test_split()
            random_seed: optional random seed (used when we want
                reproducible results between trainings)

        Returns:
            None

        raises ValueError for having either unset project settings or an unset classifier
        """
        if self._behavior_settings is None:
            raise ValueError("Project settings for classifier unset, cannot train classifier.")

        # Assume that feature names is provided, otherwise extract it from the dataframe
        if "feature_names" in data:
            self._feature_names = data["feature_names"]
        else:
            self._feature_names = data["training_data"].columns.to_list()

        # Obtain the feature and label matrices
        features = data["training_data"]
        labels = data["training_labels"]
        # Symmetric augmentation should occur before balancing so that the class with more labels can sample from the whole set
        if self._behavior_settings.get("symmetric_behavior", False):
            features, labels = self.augment_symmetric(features, labels)
        if self._behavior_settings.get("balance_labels", False):
            features, labels = self.downsample_balance(features, labels, random_seed)

        # Optional probability calibration
        if self.calibrate_probabilities:
            # get and validate calibration settings
            calibration_method = self._jabs_settings.get(
                "calibration_method", DEFAULT_CALIBRATION_METHOD
            )
            if calibration_method.lower() not in self.CALIBRATION_METHODS:
                raise ValueError(
                    f"Invalid calibration method: {calibration_method}. Must be one of {self.CALIBRATION_METHODS}"
                )
            calibration_cv = self._jabs_settings.get("calibration_cv", DEFAULT_CALIBRATION_CV)

            # Auto-select method if requested, always figure out what the auto method would be because some of the
            # selection info is still useful for warnings/logging purposes if the user specified a method explicitly
            auto_method, auto_method_info = self._choose_auto_calibration_method(
                labels, calibration_cv
            )
            if calibration_method.lower() == "auto":
                calibration_method = auto_method
            else:
                # Optional safety warning: isotonic with small per-fold sets can overfit
                if (
                    str(calibration_method).lower() == "isotonic"
                    and auto_method_info["per_fold_per_class"] < auto_method_info["threshold"]
                ):
                    warnings.warn(
                        (
                            "Isotonic calibration selected but per-fold per-class count appears small "
                            f"(~{auto_method_info['per_fold_per_class']}). Consider 'sigmoid' or lowering calibration_cv."
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )

            # Build an unfitted base estimator
            if self._classifier_type == ClassifierType.RANDOM_FOREST:
                base_estimator = self._make_random_forest(random_seed=random_seed)
            elif self._classifier_type == ClassifierType.GRADIENT_BOOSTING:
                base_estimator = self._make_gradient_boost(random_seed=random_seed)
            elif _xgboost is not None and self._classifier_type == ClassifierType.XGBOOST:
                base_estimator = self._make_xgboost(random_seed=random_seed)
            else:
                raise ValueError("Unsupported classifier")

            # Wrap with calibrated classifier and fit
            self._classifier = CalibratedClassifierCV(
                estimator=base_estimator, method=calibration_method, cv=calibration_cv
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                self._classifier.fit(self._clean_features_for_training(features), labels)
        else:
            # Fit without calibration (original behavior)
            if self._classifier_type == ClassifierType.RANDOM_FOREST:
                self._classifier = self._fit_random_forest(
                    features, labels, random_seed=random_seed
                )
            elif self._classifier_type == ClassifierType.GRADIENT_BOOSTING:
                self._classifier = self._fit_gradient_boost(
                    features, labels, random_seed=random_seed
                )
            elif _xgboost is not None and self._classifier_type == ClassifierType.XGBOOST:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    self._classifier = self._fit_xgboost(features, labels, random_seed=random_seed)
            else:
                raise ValueError("Unsupported classifier")

        # Classifier may have been re-used from a prior training, blank the logging attributes
        self._classifier_file = None
        self._classifier_hash = None
        self._classifier_source = None

    def _clean_features_for_training(self, features: pd.DataFrame):
        """Clean feature matrix prior to fitting based on classifier type.

        For XGBoost, only replace +/- inf with 0 (XGBoost can handle NaN).
        For sklearn tree models, also fill NaNs with 0.
        """
        if self._classifier_type == ClassifierType.XGBOOST:
            return features.replace([np.inf, -np.inf], 0)
        return features.replace([np.inf, -np.inf], 0).fillna(0)

    def _make_random_forest(self, random_seed: int | None = None):
        if random_seed is not None:
            return RandomForestClassifier(n_jobs=self._n_jobs, random_state=random_seed)
        return RandomForestClassifier(n_jobs=self._n_jobs)

    def _make_gradient_boost(self, random_seed: int | None = None):
        if random_seed is not None:
            return GradientBoostingClassifier(random_state=random_seed)
        return GradientBoostingClassifier()

    def _make_xgboost(self, random_seed: int | None = None):
        if random_seed is not None:
            return _xgboost.XGBClassifier(n_jobs=self._n_jobs, random_state=random_seed)
        return _xgboost.XGBClassifier(n_jobs=self._n_jobs)

    def sort_features_to_classify(self, features):
        """sorts features to match the current classifier"""
        if isinstance(self._classifier, CalibratedClassifierCV):
            # Use the training-time feature order we stored
            classifier_columns = self._feature_names
        elif self._classifier_type == ClassifierType.XGBOOST:
            classifier_columns = self._classifier.get_booster().feature_names
        else:
            classifier_columns = self._classifier.feature_names_in_
        features_sorted = features[classifier_columns]
        return features_sorted

    def predict(self, features):
        """predict classes for a given set of features"""
        if self._classifier_type == ClassifierType.XGBOOST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                result = self._classifier.predict(
                    self.sort_features_to_classify(features.replace([np.inf, -np.inf], 0))
                )
            return result
        # Random forests and gradient boost can't handle NAs & infs, so fill them with 0s
        return self._classifier.predict(
            self.sort_features_to_classify(features.replace([np.inf, -np.inf], 0).fillna(0))
        )

    def predict_proba(self, features):
        """predict probabilities for a given set of features"""
        if self._classifier_type == ClassifierType.XGBOOST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                result = self._classifier.predict_proba(self.sort_features_to_classify(features))
            return result
        # Random forests and gradient boost can't handle NAs & infs, so fill them with 0s
        return self._classifier.predict_proba(
            self.sort_features_to_classify(features.replace([np.inf, -np.inf], 0).fillna(0))
        )

    def save(self, path: Path):
        """save the classifier to a file

        Uses joblib to serialize the classifier object to a file.
        """
        joblib.dump(self, path)

        # If the classifier was not generated from exported training data
        # we can hash the serialized classifier.
        # Note that this hash changes every time the "train" button is
        # pressed, regardless of whether the training data changes.
        if self._classifier_file is None:
            self._classifier_file = Path(path).name
            self._classifier_hash = hash_file(Path(path))
            self._classifier_source = "serialized"

    def load(self, path: Path):
        """load a classifier from a file

        Uses joblib to deserialize the classifier object that was previously saved
        using the joblib.dump() method.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            c = joblib.load(path)
            for warning in caught_warnings:
                if issubclass(warning.category, InconsistentVersionWarning):
                    raise ValueError("Classifier trained with different version of sklearn.")
                else:
                    warnings.warn(warning.message, warning.category, stacklevel=2)

        if not isinstance(c, Classifier):
            raise ValueError(f"{path} is not instance of Classifier")

        if c.version != _VERSION:
            raise ValueError(
                f"Error deserializing classifier. File version {c.version}, expected {_VERSION}."
            )

            # make sure the value passed for the classifier parameter is valid
        if c._classifier_type not in _classifier_choices:
            raise ValueError("Invalid classifier type")

        self._classifier = c._classifier
        self._behavior = c._behavior
        self._behavior_settings = c._behavior_settings
        self._jabs_settings = c._jabs_settings
        self._classifier_type = c._classifier_type
        if c._classifier_file is not None:
            self._classifier_file = c._classifier_file
            self._classifier_hash = c._classifier_hash
            self._classifier_source = c._classifier_source
        else:
            self._classifier_file = Path(path).name
            self._classifier_hash = hash_file(Path(path))
            self._classifier_source = "pickle"

    def _update_classifier_type(self):
        # we may need to update the classifier type based
        # on the type of the loaded object
        if isinstance(self._classifier, RandomForestClassifier):
            self._classifier_type = ClassifierType.RANDOM_FOREST
        elif isinstance(self._classifier, GradientBoostingClassifier):
            self._classifier_type = ClassifierType.GRADIENT_BOOSTING
        else:
            self._classifier_type = ClassifierType.XGBOOST

    @staticmethod
    def accuracy_score(truth, predictions):
        """return accuracy score"""
        return accuracy_score(truth, predictions)

    @staticmethod
    def precision_recall_score(truth, predictions):
        """return precision recall score"""
        return precision_recall_fscore_support(truth, predictions)

    @staticmethod
    def confusion_matrix(truth, predictions):
        """return the confusion matrix using sklearn's confusion_matrix function"""
        return confusion_matrix(truth, predictions)

    @staticmethod
    def brier_score(truth: np.ndarray, probabilities: np.ndarray) -> float:
        """Return the Brier score (lower is better).

        Args:
            truth (ndarray): array of true binary labels (0/1).
            probabilities (ndarray): array of predicted probabilities for the positive class; can be shape (n_samples,)
                   or a (n_samples, 2) array from `predict_proba`.

        Returns:
            float Brier score.
        """
        if probabilities.ndim == 2:
            # assume columns [P(neg), P(pos)] as returned by predict_proba
            probabilities = probabilities[:, 1]
        return brier_score_loss(truth, probabilities)

    @staticmethod
    def plot_reliability(
        truth: np.ndarray,
        probabilities: np.ndarray,
        out_path: Path | str,
        n_bins: int = 10,
        strategy: str = "uniform",
        title: str | None = None,
        show_hist: bool = True,
    ) -> dict:
        """Create and save a reliability (calibration) plot.

        Args:
            truth: Binary ground truth labels (0 or 1).
            probabilities: Predicted probabilities (2D array where second column is positive class).
            out_path: File path to save the reliability plot.
            n_bins: Number of bins for calibration curve.
            strategy: Binning strategy ('uniform' or 'quantile').
            title: Optional plot title.
            show_hist: If True, adds a histogram of predicted probabilities below the curve.

        Returns:
            Dict with calibration data: 'bins', 'mean_pred', 'frac_pos', and 'counts'.
        """
        prob = probabilities[:, 1]
        y = np.asarray(truth).astype(int)

        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        if pos == 0 or neg == 0:
            warnings.warn(
                "plot_reliability: need both positive and negative labels.", stacklevel=2
            )

        # Compute calibration curve
        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=n_bins, strategy=strategy)

        # Bin edges and counts
        if strategy == "uniform":
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            counts, _ = np.histogram(prob, bins=bins)
        else:
            q = np.linspace(0.0, 1.0, n_bins + 1)
            bins = np.quantile(prob, q)
            bins = np.unique(bins)
            counts, _ = np.histogram(prob, bins=bins)

        # Plot
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, marker="o", color="C0", label="Model")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical frequency")
        if title:
            ax.set_title(title)
        ax.legend(loc="best")

        if show_hist:
            ax_hist = ax.twinx()
            ax_hist.set_ylim(0, max(counts) * 1.2 if counts.size else 1)
            display_bins = bins if strategy == "uniform" else 10
            ax_hist.hist(prob, bins=display_bins, alpha=0.25, color="C1")
            ax_hist.set_yticks([])

        fig.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return {
            "bins": bins,
            "mean_pred": mean_pred,
            "frac_pos": frac_pos,
            "counts": counts,
        }

    @staticmethod
    def combine_data(per_frame, window):
        """combine feature sets together

        Args:
            per_frame: per frame features dataframe
            window: window feature dataframe

        Returns:
            merged dataframe
        """
        return pd.concat([per_frame, window], axis=1)

    def _fit_random_forest(self, features, labels, random_seed: int | None = None):
        classifier = RandomForestClassifier(n_jobs=self._n_jobs, random_state=random_seed)
        return classifier.fit(features.replace([np.inf, -np.inf], 0).fillna(0), labels)

    def _fit_gradient_boost(self, features, labels, random_seed: int | None = None):
        classifier = GradientBoostingClassifier(random_state=random_seed)
        return classifier.fit(features.replace([np.inf, -np.inf], 0).fillna(0), labels)

    def _fit_xgboost(self, features, labels, random_seed: int | None = None):
        classifier = _xgboost.XGBClassifier(n_jobs=self._n_jobs, random_state=random_seed)
        classifier.fit(features.replace([np.inf, -np.inf]), labels)
        return classifier

    def _get_estimator_with_feature_importances(self):
        """Return the underlying estimator that exposes `feature_importances_`, if available.

        Handles calibrated classifiers by retrieving the estimator from the first
        calibrated fold. Returns None if no estimator with `feature_importances_` is found.
        """
        est = self._classifier
        # If wrapped by CalibratedClassifierCV, peel off the estimator
        if isinstance(est, CalibratedClassifierCV):
            try:
                cc0 = est.calibrated_classifiers_[0]
                est = cc0.estimator
            except Exception:
                return None
        # Some sklearn/xgboost estimators expose feature_importances_
        return est if hasattr(est, "feature_importances_") else None

    def get_calibrated_feature_importances(self):
        """Return averaged feature importances across calibrated folds.

        For CalibratedClassifierCV with tree-based base estimators (RF/GBT/XGBoost),
        this computes the mean and std of `feature_importances_` across
        `calibrated_classifiers_` estimators and returns a list of tuples:
        [(feature_name, mean_importance, std_importance), ...] sorted by mean desc.

        Returns None if unavailable (e.g., non-tree base estimators).
        """
        if not isinstance(self._classifier, CalibratedClassifierCV):
            return None
        try:
            base_ests = [cc.estimator for cc in self._classifier.calibrated_classifiers_]
        except Exception:
            return None

        # get the base estimators that have feature_importances_
        base_ests = [be for be in base_ests if hasattr(be, "feature_importances_")]
        if not base_ests:
            return None

        # get the mean and standard deviation of feature importances from the base estimators
        importances = np.vstack([be.feature_importances_ for be in base_ests])
        mean_imp = importances.mean(axis=0)
        std_imp = importances.std(axis=0)

        # combine with feature names and sort by mean importance
        items = list(zip(self._feature_names, mean_imp, std_imp, strict=True))
        items.sort(key=lambda t: t[1], reverse=True)
        return items

    def print_feature_importance(self, feature_list, limit=20):
        """print the most important features and their importance

        Args:
            feature_list: list of feature names used in the classifier
            limit: maximum number of features to print, defaults to 20
        """
        # Prefer calibrated importances if available
        if isinstance(self._classifier, CalibratedClassifierCV):
            items = self.get_calibrated_feature_importances()
            if items is not None:
                print(f"{'Feature Name':100} Mean Importance   Std")
                print("-" * 120)
                for name, mean_imp, std_imp in items[:limit]:
                    print(f"{name:100} {mean_imp:0.4f}         {std_imp:0.4f}")
                return
            # fall through to base-estimator single-source path if calibrated but no importances

        # Fallback: single estimator feature_importances_
        est = self._get_estimator_with_feature_importances()
        if est is None:
            print("Feature importances are unavailable for the current classifier.")
            return
        importances = list(est.feature_importances_)
        names = feature_list if feature_list is not None else (self._feature_names or [])
        if len(importances) != len(names):
            names = [f"feature_{i}" for i in range(len(importances))]
        feature_importance = [
            (feature, round(importance, 4))
            for feature, importance in zip(names, importances, strict=False)
        ]
        feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        print(f"{'Feature Name':100} Importance")
        print("-" * 120)
        for feature, importance in feature_importance[:limit]:
            print(f"{feature:100} {importance:0.4f}")

    @staticmethod
    def count_label_threshold(all_counts: dict):
        """counts the number of groups that meet label threshold criteria

        Args:
            all_counts: labeled frame and bout counts for the entire
                project


            all_counts is a dict with the following form
            {
                '<video name>': {
                    <identity>: {
                        "fragmented_frame_counts": (
                            behavior frame count: fragmented,
                            not behavior frame count: fragmented),
                        "fragmented_bout_counts": (
                            behavior bout count: fragmented,
                            not behavior bout count: fragmented
                        ),
                        "unfragmented_frame_counts": (
                            behavior frame count: unfragmented,
                            not behavior frame count: unfragmented
                        ),
                        "unfragmented_bout_counts": (
                            behavior bout count: unfragmented,
                            not behavior bout count: unfragmented
                        ),
                    },
                }
            }

        Returns:
            number of groups that meet label criteria

        Note: uses "fragmented" label counts, since these reflect the counts of labels that are usable for training
        """
        group_count = 0
        for video in all_counts:
            for identity_count in all_counts[video].values():
                if (
                    identity_count["fragmented_frame_counts"][0] >= Classifier.LABEL_THRESHOLD
                    and identity_count["fragmented_frame_counts"][1] >= Classifier.LABEL_THRESHOLD
                ):
                    group_count += 1
        return group_count

    @staticmethod
    def label_threshold_met(all_counts: dict, min_groups: int):
        """determine if the labeling threshold is met

        Args:
            all_counts: labeled frame and bout counts for the entire
                project
            min_groups: minimum number of groups required (more than one
                group is always required for the "leave one group out" train/test split,
                but may be more than 2 for k-fold cross validation if k > 2)

        Returns:
            bool if requested valid groups is > valid group
        """
        group_count = Classifier.count_label_threshold(all_counts)
        return 1 < group_count >= min_groups
