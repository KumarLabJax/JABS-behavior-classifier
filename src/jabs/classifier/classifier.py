import random
import re
import typing
import warnings
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from jabs.project import Project, TrackLabels, load_training_data
from jabs.types import ClassifierType
from jabs.utils import hash_file

_VERSION = 9

_classifier_choices = [ClassifierType.RANDOM_FOREST, ClassifierType.GRADIENT_BOOSTING]

try:
    _xgboost = import_module("xgboost")
    # we were able to import xgboost, make it available as an option:
    _classifier_choices.append(ClassifierType.XGBOOST)
except Exception:
    # we were unable to import the xgboost module. It's either not
    # installed (it should be if the user used our requirements-old.txt)
    # or it may have been unable to be imported due to a missing
    # libomp. Either way, we won't add it to the available choices and
    # we can otherwise ignore this exception
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

    _CLASSIFIER_NAMES: typing.ClassVar[dict] = {
        ClassifierType.RANDOM_FOREST: "Random Forest",
        ClassifierType.GRADIENT_BOOSTING: "Gradient Boosting",
        ClassifierType.XGBOOST: "XGBoost",
    }

    def __init__(self, classifier=ClassifierType.RANDOM_FOREST, n_jobs=1):
        self._classifier_type = classifier
        self._classifier = None
        self._project_settings = None
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
        classifier.set_dict_settings(loaded_training_data["settings"])
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
    def project_settings(self) -> dict:
        """return a copy of dictionary of project settings for this classifier"""
        if self._project_settings is not None:
            return dict(self._project_settings)
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
            self._project_settings = project.get_project_defaults()
        else:
            self._project_settings = project.settings_manager.get_behavior(self._behavior)

    def set_dict_settings(self, settings: dict):
        """assign project settings via a dict to the classifier

        Args:
            settings: dict of project settings. Must be same structure as project.settings_manager.get_behavior

        TODO: Add checks to enforce conformity to project settings
        """
        self._project_settings = dict(settings)

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

    def train(self, data, random_seed: int | None = None):
        """train the classifier

        Args:
            data: dict returned from train_test_split()
            random_seed: optional random seed (used when we want
                reproducible results between trainings)

        Returns:
            None

        raises ValueError for having either unset project settings or an unset classifier
        """
        if self._project_settings is None:
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
        if self._project_settings.get("symmetric_behavior", False):
            features, labels = self.augment_symmetric(features, labels)
        if self._project_settings.get("balance_labels", False):
            features, labels = self.downsample_balance(features, labels, random_seed)

        if self._classifier_type == ClassifierType.RANDOM_FOREST:
            self._classifier = self._fit_random_forest(features, labels, random_seed=random_seed)
        elif self._classifier_type == ClassifierType.GRADIENT_BOOSTING:
            self._classifier = self._fit_gradient_boost(features, labels, random_seed=random_seed)
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

    def sort_features_to_classify(self, features):
        """sorts features to match the current classifier"""
        if self._classifier_type == ClassifierType.XGBOOST:
            classifier_columns = self._classifier.get_booster().feature_names
        # sklearn places feature names in the same spot
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
        self._project_settings = c._project_settings
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
        if random_seed is not None:
            classifier = RandomForestClassifier(n_jobs=self._n_jobs, random_state=random_seed)
        else:
            classifier = RandomForestClassifier(n_jobs=self._n_jobs)
        return classifier.fit(features.replace([np.inf, -np.inf], 0).fillna(0), labels)

    def _fit_gradient_boost(self, features, labels, random_seed: int | None = None):
        if random_seed is not None:
            classifier = GradientBoostingClassifier(random_state=random_seed)
        else:
            classifier = GradientBoostingClassifier()
        return classifier.fit(features.replace([np.inf, -np.inf], 0).fillna(0), labels)

    def _fit_xgboost(self, features, labels, random_seed: int | None = None):
        if random_seed is not None:
            classifier = _xgboost.XGBClassifier(n_jobs=self._n_jobs, random_state=random_seed)
        else:
            classifier = _xgboost.XGBClassifier(n_jobs=self._n_jobs)
        classifier.fit(features.replace([np.inf, -np.inf]), labels)
        return classifier

    def print_feature_importance(self, feature_list, limit=20):
        """print the most important features and their importance

        Args:
            feature_list: list of feature names used in the classifier
            limit: maximum number of features to print, defaults to 20
        """
        # Get numerical feature importance
        importances = list(self._classifier.feature_importances_)
        # List of tuples with variable and importance
        feature_importance = [
            (feature, round(importance, 2))
            for feature, importance in zip(feature_list, importances, strict=True)
        ]
        # Sort the feature importance by most important first
        feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        # Print out the feature and importance
        print(f"{'Feature Name':100} Importance")
        print("-" * 120)
        for feature, importance in feature_importance[:limit]:
            print(f"{feature:100} {importance:0.2f}")

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
