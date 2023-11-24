import random
import typing
from enum import IntEnum
from importlib import import_module
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, LeaveOneGroupOut

from src.project import TrackLabels
from src.project import ProjectDistanceUnit

_VERSION = 4


class ClassifierType(IntEnum):
    RANDOM_FOREST = 1
    GRADIENT_BOOSTING = 2
    XGBOOST = 3


_classifier_choices = [
    ClassifierType.RANDOM_FOREST,
    ClassifierType.GRADIENT_BOOSTING
]

try:
    _xgboost = import_module("xgboost")
    # we were able to import xgboost, make it available as an option:
    _classifier_choices.append(ClassifierType.XGBOOST)
except Exception:
    # we were unable to import the xgboost module. It's either not
    # installed (it should be if the user used our requirements.txt)
    # or it may have been unable to be imported due to a missing
    # libomp. Either way, we won't add it to the available choices and
    # we can otherwise ignore this exception
    _xgboost = None


class Classifier:
    LABEL_THRESHOLD = 20

    _classifier_names = {
        ClassifierType.RANDOM_FOREST: "Random Forest",
        ClassifierType.GRADIENT_BOOSTING: "Gradient Boosting",
        ClassifierType.XGBOOST: "XGBoost"
    }

    def __init__(self, classifier=ClassifierType.RANDOM_FOREST, n_jobs=1):
        """
        :param classifier: type of classifier to use. Must be ClassifierType
        :param n_jobs: number of jobs to use for classifiers that support
        this parameter for parallelism
        enum value. Defaults to ClassifierType.RANDOM_FOREST
        """

        self._classifier_type = classifier
        self._classifier = None
        self._window_size = None
        self._uses_social = None
        self._extended_features = None
        self._behavior = None
        self._distance_unit = None
        self._n_jobs = n_jobs
        self._version = _VERSION

        # make sure the value passed for the classifier parameter is valid
        if classifier not in _classifier_choices:
            raise ValueError("Invalid classifier type")

    @property
    def classifier_name(self) -> str:
        """ return the name of the classifier used as a string """
        return self._classifier_names[self._classifier_type]

    @property
    def classifier_type(self) -> ClassifierType:
        """ return classifier type """
        return self._classifier_type

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def uses_social(self) -> bool:
        return self._uses_social

    @property
    def extended_features(self) -> typing.Dict[str, typing.List[str]]:
        return self._extended_features

    @property
    def behavior_name(self) -> str:
        return self._behavior

    @property
    def version(self) -> int:
        return self._version

    @property
    def distance_unit(self) -> ProjectDistanceUnit:
        """
        return the distance unit for the features that were used to train
        this classifier
        """
        return self._distance_unit

    @staticmethod
    def train_test_split(per_frame_features, window_features, label_data):
        """
        split features and labels into training and test datasets

        :param per_frame_features: per frame features as returned from
        IdentityFeatures object, filtered to only include labeled frames
        :param window_features: window features as returned from
        IdentityFeatures object, filtered to only include labeled frames
        :param label_data: labels that correspond to the features
        :return: dictionary of training and test data and labels:

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
        split_data = train_test_split(all_features, label_data)

        return {
            'test_labels': split_data.pop(),
            'training_labels': split_data.pop(),
            'training_data': split_data[::2],
            'test_data': split_data[1::2],
            'feature_names': all_features.columns.to_list()
        }

    @staticmethod
    def leave_one_group_out(per_frame_features, window_features, labels,
                            groups):
        """
        implements "leave one group out" data splitting strategy
        :param per_frame_features: per frame features for all labeled data
        :param window_features: window features for all labeled data
        :param labels: labels corresponding to each feature row
        :param groups: group id corresponding to each feature row
        :return: dictionary of training and test data and labels:
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

            behavior_count = np.count_nonzero(
                labels[split[1]] == TrackLabels.Label.BEHAVIOR)
            not_behavior_count = np.count_nonzero(
                labels[split[1]] == TrackLabels.Label.NOT_BEHAVIOR)

            if (behavior_count >= Classifier.LABEL_THRESHOLD and
                    not_behavior_count >= Classifier.LABEL_THRESHOLD):
                count += 1
                yield {
                    'training_labels': labels[split[0]],
                    'training_data': x.iloc[split[0]],
                    'test_labels': labels[split[1]],
                    'test_data': x.iloc[split[1]],
                    'test_group': groups[split[1]][0],
                    'feature_names': x.columns.to_list()
                }

        # number of splits exhausted without finding at least one that meets
        # criteria
        # the UI won't allow us to reach this case
        if count == 0:
            raise ValueError("unable to split data")

    def set_classifier(self, classifier):
        """ change the type of the classifier being used """
        if classifier not in _classifier_choices:
            raise ValueError("Invalid Classifier Type")
        self._classifier_type = classifier

    def classifier_choices(self):
        """
        get the available classifier types
        :return: dict where keys are ClassifierType enum values, and the
        values are string names for the classifiers. example:

        {
            <ClassifierType.RANDOM_FOREST: 1>: 'Random Forest',
            <ClassifierType.GRADIENT_BOOSTING: 2>: 'Gradient Boosting',
            <ClassifierType.XGBOOST: 3>: 'XGBoost'
        }
        """
        return {
            d: self._classifier_names[d] for d in _classifier_choices
        }

    def train(self, data, behavior: str, window_size: int, uses_social: bool,
              extended_features: typing.Dict,
              distance_unit: ProjectDistanceUnit,
              random_seed: typing.Optional[int] = None):
        """
        train the classifier
        :param data: dict returned from train_test_split()
        :param behavior: string name of behavior we are training for
        :param window_size: window size used for training
        :param uses_social: does training data include social features?
        :param extended_features: additional features used by classifier
        :param distance_unit: the distance unit used for training
        :param random_seed: optional random seed (used when we want reproducible
        results between trainings)
        :return: None

        NOTE: window_size, uses_social, extended_features, and distance_unit
        is used only to verify that a trained classifer can be used
        (check the classifier doesn't use features that are not supported the
        project)
        """
        features = data['training_data']
        labels = data['training_labels']

        self._uses_social = uses_social
        self._window_size = window_size
        self._behavior = behavior
        self._distance_unit = distance_unit
        self._extended_features = extended_features

        if self._classifier_type == ClassifierType.RANDOM_FOREST:
            self._classifier = self._fit_random_forest(features, labels,
                                                       random_seed=random_seed)
        elif self._classifier_type == ClassifierType.GRADIENT_BOOSTING:
            self._classifier = self._fit_gradient_boost(features, labels,
                                                        random_seed=random_seed)
        elif _xgboost is not None and self._classifier_type == ClassifierType.XGBOOST:
            self._classifier = self._fit_xgboost(features, labels,
                                                 random_seed=random_seed)
        else:
            raise ValueError("Unsupported classifier")

    def predict(self, features):
        """
        predict classes for a given set of features
        """
        return self._classifier.predict(features)

    def predict_proba(self, features):
        return self._classifier.predict_proba(features)

    def save(self, path: Path):
        joblib.dump(self, path)

    def load(self, path: Path):
        c = joblib.load(path)

        if not isinstance(c, Classifier):
            raise ValueError(
                f"{path} is not instance of Classifier")

        if c.version != _VERSION:
            raise ValueError(f"Error deserializing classifier. "
                             f"File version {c.version}, expected {_VERSION}.")

            # make sure the value passed for the classifier parameter is valid
        if c._classifier_type not in _classifier_choices:
            raise ValueError("Invalid classifier type")

        self._classifier = c._classifier
        self._behavior = c._behavior
        self._window_size = c._window_size
        self._uses_social = c._uses_social
        self._classifier_type = c._classifier_type
        self._distance_unit = c._distance_unit

    def _update_classifier_type(self):
        # we may need to update the classifier type based on
        # on the type of the loaded object
        if isinstance(self._classifier, RandomForestClassifier):
            self._classifier_type = ClassifierType.RANDOM_FOREST
        elif isinstance(self._classifier, GradientBoostingClassifier):
            self._classifier_type = ClassifierType.GRADIENT_BOOSTING
        else:
            self._classifier_type = ClassifierType.XGBOOST

    @staticmethod
    def accuracy_score(truth, predictions):
        return accuracy_score(truth, predictions)

    @staticmethod
    def precision_recall_score(truth, predictions):
        return precision_recall_fscore_support(truth, predictions)

    @staticmethod
    def confusion_matrix(truth, predictions):
        return confusion_matrix(truth, predictions)

    @staticmethod
    def combine_data(per_frame, window):
        """
        combine feature sets together
        :param per_frame: per frame features dataframe
        :param window: window feature dataframe
        :return: merged dataframe
        """
        return pd.concat([per_frame, window], axis=1)

    def _fit_random_forest(self, features, labels,
                           random_seed: typing.Optional[int] = None):
        if random_seed is not None:
            classifier = RandomForestClassifier(n_jobs=self._n_jobs,
                                                random_state=random_seed)
        else:
            classifier = RandomForestClassifier(n_jobs=self._n_jobs)
        return classifier.fit(features, labels)

    def _fit_gradient_boost(self, features, labels,
                            random_seed: typing.Optional[int] = None):
        if random_seed is not None:
            classifier = GradientBoostingClassifier(random_state=random_seed)
        else:
            classifier = GradientBoostingClassifier()
        return classifier.fit(features, labels)

    def _fit_xgboost(self, features, labels,
                     random_seed: typing.Optional[int] = None):
        if random_seed is not None:
            classifier = _xgboost.XGBClassifier(n_jobs=self._n_jobs,
                                                random_state=random_seed)
        else:
            classifier = _xgboost.XGBClassifier(n_jobs=self._n_jobs)
        classifier.fit(features, labels)
        return classifier

    def print_feature_importance(self, feature_list, limit=20):
        """
        print the most important features and their importance
        :param feature_list:
        :param limit:
        :return:
        """
        # Get numerical feature importance
        importances = list(self._classifier.feature_importances_)
        # List of tuples with variable and importance
        feature_importance = [(feature, round(importance, 2)) for
                              feature, importance in
                              zip(feature_list, importances)]
        # Sort the feature importance by most important first
        feature_importance = sorted(feature_importance, key=lambda x: x[1],
                                    reverse=True)
        # Print out the feature and importance
        print(f"{'Feature Name':55} Importance")
        print('-' * 70)
        for feature, importance in feature_importance[:limit]:
            print(f"{feature:55} {importance:0.2f}")

    @staticmethod
    def label_threshold_met(all_counts: dict, min_groups: int):
        """
        determine if the labeling threshold is met
        :param all_counts: labeled frame and bout counts for the entire project
        parameter is a dict with the following form
        {
            '<video name>': [
                (
                    <identity>,
                    (behavior frame count, not behavior frame count),
                    (behavior bout count, not behavior bout count)
                ),
            ]
        }

        :param min_groups: minimum number of groups required (more than one
        group is always required for the "leave one group out" train/test split,
        but may be more than 2 for k-fold cross validation if k > 2)

        """
        group_count = 0
        for video, counts in all_counts.items():
            for count in counts:
                if (count[1][0] >= Classifier.LABEL_THRESHOLD and
                        count[1][1] >= Classifier.LABEL_THRESHOLD):
                    group_count += 1

        return True if 1 < group_count >= min_groups else False
