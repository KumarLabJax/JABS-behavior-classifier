import pickle
import random
import typing
from enum import IntEnum
from importlib import import_module
from pathlib import Path

import numpy as np
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


class Classifier:
    TRAINING_FILE_VERSION = 1
    LABEL_THRESHOLD = 20

    class ClassifierType(IntEnum):
        RANDOM_FOREST = 1
        GRADIENT_BOOSTING = 2
        XGBOOST = 3

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
        self._n_jobs = n_jobs

        self._classifier_choices = [
            self.ClassifierType.RANDOM_FOREST,
            self.ClassifierType.GRADIENT_BOOSTING
        ]

        try:
            self._xgboost = import_module("xgboost")
            # we were able to import xgboost, make it available as an option:
            self._classifier_choices.append(self.ClassifierType.XGBOOST)
        except Exception:
            # we were unable to import the xgboost module. It's either not
            # installed (it should be if the user used our requirements.txt)
            # or it may have been unable to be imported due to a missing
            # libomp. Either way, we won't add it to the available choices and
            # we can otherwise ignore this exception
            self._xgboost = None

        # make sure the value passed for the classifier parameter is valid
        if classifier not in self._classifier_choices:
            raise ValueError("Invalid classifier type")

    @property
    def classifier_name(self):
        """ return the name of the classifier used as a string """
        return self._classifier_names[self._classifier_type]

    @property
    def classifier_type(self):
        """ return classifier type (Classifier.ClassifierType enum value) """
        return self._classifier_type

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
        }
        """
        datasets = []

        # add per frame features to our data set
        for feature in sorted(per_frame_features):
            datasets.append(per_frame_features[feature])

        # add window features to our data set
        for feature in sorted(window_features):
            if feature == 'percent_frames_present':
                datasets.append(window_features[feature])
            else:
                # [source_feature_name][operator_applied] : numpy array
                # iterate over operator names
                for op in sorted(window_features[feature]):
                    # append the numpy array to the dataset
                    datasets.append(window_features[feature][op])

        # split labeled data and labels
        split_data = train_test_split(np.concatenate(datasets, axis=1),
                                      label_data)

        return {
            'test_labels': split_data.pop(),
            'training_labels': split_data.pop(),
            'training_data': split_data[::2],
            'test_data': split_data[1::2]
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
            not_behavior_count = np.count_nonzero(labels[split[1]] == TrackLabels.Label.NOT_BEHAVIOR)

            if (behavior_count >= Classifier.LABEL_THRESHOLD and
                    not_behavior_count >= Classifier.LABEL_THRESHOLD):
                count += 1
                yield {
                    'training_labels': labels[split[0]],
                    'training_data': x[split[0]],
                    'test_labels': labels[split[1]],
                    'test_data': x[split[1]],
                    'test_group': groups[split[1]][0]
                }

        # number of splits exhausted without finding at least one that meets
        # criteria
        # the UI won't allow us to reach this case
        if count == 0:
            raise ValueError("unable to split data")

    def set_classifier(self, classifier):
        """ change the type of the classifier being used """
        if classifier not in self._classifier_choices:
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
            d: self._classifier_names[d] for d in self._classifier_choices
        }

    def train(self, data, random_seed: typing.Optional[int]=None):
        """
        train the classifier
        :param data: dict returned from train_test_split()
        :param random_seed: optional random seed (used when we want reproducible
        results between trainings)
        :return: None
        """

        features = data['training_data']
        labels = data['training_labels']

        if self._classifier_type == self.ClassifierType.RANDOM_FOREST:
            self._classifier = self._fit_random_forest(features, labels,
                                                       random_seed=random_seed)
        elif self._classifier_type == self.ClassifierType.GRADIENT_BOOSTING:
            self._classifier = self._fit_gradient_boost(features, labels,
                                                        random_seed=random_seed)
        elif self._xgboost is not None and self._classifier_type == self.ClassifierType.XGBOOST:
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

    def load_classifier(self, path: Path):
        with path.open('rb') as f:
            self._classifier = pickle.load(f)

            # we may need to update the classifier type based on
            # on the type of the loaded object
            if isinstance(self._classifier, RandomForestClassifier):
                self._classifier_type = self.ClassifierType.RANDOM_FOREST
            elif isinstance(self._classifier, GradientBoostingClassifier):
                self._classifier_type = self.ClassifierType.GRADIENT_BOOSTING
            else:
                self._classifier_type = self.ClassifierType.XGBOOST

    def save_classifier(self, path: Path):
        with path.open('wb') as f:
            pickle.dump(self._classifier, f)

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
        iterate over feature sets and combine them to create a dataset with the
        shape #frames, #features
        :param per_frame: per frame features dictionary
        :param window: window feature dictionary
        :return: numpy array with shape #frames,#features
        """
        datasets = []
        # add per frame features to our data set
        # sort the feature names in the dict so the order is consistent
        for feature in sorted(per_frame):
            datasets.append(per_frame[feature])

        # add window features to our data set
        # sort the feature names in the dict so the order is consistent
        for feature in sorted(window):
            if isinstance(window[feature], dict):
                # [source_feature_name][operator_applied] : numpy array
                # iterate over operator names
                for op in sorted(window[feature]):
                    # append the numpy array to the dataset
                    datasets.append(window[feature][op])
            else:
                datasets.append(window[feature])

        # expand any 1D features to 2D so that we can concatenate in one call
        datasets = [(d[:, np.newaxis] if d.ndim == 1 else d) for d in datasets]
        return np.concatenate(datasets, axis=1)

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
            classifier = GradientBoostingClassifier(n_jobs=self._n_jobs,
                                                    random_state=random_seed)
        else:
            classifier = GradientBoostingClassifier(n_jobs=self._n_jobs)
        return classifier.fit(features, labels)

    def _fit_xgboost(self, features, labels,
                     random_seed: typing.Optional[int] = None):
        if random_seed is not None:
            classifier = self._xgboost.XGBClassifier(n_jobs=self._n_jobs,
                                                     random_state=random_seed)
        else:
            classifier = self._xgboost.XGBClassifier(n_jobs=self._n_jobs)
        classifier.fit(features, labels)
        return classifier

    def print_feature_importance(self, feature_list, limit=20):
        """
        print the most important features and their importance
        :param feature_list:
        :param limit:
        :return:
        """
        # Get numerical feature importances
        importances = list(self._classifier.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for
                               feature, importance in
                               zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                     reverse=True)
        # Print out the feature and importance
        print(f"{'Feature Name':55} Importance")
        print('-' * 70)
        for feature, importance in feature_importances[:limit]:
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
