import random
import typing
from enum import IntEnum
from importlib import import_module
from pathlib import Path
import joblib
import re

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
from src.project import ProjectDistanceUnit

_VERSION = 3


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
        self._uses_balance = None
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
    def uses_balance(self) -> bool:
        return self._uses_balance

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

            behavior_count = np.count_nonzero(
                labels[split[1]] == TrackLabels.Label.BEHAVIOR)
            not_behavior_count = np.count_nonzero(
                labels[split[1]] == TrackLabels.Label.NOT_BEHAVIOR)

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

    @staticmethod
    def downsample_balance(features, labels, random_seed=None):
        """
        downsamples features and labels such that labels are equally distributed
        :param features: features to downsample
        :param labels: labels to downsample
        :return: tuple of downsampled features, labels
        """
        label_states, label_counts = np.unique(labels, return_counts=True)
        max_examples_per_class = np.min(label_counts)
        selected_samples = []
        for cur_label in label_states:
            idxs = np.where(labels==cur_label)[0]
            if random_seed is not None:
                np.random.seed(random_seed)
            sampled_idxs = np.random.choice(idxs, max_examples_per_class, replace=False)
            selected_samples.append(sampled_idxs)
        selected_samples = np.sort(np.concatenate(selected_samples))
        features = features[selected_samples,:]
        labels = labels[selected_samples]
        return features, labels

    @staticmethod
    def augment_symmetric(features, labels, feature_names, random_str='ASygRQDZJD'):
        """
        augments the features to include L-R and R-L duplicates
        This requires 'left' or 'right' to be in the feature name to be swapped
        Features that don't include these terms will not be swapped
        :param features: features to augment
        :param labels: labels to augment
        :param feature_names: feature names to detect LR exchanges
        :param random_str: a random string to use as a temporary replacement when swapping left/right
        :return: tuple of augmented features, labels
        """
        assert len(feature_names)==np.shape(features)[1]
        # Figure out the L-R swapping of features
        lowercase_features = np.array([x.lower() for x in feature_names])
        reflected_feature_names = [re.sub(r'left', random_str, x) for x in lowercase_features]
        reflected_feature_names = [re.sub(r'right', 'left', x) for x in reflected_feature_names]
        reflected_feature_names = [re.sub(random_str, 'right', x) for x in reflected_feature_names]
        reflected_idxs = [np.where(lowercase_features==x)[0][0] if x in lowercase_features else i for i,x in enumerate(reflected_feature_names)]
        # expand the features with reflections
        features = np.concatenate([features, features[:,reflected_idxs]])
        labels = np.concatenate([labels, labels])
        # TODO: Add this as a test-case that these features are the complete list that should be swapped.
        # They were manually checked with the full feature set
        # print('Swapping the following features:')
        # swapped_features = np.where(reflected_idxs!=np.arange(len(reflected_idxs)))[0]
        # for idx in swapped_features:
        #     print(str(lowercase_features[idx]) + ' -> ' + str(reflected_feature_names[idx]))
        return features, labels

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

    def train(self, data, feature_names, behavior: str, window_size: int, uses_social: bool,
              uses_balance: bool,
              extended_features: typing.Dict,
              distance_unit: ProjectDistanceUnit,
              random_seed: typing.Optional[int] = None):
        """
        train the classifier
        :param data: dict returned from train_test_split()
        :param feature_names: a list of feature names
        :param behavior: string name of behavior we are training for
        :param window_size: window size used for training
        :param uses_social: does training data include social features?
        :param uses_balance: does the training balance labels through downsampling before training?
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
        features, labels = self.augment_symmetric(features, labels, feature_names)
        if uses_balance:
            features, labels = self.downsample_balance(features, labels, random_seed)


        self._uses_social = uses_social
        self._uses_balance = uses_balance
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
        self._uses_balance = c._uses_balance
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
            # [source_feature_name][operator_applied] : numpy array
            # iterate over operator names
            for op in sorted(window[feature]):
                # append the numpy array to the dataset
                datasets.append(window[feature][op])

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
