import random
import typing
from enum import IntEnum
from importlib import import_module
from pathlib import Path
import joblib
import re
import json
from ast import literal_eval
import warnings

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

from src.project import TrackLabels, ProjectDistanceUnit, Project

_VERSION = 7

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

def load_hyperparameters()->dict:
    """ 
    This function loads the hyperparameters for each classifier from the hyperparameters.json file.

    :return: a dictionary of hyperparameters for each classifier.
    """
    mapped_parameters = {"random_forest": ClassifierType.RANDOM_FOREST, "xg_boost": ClassifierType.XGBOOST, "gradient_boost": ClassifierType.GRADIENT_BOOSTING}

    with open(Path(__file__).parent.parent.parent / 'hyperparameters.json', "rb") as j:
        data = json.loads(j.read())

    parameters = data["parameters"]

    for classifier in parameters:
        for key in parameters[classifier]:
            try:
                parameters[classifier][key] = literal_eval(parameters[classifier][key])
            except Exception as e:
                continue
    
    return {mapped_parameters[key]: parameters[key] for key in parameters}

class Classifier:
    LABEL_THRESHOLD = 20

    _classifier_names = {
        ClassifierType.RANDOM_FOREST: "Random Forest",
        ClassifierType.GRADIENT_BOOSTING: "Gradient Boosting",
        ClassifierType.XGBOOST: "XGBoost"
    }

    _classifier_hyperparameters = load_hyperparameters()

    def __init__(self, classifier=ClassifierType.RANDOM_FOREST, n_jobs=1):
        """
        :param classifier: type of classifier to use. Must be ClassifierType
        :param n_jobs: number of jobs to use for classifiers that support
        this parameter for parallelism
        enum value. Defaults to ClassifierType.RANDOM_FOREST
        """

        self._classifier_type = classifier
        self._classifier = None
        self._project_settings = None
        self._behavior = None
        self._feature_names = None
        self._n_jobs = n_jobs
        self._version = _VERSION
        self._hyperparameters = self._classifier_hyperparameters[classifier]

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
    def project_settings(self) -> dict:
        """ return a copy of dictionary of project settings for this classifier """
        if self._project_settings is not None:
            return dict(self._project_settings)
        return {}

    @property
    def behavior_name(self) -> str:
        return self._behavior

    @behavior_name.setter
    def behavior_name(self, value) -> str:
        self._behavior = value

    @property
    def version(self) -> int:
        return self._version

    @property
    def feature_names(self) -> list:
        """
        returns the list of feature names used when training this classifier
        """
        return self._feature_names

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
        x_train, x_test, y_train, y_test = train_test_split(all_features, label_data)

        return {
            'training_data': x_train,
            'training_labels': y_train,
            'test_data': x_test,
            'test_labels': y_test,
            'feature_names': all_features.columns.to_list()
        }

    @staticmethod
    def get_leave_one_group_out_max(labels, groups):
        """
        counts the number of possible leave one out groups for k-fold cross validation
        :param labels: labels to check if they were above the threshold
        :param groups: group id corrosponding to the labels
        :return: int of the maximum number of cross validation to use
        """
        unique_groups = np.unique(groups)
        count_behavior = [np.sum(np.asarray(labels)[np.asarray(groups)==x] == TrackLabels.Label.BEHAVIOR) for x in unique_groups]
        count_not_behavior = [np.sum(np.asarray(labels)[np.asarray(groups)==x] == TrackLabels.Label.NOT_BEHAVIOR) for x in unique_groups]
        can_kfold = np.logical_and(np.asarray(count_behavior)>Classifier.LABEL_THRESHOLD, np.asarray(count_not_behavior)>Classifier.LABEL_THRESHOLD)
        return np.sum(can_kfold)

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
                    'training_data': x.iloc[split[0]],
                    'training_labels': labels[split[0]],
                    'test_data': x.iloc[split[1]],
                    'test_labels': labels[split[1]],
                    'test_group': groups[split[1]][0],
                    'feature_names': x.columns.to_list()
                }

        # number of splits exhausted without finding at least one that meets
        # criteria
        # the UI won't allow us to reach this case
        if count == 0:
            raise ValueError("unable to split data")
        # If there are no more splits to yield, just let generator end

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
        features = features.iloc[selected_samples]
        labels = labels[selected_samples]
        return features, labels

    @staticmethod
    def augment_symmetric(features, labels, random_str='ASygRQDZJD'):
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

        # Figure out the L-R swapping of features
        lowercase_features = np.array([x.lower() for x in features.columns.to_list()])
        reflected_feature_names = [re.sub(r'left', random_str, x) for x in lowercase_features]
        reflected_feature_names = [re.sub(r'right', 'left', x) for x in reflected_feature_names]
        reflected_feature_names = [re.sub(random_str, 'right', x) for x in reflected_feature_names]
        reflected_idxs = [np.where(lowercase_features == x)[0][0] if x in lowercase_features else i for i, x in enumerate(reflected_feature_names)]
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
        """ change the type of the classifier being used """
        if classifier not in _classifier_choices:
            raise ValueError("Invalid Classifier Type")
        self._classifier_type = classifier
        self._hyperparameters = self._classifier_hyperparameters[classifier]

    def set_project_settings(self, project: Project):
        """
        assign project settings to the classifier
        :project: project to copy classifier-relevant settings from for the current behavior

        if no behavior is currently set, will simply use project defaults
        """
        if self._behavior is None:
            self._project_settings = project.get_project_defaults()
        else:
            self._project_settings = project.get_behavior_metadata(self._behavior)

    def set_dict_settings(self, settings: dict):
        """
        assign project settings via a dict to the classifier
        :settings: dict of project settings. Must be same structure as project.get_behavior_metadata

        TODO: Add checks to enforce conformity to project settings
        """
        self._project_settings = dict(settings)

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

    def train(self, data, random_seed: typing.Optional[int] = None):
        """
        train the classifier
        :param data: dict returned from train_test_split()
        :param random_seed: optional random seed (used when we want reproducible
        results between trainings)
        :return: None

        raises ValueError for having either unset project settings or an unset classifier
        """
        if self._project_settings is None:
            raise ValueError('Project settings for classifier unset, cannot train classifier.')

        # Assume that feature names is provided, otherwise extract it from the dataframe
        if 'feature_names' in data.keys():
            self._feature_names = data['feature_names']
        else:
            self._feature_names = data['training_data'].columns.to_list()

        # Obtain the feature and label matrices
        features = data['training_data']
        labels = data['training_labels']
        # Symmetric augmentation should occur before balancing so that the class with more labels can sample from the whole set
        if self._project_settings.get('symmetric_behavior', False):
            features, labels = self.augment_symmetric(features, labels)
        if self._project_settings.get('balance_labels', False):
            features, labels = self.downsample_balance(features, labels, random_seed)

        if self._classifier_type == ClassifierType.RANDOM_FOREST:
            self._classifier = self._fit_random_forest(features, labels,
                                                       random_seed=random_seed)
        elif self._classifier_type == ClassifierType.GRADIENT_BOOSTING:
            self._classifier = self._fit_gradient_boost(features, labels,
                                                        random_seed=random_seed)
        elif _xgboost is not None and self._classifier_type == ClassifierType.XGBOOST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                self._classifier = self._fit_xgboost(features, labels,
                                                     random_seed=random_seed)
        else:
            raise ValueError("Unsupported classifier")

    def sort_features_to_classify(self, features):
        """
        sorts features to match the current classifier
        """
        if self._classifier_type == ClassifierType.XGBOOST:
            classifier_columns = self._classifier.get_booster().feature_names
        # sklearn places feature names in the same spot
        else:
            classifier_columns = self._classifier.feature_names_in_
        features_sorted = features[classifier_columns]
        return features_sorted

    def predict(self, features):
        """
        predict classes for a given set of features
        """
        if self._classifier_type == ClassifierType.XGBOOST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                result = self._classifier.predict(self.sort_features_to_classify(features))
            return result
        # Random forests and gradient boost can't handle NAs, so fill them with 0s
        return self._classifier.predict(self.sort_features_to_classify(features.fillna(0)))

    def predict_proba(self, features):
        """
        predict probabilities for a given set of features
        """
        if self._classifier_type == ClassifierType.XGBOOST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                result = self._classifier.predict_proba(self.sort_features_to_classify(features))
            return result
        # Random forests and gradient boost can't handle NAs, so fill them with 0s
        return self._classifier.predict_proba(self.sort_features_to_classify(features.fillna(0)))

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
        self._project_settings = c._project_settings
        self._classifier_type = c._classifier_type

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
                                                random_state=random_seed, **self._hyperparameters)
        else:
            classifier = RandomForestClassifier(n_jobs=self._n_jobs, **self._hyperparameters)
        return classifier.fit(features.fillna(0), labels)

    def _fit_gradient_boost(self, features, labels,
                            random_seed: typing.Optional[int] = None):
        if random_seed is not None:
            classifier = GradientBoostingClassifier(random_state=random_seed, **self._hyperparameters)
        else:
            classifier = GradientBoostingClassifier(**self._hyperparameters)
        return classifier.fit(features.fillna(0), labels)

    def _fit_xgboost(self, features, labels,
                     random_seed: typing.Optional[int] = None):
        if random_seed is not None:
            classifier = _xgboost.XGBClassifier(n_jobs=self._n_jobs,
                                                random_state=random_seed, **self._hyperparameters)
        else:
            classifier = _xgboost.XGBClassifier(n_jobs=self._n_jobs, **self._hyperparameters)
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
    def count_label_threshold(all_counts: dict):
        """
        counts the number of groups that meet label threshold criteria
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
        :return: number of groups that meet label criteria
        """
        group_count = 0
        for video, counts in all_counts.items():
            for count in counts:
                if (count[1][0] >= Classifier.LABEL_THRESHOLD and
                        count[1][1] >= Classifier.LABEL_THRESHOLD):
                    group_count += 1
        return group_count

    @staticmethod
    def label_threshold_met(all_counts: dict, min_groups: int):
        """
        determine if the labeling threshold is met
        :param all_counts: labeled frame and bout counts for the entire project
        :param min_groups: minimum number of groups required (more than one
        group is always required for the "leave one group out" train/test split,
        but may be more than 2 for k-fold cross validation if k > 2)
        :return: bool if requested valid groups is > valid group
        """
        group_count = Classifier.count_label_threshold(all_counts)
        return True if 1 < group_count >= min_groups else False
