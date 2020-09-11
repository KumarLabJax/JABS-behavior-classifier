import random
from enum import IntEnum

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, LeaveOneGroupOut


class SklClassifier:

    LABEL_THRESHOLD = 100
    MIN_GROUPS = 2

    class ClassifierType(IntEnum):
        RANDOM_FOREST = 1
        ADABOOST = 2

    def __init__(self, classifier=ClassifierType.RANDOM_FOREST):
        """
        initialize a new
        :param classifier:
        """

        self._classifier_type = classifier
        self._classifier = None

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
        split_data = train_test_split(np.concatenate(datasets, axis=1), label_data)

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

        x = SklClassifier.combine_data(per_frame_features, window_features)

        splits = logo.split(x, labels, groups)

        # pick random split
        split = random.choice(list(splits))

        return {
            'training_labels': labels[split[0]],
            'training_data': x[split[0]],
            'test_labels': labels[split[1]],
            'test_data':  x[split[1]]
        }

    def train(self, data):
        """
        train the classifier
        :param data: dict returned from train_test_split()
        :return: None
        """

        features = data['training_data']
        labels = data['training_labels']

        if self._classifier_type == self.ClassifierType.RANDOM_FOREST:
            self._classifier = self._fit_random_forest(features, labels)

    def predict(self, features):
        """
        predict classes for a given set of features

        """
        return self._classifier.predict(features)

    def predict_proba(self, features):
        return self._classifier.predict_proba(features)

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
        for feature in sorted(per_frame):
            datasets.append(per_frame[feature])

        # add window features to our data set
        for feature in sorted(window):
            if feature == 'percent_frames_present':
                datasets.append(window[feature])
            else:
                # [source_feature_name][operator_applied] : numpy array
                # iterate over operator names
                for op in sorted(window[feature]):
                    # append the numpy array to the dataset
                    datasets.append(window[feature][op])

        return np.concatenate(datasets, axis=1)

    @staticmethod
    def _fit_random_forest(features, labels):

        classifier = RandomForestClassifier()
        classifier.fit(features, labels)

        return classifier

    def print_feature_importance(self, feature_list, limit=20):
        """

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
        # Print out the feature and importances
        for feature, importance in feature_importances[:limit]:
            print(f"Variable: {feature:20} Importance: {importance}")

    @staticmethod
    def label_threshold_met(label_counts):
        group_count = 0
        for video, counts in label_counts.items():
            for count in counts:
                if count[1] >= SklClassifier.LABEL_THRESHOLD:
                    group_count += 1

        return True if group_count >= SklClassifier.MIN_GROUPS else False