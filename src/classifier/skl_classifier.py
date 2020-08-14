from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import numpy as np
from enum import IntEnum

from src.feature_extraction.features import AngleIndex,IdentityFeatures


class SklClassifier:

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
            'feature_list': list,

        }
        """
        dataset = []
        feature_list = []

        # add per frame features to our dataset
        for feature in per_frame_features:
            dataset.append(per_frame_features[feature])

            if feature == 'angles':
                feature_list.extend([f"angle {angle.name}" for angle in AngleIndex])
            elif feature == 'pairwise_distances':
                feature_list.extend(IdentityFeatures.get_distance_names())

        # add window features to our dataset
        for feature in window_features:
            if feature == 'percent_frames_present':
                dataset.append(window_features[feature])
                feature_list.append(feature)
            else:
                # [source_feature_name][operator_applied] : numpy array
                # iterate over operator names
                for op in window_features[feature]:
                    # append the numpy array to the dataset
                    dataset.append(window_features[feature][op])

                    if feature == 'angles':
                        feature_list.extend(
                            [f"{op} angle {angle.name}" for angle in AngleIndex])
                    elif feature == 'pairwise_distances':
                        feature_list.extend(
                            [f"{op} {d}" for d in IdentityFeatures.get_distance_names()])

        # split labeled data and labels
        split_data = train_test_split(*dataset, label_data)

        return {
            'test_labels': split_data.pop(),
            'training_labels': split_data.pop(),
            'training_data': split_data[::2],
            'test_data': split_data[1::2],
            'feature_list': feature_list
        }

    def train(self, data):
        """
        train the classifier
        :param data: dict returned from train_test_split()
        :return: None
        """

        features = np.concatenate(data['training_data'], axis=1)
        labels = data['training_labels']

        if self._classifier_type == self.ClassifierType.RANDOM_FOREST:
            self._classifier = self._fit_random_forest(features, labels)

    def predict(self, features):
        """
        predict classes for a given set of features

        """
        return self._classifier.predict(features)

    @staticmethod
    def _fit_random_forest(features, labels):

        classifier = RandomForestClassifier()
        classifier.fit(features, labels)

        return classifier

    def print_feature_importance(self, feature_list):
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
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in
         feature_importances];
