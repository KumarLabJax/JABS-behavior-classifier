from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from enum import IntEnum

from src.feature_extraction import IdentityFeatures

from src.labeler.project import Project
from src.pose_estimation import PoseEstimationV3
from src.labeler.track_labels import TrackLabels


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
            'test_labels': numpy_array

        }
        """
        dataset = []

        # add per frame features to our dataset
        for feature in per_frame_features:
            dataset.append(per_frame_features[feature])

        # add window features to our dataset
        for feature in window_features:
            if feature == 'percent_frames_present':
                dataset.append(window_features[feature])
            else:
                # [source_feature_name][operator_applied] : numpy array
                # iterate over operator names
                for op in window_features[feature]:
                    # append the numpy array to the dataset
                    dataset.append(window_features[feature][op])

        # split labeled data and labels
        split_data = train_test_split(*dataset, label_data)

        return {
            'test_labels': split_data.pop(),
            'training_labels': split_data.pop(),
            'training_data': split_data[::2],
            'test_data': split_data[1::2]
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
