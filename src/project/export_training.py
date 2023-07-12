"""
TODO: change exported training data from a single h5 file with pre-computed
 features to a bundle of pose files, labels, and list of features used for
 the classifier
"""
import h5py
import typing
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

import src.version
import src.classifier
import src.feature_extraction
from src.project import ProjectDistanceUnit

# these are used for type hints, but cause circular imports
# TYPE_CHECKING is always false at runtime, so this gets around that
# also requires enclosing Project and Classifier type hints in quotes
if TYPE_CHECKING:
    from src.project import Project
    from src.classifier import ClassifierType


def export_training_data(project: 'Project',
                         behavior: str,
                         window_size: int,
                         use_social: bool,
                         use_balanced: bool,
                         classifier_type: 'ClassifierType',
                         training_seed: int,
                         out_file: typing.Optional[Path] = None):
    """
    export training data from a project in a format that can be used to
    retrain a classifier elsewhere (for example, by the command line batch
    tool)

    writes exported data to the project directory
    :param project: Project from which to export training data
    :param behavior: Behavior to export
    :param window_size: Window size used for this behavior
    :param use_social: does classifer use social features or not?
    :param use_balanced: should labels be balanced for export?
    :param classifier_type: Preferred classifier type
    :param training_seed: random seed to use for training to get reproducable
    results
    :param out_file: optional output path, if None write to project dir
    with a file name of the form {behavior}_training_YYYYMMDD_hhmmss.h5
    :return: path of output file
    :raises: OSError if unable to create output file (e.g. permission denied,
    no such file or directory, etc)
    """

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    features, group_mapping = project.get_labeled_features(
        behavior, window_size, use_social)

    if out_file is None:
        out_file = (project.dir /
                    f"{project.to_safe_name(behavior)}_training_{ts}.h5")

    string_type = h5py.special_dtype(vlen=str)

    with h5py.File(out_file, 'w') as out_h5:
        out_h5.attrs['file_version'] = src.feature_extraction.FEATURE_VERSION
        out_h5.attrs['app_version'] = src.version.version_str()
        out_h5.attrs['has_social_features'] = use_social
        out_h5.attrs['balance_labels'] = use_balanced
        out_h5.attrs['window_size'] = window_size
        out_h5.attrs['behavior'] = behavior
        out_h5.attrs['classifier_type'] = classifier_type.value
        out_h5.attrs['distance_unit'] = project.distance_unit.name
        out_h5.attrs['training_seed'] = training_seed
        feature_group = out_h5.create_group('features')
        for feature, data in features['per_frame'].items():
            feature_group.create_dataset(f'per_frame/{feature}', data=data)
        for feature in features['window']:
            if isinstance(features['window'][feature], dict):
                for op, data in features['window'][feature].items():
                    feature_group.create_dataset(
                        f'window/{feature}/{op}',
                        data=data)
            else:
                feature_group.create_dataset(f'window/{feature}',
                                             data=features['window'][feature])

        out_h5.create_dataset('group', data=features['groups'])
        out_h5.create_dataset('label', data=features['labels'])

        # store the video/identity to group mapping in the h5 file
        for group in group_mapping:
            dset = out_h5.create_dataset(f'group_mapping/{group}/identity',
                                         (1,), dtype=np.int)
            dset[:] = group_mapping[group]['identity']
            dset = out_h5.create_dataset(f'group_mapping/{group}/video_name',
                                         (1,), dtype=string_type)
            dset[:] = group_mapping[group]['video']

        # store extended features used for this training file
        # structure is:
        #   extended_features/<feature_group_name> = list of feature names
        if project.extended_features is not None:
            feature_group = out_h5.create_group('extended_features')
            for ef in project.extended_features:
                feature_group.create_dataset(
                    ef, data=project.extended_features[ef], dtype=string_type)

    # return output path, so if it was generated automatically the caller
    # will know
    return out_file


def load_training_data(training_file: Path):
    """
    load training data from file

    :param training_file: path to training h5 file
    :return: features, group_mapping
        features: dict containing training data with the following format:
        {
            'per_frame': {}
            'window_features': {},
            'labels': [int],
            'groups': [int],
            'window_size': int,
            'has_social_features': bool,
            'balance_labels': bool,
            'behavior': str,
            'distance_unit': ProjectDistanceUnit,
            'classifier':
            'extended_features': {}
        }

        group_mapping: dict containing group to identity/video mapping:
        {
            group_id: {
                'identity': int,
                'video': str
            },
        }
    :raises: OSError if unable to open h5 file for reading
    """

    features = {
        'per_frame': {},
        'window': {}
    }
    group_mapping = {}

    with h5py.File(training_file, 'r') as in_h5:
        features['has_social_features'] = in_h5.attrs['has_social_features']
        features['balance_labels'] = in_h5.attrs['balance_labels']
        features['window_size'] = in_h5.attrs['window_size']
        features['behavior'] = in_h5.attrs['behavior']
        features['training_seed'] = in_h5.attrs['training_seed']
        features['classifier_type'] = src.classifier.ClassifierType(
            in_h5.attrs['classifier_type'])
        # convert the string distance_unit attr to corresponding
        # ProjectDistanceUnit enum
        unit = in_h5.attrs.get('distance_unit')
        if unit is None:
            # if the training file doesn't include distance_unit it is old and
            # definitely used pixel based distances
            features['distance_unit'] = ProjectDistanceUnit.PIXEL
        else:
            features['distance_unit'] = ProjectDistanceUnit[unit]

        features['labels'] = in_h5['label'][:]
        features['groups'] = in_h5['group'][:]

        # per frame features
        for name, val in in_h5['features/per_frame'].items():
            features['per_frame'][name] = val[:]
        # window features
        for name, val in in_h5['features/window'].items():
            if isinstance(val, h5py.Dataset):
                features['window'][name] = val[:]
            else:
                features['window'][name] = {}
                for op, nested_val in val.items():
                    features['window'][name][op] = nested_val[:]

        # extract the group mapping from h5 file
        for name, val in in_h5['group_mapping'].items():
            group_mapping[int(name)] = {
                'identity': val['identity'][0],
                'video': val['video_name'][0]
            }

        # load required extended features
        if 'extended_features' in in_h5:
            features['extended_features'] = {}
            for group in in_h5['extended_features']:
                features['extended_features'][group] = []
                for f in in_h5[f'extended_features/{group}']:
                    features['extended_features'][group].append(f.decode('utf-8'))
        else:
            features['extended_features'] = None

    return features, group_mapping
