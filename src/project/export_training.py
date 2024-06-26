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
import pandas as pd

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
                         pose_version: int,
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
    :param pose_version: Minimum required pose version for this classifier
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
        behavior)

    if out_file is None:
        out_file = (project.dir /
                    f"{project.to_safe_name(behavior)}_training_{ts}.h5")

    string_type = h5py.special_dtype(vlen=str)

    with h5py.File(out_file, 'w') as out_h5:
        out_h5.attrs['file_version'] = src.feature_extraction.FEATURE_VERSION
        out_h5.attrs['app_version'] = src.version.version_str()
        out_h5.attrs['min_pose_version'] = pose_version
        out_h5.attrs['behavior'] = behavior
        write_project_settings(out_h5, project.get_behavior_metadata(behavior), 'settings')
        out_h5.attrs['classifier_type'] = classifier_type.value
        out_h5.attrs['training_seed'] = training_seed
        feature_group = out_h5.create_group('features')
        for feature, data in features['per_frame'].items():
            feature_group.create_dataset(f'per_frame/{feature}', data=data)
        for feature, data in features['window'].items():
            feature_group.create_dataset(f'window/{feature}', data=data)

        out_h5.create_dataset('group', data=features['groups'])
        out_h5.create_dataset('label', data=features['labels'])

        # store the video/identity to group mapping in the h5 file
        for group in group_mapping:
            dset = out_h5.create_dataset(f'group_mapping/{group}/identity',
                                         (1,), dtype=np.int64)
            dset[:] = group_mapping[group]['identity']
            dset = out_h5.create_dataset(f'group_mapping/{group}/video_name',
                                         (1,), dtype=string_type)
            dset[:] = group_mapping[group]['video']

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
            'behavior': str,
            'settings': {},
            'classifier':
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
        features['min_pose_version'] = in_h5.attrs['min_pose_version']
        features['behavior'] = in_h5.attrs['behavior']
        features['settings'] = read_project_settings(in_h5['settings'])
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
        features['per_frame'] = pd.DataFrame(features['per_frame'])
        # window features
        for name, val in in_h5['features/window'].items():
            features['window'][name] = val[:]
        features['window'] = pd.DataFrame(features['window'])

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

def read_project_settings(h5_file: h5py.Group) -> dict:
    """
    read dict of project settings

    :param h5_file: open h5 file to read settings from
    :return: dictionary of all project settings
    """
    all_settings = {}
    root_len = len(h5_file.name) + 1

    def _walk_project_settings(name, node) -> dict:
        """
        read dict of project settings walker

        :param name: root where node is located
        :param node: name of node currently visiting
        :return: dictionary of walked setting (if valid node)
        :raises: ValueError if settings have too much depth

        meant to be used with h5py's visititems
        this walk can't use return/yield, so we just mutate the dict each visit
        settings can only be a max of 1 deep
        """
        fullname = node.name[root_len:]
        if isinstance(node, h5py.Dataset):
            if '/' in fullname:
                level_name, key = fullname.split('/')
                level_settings = all_settings.get(level_name, {})
                level_settings.update({key: node[...].item()})
                all_settings.update({level_name: level_settings})
            else:
                all_settings.update({fullname: node[...].item()})
    
    h5_file.visititems(_walk_project_settings)
    return all_settings

def write_project_settings(h5_file: typing.Union[h5py.File, h5py.Group], settings: dict, node: str = 'settings'):
    """
    write project settings to a training h5 file recursively

    :param h5_file: open h5 file to write to
    :param settings: dict of project settings
    :param node: name of the node to write to
    """
    current_group = h5_file.require_group(node)
    for key, val in settings.items():
        if type(val) is dict:
            write_project_settings(current_group, val, key)
        else:
            current_group.create_dataset(key, data=val)
