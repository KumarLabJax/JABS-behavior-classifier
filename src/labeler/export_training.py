import h5py
import typing
from pathlib import Path
from datetime import datetime

import numpy as np

import src.version
from .project import Project
from src.classifier import Classifier


def export_training_data(project: Project, behavior: str,
                         window_size: int,
                         out_file: typing.Optional[Path]=None):
    """
    export training data from a project in a format that can be used to
    retrain a classifier elsewhere (for example, by the command line batch
    tool)

    writes exported data to the project directory
    :param project: Project to export training data for
    :param behavior:
    :param window_size:
    :param out_file
    :return:
    """

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    features, group_mapping = project.get_labeled_features(behavior)

    if out_file is None:
        out_file = (project.dir /
                    f"{project.to_safe_name(behavior)}_training_{ts}.h5")

    string_type = h5py.special_dtype(vlen=str)

    with h5py.File(out_file, 'w') as out_h5:
        out_h5.attrs['file_version'] = Classifier.TRAINING_FILE_VERSION
        out_h5.attrs['app_version'] = src.version.version_str()
        out_h5.attrs['has_social_features'] = project.has_social_features
        feature_group = out_h5.create_group('features')
        for feature, data in features['per_frame'].items():
            feature_group.create_dataset(f'per_frame/{feature}', data=data)
        for feature, data in features['window'].items():
            if feature == 'percent_frames_present':
                feature_group.create_dataset(f'window/{window_size}/{feature}',
                                             data=data)
            else:
                for op, vals in data.items():
                    feature_group.create_dataset(
                        f'window/{window_size}/{feature}/{op}',
                        data=vals)
        out_h5.create_dataset('group', data=features['groups'])
        out_h5.create_dataset('labels', data=features['labels'])

        # store the video/identity to group mapping in the h5 file
        for group in group_mapping:
            dset = out_h5.create_dataset(f'group_mapping/{group}/identity',
                                           (1,), dtype=np.int)
            dset[:] = group_mapping[group]['identity']
            dset = out_h5.create_dataset(f'group_mapping/{group}/video_name',
                                           (1,), dtype=string_type)
            dset[:] = group_mapping[group]['video']
