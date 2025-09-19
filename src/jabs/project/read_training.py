from pathlib import Path
from typing import Any

import h5py
import pandas as pd

from jabs.types import ClassifierType, ProjectDistanceUnit


def read_project_settings(h5_file: h5py.Group) -> dict:
    """read dict of project settings

    Args:
        h5_file: open h5 file to read settings from

    Returns:
        dictionary of all project settings
    """
    all_settings = {}
    root_len = len(h5_file.name) + 1

    def _walk_project_settings(name, node) -> None:
        """read dict of project settings walker

        Args:
            name: root where node is located
            node: name of node currently visiting

        Returns:
            dictionary of walked setting (if valid node)

        meant to be used with h5py's visititems
        this walk can't use return/yield, so we just mutate the dict each visit
        settings can only be a max of 1 deep
        """
        fullname = node.name[root_len:]
        if isinstance(node, h5py.Dataset):
            if "/" in fullname:
                level_name, key = fullname.split("/")
                level_settings = all_settings.get(level_name, {})
                level_settings.update({key: node[...].item()})
                all_settings.update({level_name: level_settings})
            else:
                all_settings.update({fullname: node[...].item()})

    h5_file.visititems(_walk_project_settings)
    return all_settings


def load_training_data(training_file: Path):
    """load training data from file

    Args:
        training_file: path to training h5 file

    Returns:
        features, group_mapping features: dict containing training data
        with the following format: {
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
    """
    features: dict[str, Any] = {"per_frame": {}, "window": {}}
    group_mapping = {}

    with h5py.File(training_file, "r") as in_h5:
        features["min_pose_version"] = in_h5.attrs["min_pose_version"]
        features["behavior"] = in_h5.attrs["behavior"]
        features["settings"] = read_project_settings(in_h5["settings"])
        features["training_seed"] = in_h5.attrs["training_seed"]
        features["classifier_type"] = ClassifierType(in_h5.attrs["classifier_type"])
        # convert the string distance_unit attr to corresponding
        # ProjectDistanceUnit enum
        unit = in_h5.attrs.get("distance_unit")
        if unit is None:
            # if the training file doesn't include distance_unit it is old and
            # definitely used pixel based distances
            features["distance_unit"] = ProjectDistanceUnit.PIXEL
        else:
            features["distance_unit"] = ProjectDistanceUnit[unit]

        features["labels"] = in_h5["label"][:]
        features["groups"] = in_h5["group"][:]

        # per frame features
        for name, val in in_h5["features/per_frame"].items():
            features["per_frame"][name] = val[:]
        features["per_frame"] = pd.DataFrame(features["per_frame"])
        # window features
        for name, val in in_h5["features/window"].items():
            features["window"][name] = val[:]
        features["window"] = pd.DataFrame(features["window"])

        # extract the group mapping from h5 file
        for name, val in in_h5["group_mapping"].items():
            group_mapping[int(name)] = {
                "identity": val["identity"][0],
                "video": val["video_name"][0],
            }

        # load required extended features
        if "extended_features" in in_h5:
            features["extended_features"] = {}
            for group in in_h5["extended_features"]:
                features["extended_features"][group] = []
                for f in in_h5[f"extended_features/{group}"]:
                    features["extended_features"][group].append(f.decode("utf-8"))
        else:
            features["extended_features"] = None

    return features, group_mapping
