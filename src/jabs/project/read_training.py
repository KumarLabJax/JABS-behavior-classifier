from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from jabs.core.enums import ClassifierType, ProjectDistanceUnit


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

        # Handle classifier_type - support both old integer format and new string format
        classifier_type_value = in_h5.attrs["classifier_type"]
        if isinstance(classifier_type_value, (int, np.integer)):  # noqa: UP038
            # Old integer format: 1 = Random Forest, 3 = XGBoost
            if classifier_type_value == 1:
                features["classifier_type"] = ClassifierType.RANDOM_FOREST
            elif classifier_type_value == 3:
                features["classifier_type"] = ClassifierType.XGBOOST
            else:
                raise ValueError(f"Unknown classifier type integer: {classifier_type_value}")
        else:
            # New string format: "Random Forest" or "XGBoost"
            features["classifier_type"] = ClassifierType(classifier_type_value)
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


def load_multiclass_training_data(training_file: Path) -> tuple[dict[str, Any], dict]:
    """Load multi-class training data from an exported HDF5 file.

    Args:
        training_file: Path to a multi-class training HDF5 file produced by
            ``export_training_data_multiclass()``.

    Returns:
        Tuple of ``(features, group_mapping)`` where ``features`` contains:
            - ``class_names``: ordered list of all class names (index 0 = background)
            - ``behavior_names``: behavior names only (class_names[1:])
            - ``labels_by_behavior``: dict mapping each class name to its label array
            - ``per_frame``: DataFrame of per-frame features
            - ``window``: DataFrame of window features
            - ``groups``: ndarray of group IDs
            - ``classifier_type``: ClassifierType enum
            - ``training_seed``: int
            - ``settings``: dict of project settings
            - ``min_pose_version``: int

        ``group_mapping`` maps group ID → ``{"identity": int | None, "video": str}``.

    Raises:
        ValueError: If the file does not contain a multi-class training export.
    """
    features: dict[str, Any] = {"per_frame": {}, "window": {}}
    group_mapping: dict[int, dict[str, Any]] = {}

    with h5py.File(training_file, "r") as in_h5:
        classifier_mode = in_h5.attrs.get("classifier_mode", "")
        if classifier_mode != "multiclass":
            raise ValueError(
                f"{training_file} is not a multi-class training file "
                f"(classifier_mode={classifier_mode!r})."
            )

        features["min_pose_version"] = in_h5.attrs["min_pose_version"]
        features["training_seed"] = in_h5.attrs["training_seed"]
        features["classifier_type"] = ClassifierType(in_h5.attrs["classifier_type"])
        features["settings"] = read_project_settings(in_h5["settings"])

        raw = in_h5["class_names"][:]
        class_names = [n.decode() if isinstance(n, bytes) else str(n) for n in raw]
        features["class_names"] = class_names
        features["behavior_names"] = class_names[1:]

        labels_by_behavior: dict[str, np.ndarray] = {}
        for i, name in enumerate(class_names):
            labels_by_behavior[name] = in_h5[f"labels/{i}"][:]
        features["labels_by_behavior"] = labels_by_behavior

        features["groups"] = in_h5["group"][:]

        for name, val in in_h5["features/per_frame"].items():
            features["per_frame"][name] = val[:]
        features["per_frame"] = pd.DataFrame(features["per_frame"])

        for name, val in in_h5["features/window"].items():
            features["window"][name] = val[:]
        features["window"] = pd.DataFrame(features["window"])

        for name, val in in_h5["group_mapping"].items():
            identity_raw = int(val["identity"][0])
            group_mapping[int(name)] = {
                "identity": None if identity_raw == -1 else identity_raw,
                "video": val["video_name"][0],
            }

    return features, group_mapping
