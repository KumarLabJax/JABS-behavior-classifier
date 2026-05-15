import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

import jabs.feature_extraction
import jabs.version
from jabs.core.constants import FINAL_TRAIN_SEED, MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode
from jabs.project.project_utils import to_safe_name

# these are used for type hints, but cause circular imports
# TYPE_CHECKING is always false at runtime, so this gets around that
# also requires enclosing Project and Classifier type hints in quotes
if TYPE_CHECKING:
    from jabs.core.enums import ClassifierType
    from jabs.project import Project


def export_training_data(
    project: "Project",
    behavior: str,
    pose_version: int,
    classifier_type: "ClassifierType",
    training_seed: int = FINAL_TRAIN_SEED,
    out_file: Path | None = None,
):
    """
    Export labeled training data from a JABS project for classifier retraining.

    This function extracts features and labels for a specified behavior and writes them,
    along with relevant project and classifier metadata, to an HDF5 file. The exported
    file can be used for retraining classifiers outside the current environment.

    Args:
        project (Project): The JABS project to export data from.
        behavior (str): Name of the behavior to export.
        pose_version (int): Minimum required pose version for the classifier.
        classifier_type (ClassifierType): The classifier type for which data is exported.
        training_seed (int): Random seed to ensure reproducible training splits.
        out_file (Path, optional): Output file path. If None, a file is created in the
            project directory with a timestamped name.

    Returns:
        Path: The path to the exported HDF5 file.

    Raises:
        OSError: If the output file cannot be created or written.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    features, group_mapping = project.get_labeled_features(behavior)

    if out_file is None:
        out_file = project.dir / f"{to_safe_name(behavior)}_training_{ts}.h5"

    string_type = h5py.special_dtype(vlen=str)

    with h5py.File(out_file, "w") as out_h5:
        out_h5.attrs["file_version"] = jabs.feature_extraction.FEATURE_VERSION
        out_h5.attrs["app_version"] = jabs.version.version_str()
        out_h5.attrs["min_pose_version"] = pose_version
        out_h5.attrs["behavior"] = behavior
        write_project_settings(out_h5, project.settings_manager.get_behavior(behavior), "settings")
        out_h5.attrs["classifier_type"] = classifier_type.value
        out_h5.attrs["training_seed"] = training_seed
        feature_group = out_h5.create_group("features")
        for feature, data in features["per_frame"].items():
            feature_group.create_dataset(f"per_frame/{feature}", data=data)
        for feature, data in features["window"].items():
            feature_group.create_dataset(f"window/{feature}", data=data)

        out_h5.create_dataset("group", data=features["groups"])
        out_h5.create_dataset("label", data=features["labels"])

        # store the video/identity to group mapping in the h5 file
        # identity is None when VIDEO grouping strategy is used; store -1 as a sentinel
        for group in group_mapping:
            dset = out_h5.create_dataset(f"group_mapping/{group}/identity", (1,), dtype=np.int64)
            identity = group_mapping[group]["identity"]
            dset[:] = identity if identity is not None else -1
            dset = out_h5.create_dataset(
                f"group_mapping/{group}/video_name", (1,), dtype=string_type
            )
            dset[:] = group_mapping[group]["video"]

    # return output path, so if it was generated automatically the caller
    # will know
    return out_file


def export_training_data_multiclass(
    project: "Project",
    pose_version: int,
    classifier_type: "ClassifierType",
    training_seed: int = FINAL_TRAIN_SEED,
    out_file: Path | None = None,
) -> Path:
    """Export labeled training data from a JABS multi-class project for classifier retraining.

    Collects per-frame and window features alongside per-behavior label arrays and writes
    them to an HDF5 file. The exported file can be used to retrain a ``MultiClassClassifier``
    outside the current environment with a stable class-index mapping.

    Args:
        project: The JABS project to export data from.
        pose_version: Minimum required pose version for the classifier.
        classifier_type: The classifier type for which data is exported.
        training_seed: Random seed to ensure reproducible training splits.
        out_file: Output file path. If None, a file is created in the project directory
            with a timestamped name.

    Returns:
        Path: The path to the exported HDF5 file.

    Raises:
        OSError: If the output file cannot be created or written.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    features, group_mapping = project.get_multiclass_labeled_features()

    if out_file is None:
        out_file = project.dir / f"multiclass_training_{ts}.h5"

    string_type = h5py.special_dtype(vlen=str)
    behavior_names = project.settings_manager.behavior_names
    # class_names[0] is always MULTICLASS_NONE_BEHAVIOR; behavior_names[i] maps to class i+1.
    class_names = [MULTICLASS_NONE_BEHAVIOR, *behavior_names]

    with h5py.File(out_file, "w") as out_h5:
        out_h5.attrs["file_version"] = jabs.feature_extraction.FEATURE_VERSION
        out_h5.attrs["app_version"] = jabs.version.version_str()
        out_h5.attrs["min_pose_version"] = pose_version
        out_h5.attrs["classifier_mode"] = ClassifierMode.MULTICLASS.value
        out_h5.attrs["classifier_type"] = classifier_type.value
        out_h5.attrs["training_seed"] = training_seed

        # Full ordered class list: class_names[i] is the name for class index i.
        names_ds = out_h5.create_dataset(
            "class_names", shape=(len(class_names),), dtype=string_type
        )
        for i, name in enumerate(class_names):
            names_ds[i] = name

        write_project_settings(out_h5, project.get_project_defaults(), "settings")

        feature_group = out_h5.create_group("features")
        for feature, data in features["per_frame"].items():
            feature_group.create_dataset(f"per_frame/{feature}", data=data)
        for feature, data in features["window"].items():
            feature_group.create_dataset(f"window/{feature}", data=data)

        # Label arrays stored by class index so behavior names are never used as HDF5 paths.
        labels_group = out_h5.create_group("labels")
        for i, name in enumerate(class_names):
            labels_group.create_dataset(str(i), data=features["labels_by_behavior"][name])

        out_h5.create_dataset("group", data=features["groups"])

        for group in group_mapping:
            dset = out_h5.create_dataset(f"group_mapping/{group}/identity", (1,), dtype=np.int64)
            identity = group_mapping[group]["identity"]
            dset[:] = identity if identity is not None else -1
            dset = out_h5.create_dataset(
                f"group_mapping/{group}/video_name", (1,), dtype=string_type
            )
            dset[:] = group_mapping[group]["video"]

    return out_file


def write_project_settings(
    h5_file: h5py.File | h5py.Group, settings: dict, node: str = "settings"
):
    """write project settings to a training h5 file recursively

    Args:
        h5_file: open h5 file to write to
        settings: dict of project settings
        node: name of the node to write to
    """
    current_group = h5_file.require_group(node)
    for key, val in settings.items():
        if type(val) is dict:
            write_project_settings(current_group, val, key)
        elif isinstance(val, list):
            # Lists (e.g. postprocessing stage configs) have no direct HDF5 equivalent;
            # store as a JSON string so the data round-trips without loss.
            current_group.create_dataset(key, data=json.dumps(val))
        else:
            current_group.create_dataset(key, data=val)
