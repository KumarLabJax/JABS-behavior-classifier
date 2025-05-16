"""TODO: change exported training data from a single h5 file with pre-computed
 features to a bundle of pose files, labels, and list of features used for
 the classifier
"""

import h5py
import typing
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

import jabs.feature_extraction
import jabs.version
from jabs.project.project_utils import to_safe_name

# these are used for type hints, but cause circular imports
# TYPE_CHECKING is always false at runtime, so this gets around that
# also requires enclosing Project and Classifier type hints in quotes
if TYPE_CHECKING:
    from jabs.project import Project
    from jabs.types import ClassifierType


def export_training_data(
    project: "Project",
    behavior: str,
    pose_version: int,
    classifier_type: "ClassifierType",
    training_seed: int,
    out_file: typing.Optional[Path] = None,
):
    """export training data from a project in a format that can be used to
    retrain a classifier elsewhere (for example, by the command line batch
    tool)

    writes exported data to the project directory

    Args:
        project: Project from which to export training data
        behavior: Behavior to export
        pose_version: Minimum required pose version for this classifier
        classifier_type: Preferred classifier type
        training_seed: random seed to use for training to get
            reproducable results
        out_file: optional output path, if None write to project dir
            with a file name of the form {behavior}_training_YYYYMMDD_hhmmss.h5

    Returns:
        path of output file

    Raises:
        OSError if unable to create output file (e.g. permission denied, no such file or directory, etc)
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
        write_project_settings(
            out_h5, project.settings_manager.get_behavior(behavior), "settings"
        )
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
        for group in group_mapping:
            dset = out_h5.create_dataset(
                f"group_mapping/{group}/identity", (1,), dtype=np.int64
            )
            dset[:] = group_mapping[group]["identity"]
            dset = out_h5.create_dataset(
                f"group_mapping/{group}/video_name", (1,), dtype=string_type
            )
            dset[:] = group_mapping[group]["video"]

    # return output path, so if it was generated automatically the caller
    # will know
    return out_file


def write_project_settings(
    h5_file: typing.Union[h5py.File, h5py.Group], settings: dict, node: str = "settings"
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
        else:
            current_group.create_dataset(key, data=val)
