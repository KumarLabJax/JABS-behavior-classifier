"""A classifier trained with social features still classifies a single-identity pose.

A pose with one identity has no social features (``PoseCapability.SOCIAL`` needs more
than one), but a behavior trained on multi-identity videos with social features enabled
keeps asking for those columns. Prediction fills the absent social columns with NaN
instead of failing with a KeyError.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import jabs.pose_estimation as pose_est
from jabs.classifier import Classifier
from jabs.core.enums import ClassifierType
from jabs.feature_extraction import IdentityFeatures
from jabs.feature_extraction.social_features import SocialFeatureGroup

_SINGLE_IDENTITY_V2 = (
    Path(__file__).parents[2]
    / "packages"
    / "jabs-io"
    / "tests"
    / "data"
    / "legacy-pose"
    / "v2-single.h5"
)

_SETTINGS = {"window_size": 5, "social": True, "segmentation": False, "cm_units": False}


def _feature_frame(pose_path: Path | str, pose) -> pd.DataFrame:
    """Combined per-frame and window features of identity 0, as jabs-classify builds them."""
    features = IdentityFeatures(pose_path, 0, None, pose, op_settings=_SETTINGS).get_features(
        _SETTINGS["window_size"]
    )
    return Classifier.combine_data(
        pd.DataFrame(features["per_frame"]), pd.DataFrame(features["window"])
    )


def _is_social(column: str) -> bool:
    return column.split(" ", 1)[0] in SocialFeatureGroup.module_names()


@pytest.fixture(scope="module")
def social_training_frame(pose_est_v5_short) -> pd.DataFrame:
    """Features of a multi-identity pose, social columns included."""
    frame = _feature_frame("short_pose_est_v5.h5", pose_est_v5_short)
    assert any(_is_social(c) for c in frame.columns)
    return frame


@pytest.fixture(scope="module")
def training_frame(
    social_training_frame: pd.DataFrame, single_identity_frame: pd.DataFrame
) -> pd.DataFrame:
    """The multi-identity features, less the landmark columns the single-identity pose lacks.

    The v5 sample has arena corners and the v2 file does not, so only the social
    columns differ between the two.
    """
    keep = [
        c
        for c in social_training_frame.columns
        if c in single_identity_frame.columns or _is_social(c)
    ]
    return social_training_frame[keep]


@pytest.fixture(scope="module")
def single_identity_frame(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Features of a one-identity pose: no social columns."""
    path = tmp_path_factory.mktemp("single") / "single_pose_est_v2.h5"
    shutil.copyfile(_SINGLE_IDENTITY_V2, path)
    pose = pose_est.open_pose_file(path)
    frame = _feature_frame(path, pose)
    assert not any(_is_social(c) for c in frame.columns)
    return frame


def _trained(frame: pd.DataFrame, classifier_type: ClassifierType) -> Classifier:
    classifier = Classifier()
    classifier.set_classifier(classifier_type)
    classifier.behavior_name = "Test"
    classifier.set_dict_settings({"balance_labels": False, "symmetric_behavior": False})
    labels = np.arange(len(frame)) % 2
    classifier.train({"training_data": frame, "training_labels": labels}, random_seed=0)
    return classifier


@pytest.mark.parametrize(
    "classifier_type",
    [ClassifierType.RANDOM_FOREST, ClassifierType.XGBOOST],
    ids=["random-forest", "xgboost"],
)
def test_single_identity_pose_classifies_with_a_social_classifier(
    training_frame: pd.DataFrame,
    single_identity_frame: pd.DataFrame,
    classifier_type: ClassifierType,
) -> None:
    """The missing social columns are filled, and cleaned as for any other NaN."""
    classifier = _trained(training_frame, classifier_type)
    model = classifier._classifier

    with mock.patch.object(model, "predict_proba", wraps=model.predict_proba) as spy:
        probabilities = classifier.predict_proba(single_identity_frame)

    assert probabilities.shape[0] == len(single_identity_frame)
    seen = spy.call_args.args[0]
    social = [c for c in seen.columns if _is_social(c)]
    assert social, "the model was not given the social columns it was trained on"
    if classifier_type is ClassifierType.RANDOM_FOREST:
        # Random forests are trained on NaN-free features; the filled columns are cleaned too
        assert not seen.isna().to_numpy().any()
    else:
        assert seen[social].isna().all().all()


def test_a_missing_non_social_column_still_fails(
    training_frame: pd.DataFrame, single_identity_frame: pd.DataFrame
) -> None:
    """Only social columns are filled; any other missing column is still an error."""
    classifier = _trained(training_frame, ClassifierType.RANDOM_FOREST)
    non_social = next(c for c in single_identity_frame.columns if not _is_social(c))

    with pytest.raises(KeyError):
        classifier.predict_proba(single_identity_frame.drop(columns=[non_social]))
