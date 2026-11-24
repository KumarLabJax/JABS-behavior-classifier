"""A social classifier on a one-mouse pose: classified as on main, or refused clearly.

A pose file whose format can hold several mice (``pose_est_v3`` on) has social
features even with one identity, as on main. A pose that cannot (a ``pose_est_v2``
file) has none, and a classifier trained with social features refuses it with
:class:`PoseCapabilityError` instead of a pandas ``KeyError``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import pytest

from jabs.classifier import Classifier
from jabs.core.enums import CacheFormat, ClassifierType, PoseCapability
from jabs.feature_extraction import (
    IdentityFeatures,
    PoseCapabilityError,
    require_pose_capabilities,
)
from jabs.pose_estimation import open_pose_file
from jabs.scripts import classify
from jabs.scripts.cli import evaluate

_DATA = Path(__file__).parents[1] / "data"
_SINGLE_V2 = Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
_SETTINGS = {"window_size": 5, "social": True, "segmentation": False, "cm_units": False}
_FRAMES = 200


def _short_v4(path: Path) -> Path:
    """Write the five-mouse v4 sample's first frames to ``path``."""
    with h5py.File(_DATA / "sample_pose_est_v4.h5", "r") as source, h5py.File(path, "w") as h5:
        pose = h5.create_group("poseest")
        for key, value in source["poseest"].attrs.items():
            pose.attrs[key] = value
        for name, dataset in source["poseest"].items():
            pose.create_dataset(
                name, data=dataset[()] if name == "instance_id_center" else dataset[:_FRAMES]
            )
    return path


@pytest.fixture(scope="module")
def social_classifier(tmp_path_factory: pytest.TempPathFactory) -> Classifier:
    """A Random Forest trained with social features on a five-mouse v4 pose."""
    path = _short_v4(tmp_path_factory.mktemp("multi") / "multi_pose_est_v4.h5")
    with open_pose_file(path) as pose:
        features = IdentityFeatures(path, 0, None, pose, op_settings=_SETTINGS).get_features(5)
    frame = Classifier.combine_data(
        pd.DataFrame(features["per_frame"]), pd.DataFrame(features["window"])
    )
    classifier = Classifier()
    classifier.set_classifier(ClassifierType.RANDOM_FOREST)
    classifier.behavior_name = "Test"
    classifier.set_dict_settings(
        {**_SETTINGS, "balance_labels": False, "symmetric_behavior": False}
    )
    classifier.train(
        {"training_data": frame, "training_labels": np.arange(len(frame)) % 2}, random_seed=0
    )
    return classifier


def test_social_classifier_classifies_a_one_mouse_v4_pose(
    tmp_path: Path, social_classifier: Classifier, single_mouse_v4: Callable[[Path], Path]
) -> None:
    """jabs-classify gives the model the one-mouse file's real social features, as main did."""
    pose_path = single_mouse_v4(tmp_path / "single_pose_est_v4.h5")
    model = social_classifier._classifier

    with mock.patch.object(model, "predict_proba", wraps=model.predict_proba) as spy:
        classify.classify_pose(social_classifier, pose_path, tmp_path / "out", behavior="Test")

    seen = spy.call_args.args[0]
    # computed, not filled in: the one mouse is a group of one
    assert (seen["huddle_features social connected group size"] == 1).all()
    assert (tmp_path / "out" / "single_behavior.h5").exists()


def _v2(tmp_path: Path) -> Path:
    path = tmp_path / "single_pose_est_v2.h5"
    shutil.copyfile(_SINGLE_V2 / "v2-single.h5", path)
    return path


def test_social_classifier_refuses_a_v2_pose(
    tmp_path: Path, social_classifier: Classifier
) -> None:
    """A pose that cannot hold several mice is refused, naming the file and the features."""
    with pytest.raises(PoseCapabilityError, match=r"single_pose_est_v2\.h5.*social"):
        classify.classify_pose(social_classifier, _v2(tmp_path), tmp_path / "out", "Test")
    assert not (tmp_path / "out").exists()


def test_jabs_classify_reports_the_refusal_without_a_traceback(
    tmp_path: Path, social_classifier: Classifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command exits with the refusal's message."""
    monkeypatch.setattr(classify, "load_classifier_from_pickle", lambda _path: social_classifier)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jabs-classify",
            "classify",
            "--classifier",
            str(tmp_path / "classifier.pickle"),
            "--input-pose",
            str(_v2(tmp_path)),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        classify.classify_main()
    assert "social" in str(raised.value.code)


def test_evaluate_refuses_a_v2_pose_for_a_social_classifier(
    tmp_path: Path, social_classifier: Classifier
) -> None:
    """jabs-cli evaluate's per-identity prediction refuses it too (and skips the video)."""
    path = _v2(tmp_path)
    with open_pose_file(path) as pose, pytest.raises(PoseCapabilityError):
        evaluate._predict_identity(social_classifier, path, pose, 0, None, CacheFormat.HDF5)


@pytest.mark.parametrize(
    ("settings", "capabilities", "missing"),
    [
        ({"social": True}, frozenset(), ["social"]),
        ({"social": True}, frozenset({PoseCapability.SOCIAL}), []),
        ({"social": False}, frozenset(), []),
        ({"segmentation": True}, frozenset({PoseCapability.SOCIAL}), ["segmentation"]),
        ({"social": True, "segmentation": True}, frozenset(), ["social", "segmentation"]),
        ({}, frozenset(), []),
    ],
)
def test_require_pose_capabilities(
    settings: dict, capabilities: frozenset[PoseCapability], missing: list[str]
) -> None:
    """Only features the settings turn on and the pose cannot provide are refused."""
    if not missing:
        require_pose_capabilities(settings, capabilities, "vid_pose.h5")
        return
    with pytest.raises(PoseCapabilityError) as raised:
        require_pose_capabilities(settings, capabilities, "vid_pose.h5")
    assert raised.value.missing == tuple(missing)
    assert "vid_pose.h5" in str(raised.value)
