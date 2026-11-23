"""Unit tests for jabs.scripts.classify helpers."""

import shutil
from pathlib import Path
from unittest import mock

import pytest

from jabs.classifier import ClassifierFeatureVersionError, MultiClassClassifier
from jabs.pose_estimation import conversion_cache
from jabs.scripts import classify
from jabs.scripts.classify import _require_pose_file_name


@pytest.mark.parametrize(
    "name",
    [
        "sample_pose_est_v2.h5",
        "sample_pose_est_v6.h5",
        "nested_name_pose_est_v8.h5",
        "sample_pose.h5",
        "nested_pose_name_pose.h5",
    ],
)
def test_require_pose_file_name_accepts_canonical(name: str) -> None:
    """Unified ``*_pose.h5`` and legacy ``*_pose_est_vN.h5`` filenames pass validation."""
    _require_pose_file_name(Path("/some/dir") / name)


@pytest.mark.parametrize(
    "name",
    [
        "sample.mp4",
        "sample.h5",
        "sample_v6.h5",
        "sample_pose_est.h5",
        "sample_pose_est_v6.hdf5",
        "predictions_pose_est_v6.h5.bak",
        "sample_pose.hdf5",
        "._sample_pose.h5",
        "._sample_pose_est_v6.h5",
        ".hidden_pose.h5",
        "sample_pose_est_v9.h5",
        "sample_pose_est_v12.h5",
    ],
)
def test_require_pose_file_name_rejects_invalid(name: str) -> None:
    """Names ``is_pose_file`` rejects are rejected: one rule for every pose file name.

    A macOS AppleDouble ``._*`` sidecar is not a pose file, nor is a legacy version
    JABS cannot open (which ``open_pose_file`` would refuse anyway).
    """
    with pytest.raises(ValueError, match="not a valid pose file path"):
        _require_pose_file_name(Path(name))


def test_classify_fps_is_deprecated_and_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """``jabs-classify classify --fps`` logs a deprecation warning and is not forwarded."""
    from unittest import mock

    from jabs.scripts import classify

    monkeypatch.setattr(
        "sys.argv",
        [
            "jabs-classify",
            "classify",
            "--training",
            str(tmp_path / "training.h5"),
            "--input-pose",
            str(tmp_path / "video_pose_est_v6.h5"),
            "--out-dir",
            str(tmp_path),
            "--fps",
            "60",
        ],
    )
    with mock.patch.object(classify, "train_and_classify") as train_and_classify:
        classify.classify_main()

    assert "fps" not in train_and_classify.call_args.kwargs
    assert any("--fps is deprecated" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("multiclass", [False, True], ids=["binary", "multi"])
def test_training_on_an_old_export_exits_with_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, multiclass: bool
) -> None:
    """A training file from another feature version ends the command with a message."""
    error = ClassifierFeatureVersionError("re-export the training data and retrain", 19, 20)
    owner = MultiClassClassifier if multiclass else classify.Classifier
    monkeypatch.setattr(owner, "from_training_file", mock.Mock(side_effect=error))
    trainer = classify.train_multiclass if multiclass else classify.train

    with pytest.raises(SystemExit) as raised:
        trainer(tmp_path / "old_training.h5")

    assert "retrain" in str(raised.value.code)


def test_classify_reports_an_unopenable_pose_file_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pose file the converter refuses ends the command with its path and the reason."""
    monkeypatch.setenv("JABS_STRICT_POSE_CONVERSION", "1")  # lenient opens it
    unusable = (
        Path(__file__).parents[2]
        / "packages"
        / "jabs-io"
        / "tests"
        / "data"
        / "legacy-pose"
        / "v8-unknown-dataset.h5"
    )
    pose_path = tmp_path / "bad_pose_est_v8.h5"
    shutil.copyfile(unusable, pose_path)
    classifier = mock.Mock(
        classifier_type=next(iter(classify.Classifier().classifier_choices())),
        behavior_name="grooming",
        project_settings={"window_size": 5, "social": False, "cm_units": False},
    )
    monkeypatch.setattr(classify, "load_classifier_from_pickle", lambda _path: classifier)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jabs-classify",
            "classify",
            "--classifier",
            str(tmp_path / "classifier.pickle"),
            "--input-pose",
            str(pose_path),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        classify.classify_main()

    message = str(raised.value.code)
    assert str(pose_path) in message
    assert "poseest/mystery is not a legacy pose dataset" in message
    assert "Traceback" not in message


class _StopAfterOpen(Exception):
    """Raised by the IdentityFeatures stand-in, once the pose is open."""


def _classify_until_features(
    monkeypatch: pytest.MonkeyPatch, pose_path: Path, out: Path, **kwargs
) -> Path:
    """Run classify_pose up to feature extraction; return the conversion it opened."""
    opened: list[Path] = []

    def stop(_source, _identity, _feature_dir, pose, **_kwargs):
        opened.append(pose._pose_path)
        raise _StopAfterOpen

    monkeypatch.setattr(classify, "IdentityFeatures", stop)
    classifier = mock.Mock(project_settings={"window_size": 5, "social": False})
    with pytest.raises(_StopAfterOpen):
        classify.classify_pose(classifier, pose_path, out, behavior="b", **kwargs)
    return opened[0]


def _legacy_pose(tmp_path: Path) -> Path:
    path = tmp_path / "clip_pose_est_v8.h5"
    shutil.copyfile(
        Path(__file__).parents[2]
        / "packages"
        / "jabs-io"
        / "tests"
        / "data"
        / "legacy-pose"
        / "v8-with-centers.h5",
        path,
    )
    return path


def test_classify_converts_into_a_temporary_directory_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch run does not fill the per-user cache: the conversion goes when the pose closes."""
    user_cache = tmp_path / "user-cache"
    monkeypatch.setattr(conversion_cache, "user_cache_dir", lambda _app: str(user_cache))

    converted = _classify_until_features(monkeypatch, _legacy_pose(tmp_path), tmp_path / "out")

    assert user_cache not in converted.parents
    assert not converted.exists()
    assert not list(user_cache.rglob("*.h5"))


def test_classify_keeps_conversions_in_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a cache directory the conversion is kept there for later runs."""
    cache = tmp_path / "conversions"

    converted = _classify_until_features(
        monkeypatch, _legacy_pose(tmp_path), tmp_path / "out", cache_dir=cache
    )

    assert cache in converted.parents
    assert converted.exists()


def test_classify_cache_dir_option_is_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``jabs-classify classify --cache-dir`` reaches classify_pose."""
    classifier = mock.Mock(
        classifier_type=next(iter(classify.Classifier().classifier_choices())),
        behavior_name="grooming",
        project_settings={"window_size": 5, "social": False, "cm_units": False},
    )
    monkeypatch.setattr(classify, "load_classifier_from_pickle", lambda _path: classifier)
    classify_pose = mock.Mock()
    monkeypatch.setattr(classify, "classify_pose", classify_pose)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jabs-classify",
            "classify",
            "--classifier",
            str(tmp_path / "classifier.pickle"),
            "--input-pose",
            str(tmp_path / "clip_pose_est_v8.h5"),
            "--out-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "conversions"),
        ],
    )

    classify.classify_main()

    assert classify_pose.call_args.kwargs["cache_dir"] == tmp_path / "conversions"
