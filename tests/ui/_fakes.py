"""Shared test doubles for UI thread tests (TrainingThread, ClassifyThread).

The classifier and project test doubles diverge by which thread they support
(training vs. classification), so they remain separate classes. Co-locating
them here keeps test files focused on their assertions and provides a single
spot to extend the fakes as more UI thread tests are added.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd

from jabs.core.enums import (
    ClassifierMode,
    CrossValidationGroupingStrategy,
    ProjectDistanceUnit,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Classifier fakes
# ---------------------------------------------------------------------------


class FakeTrainingClassifier:
    """Classifier test double exposing the API ``TrainingThread`` calls.

    Records ``train(...)`` calls for assertion; ``get_feature_importance`` and
    ``combine_data`` return deterministic placeholders.
    """

    def __init__(
        self,
        name: str = "random_forest",
        project_settings: dict | None = None,
    ) -> None:
        self.classifier_name = name
        self.project_settings = {} if project_settings is None else dict(project_settings)
        self.behavior_names: list[str] = ["Walk", "Run"]
        self.train_calls: list[dict] = []

    @staticmethod
    def combine_data(per_frame: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
        """Concatenate per-frame and window features side-by-side."""
        return pd.concat([per_frame, window], axis=1)

    def train(self, data: dict, random_seed: int | None = None) -> None:
        """Record the call payload and seed for later assertion."""
        call = dict(data)
        call["random_seed"] = random_seed
        self.train_calls.append(call)

    @staticmethod
    def get_feature_importance(limit: int = 20) -> list[tuple[str, float]]:
        """Return a placeholder feature-importance list capped at ``limit``."""
        return [("feat_a", 1.0)][:limit]


class FakeClassifyingClassifier:
    """Classifier test double exposing the API ``ClassifyThread`` calls.

    ``predict_proba`` returns a fixed probability matrix shaped by ``multiclass``
    (binary: ``(n, 2)``; multi-class: ``(n, 3)``). ``derive_predictions`` argmax-es
    that matrix and zeroes out frames with no pose data.
    """

    def __init__(self, multiclass: bool = False) -> None:
        self._multiclass = multiclass
        self.project_settings: dict = {"window_size": 7}
        self.behavior_names: list[str] = ["Walk", "Run"]

    @staticmethod
    def combine_data(per_frame: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
        """Concatenate per-frame and window features side-by-side."""
        return pd.concat([per_frame, window], axis=1)

    def predict_proba(
        self,
        data: pd.DataFrame,
        frame_indexes: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float32]:
        """Return a fixed probability matrix; zeroes rows with frame_indexes == -1."""
        n = len(data)
        if self._multiclass:
            probs = np.zeros((n, 3), dtype=np.float32)
            probs[:, 0] = 0.1
            probs[:, 1] = 0.7
            probs[:, 2] = 0.2
            probs[frame_indexes == -1] = 0.0
            return probs

        probs = np.zeros((n, 2), dtype=np.float32)
        probs[:, 0] = 0.2
        probs[:, 1] = 0.8
        return probs

    def derive_predictions(
        self,
        probabilities: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float32]]:
        """Argmax-derived predictions and per-frame confidence, with -1 on empties."""
        predictions = np.argmax(probabilities, axis=1).astype(np.int8)
        confidence = probabilities[np.arange(len(probabilities)), predictions].astype(np.float32)
        predictions[confidence == 0] = -1
        return predictions, confidence

    @staticmethod
    def get_class_names() -> list[str]:
        """Return the ordered class names for multi-class persistence."""
        return ["background", "Walk", "Run"]


# ---------------------------------------------------------------------------
# Project fakes
# ---------------------------------------------------------------------------


class FakeTrainingProject:
    """Project test double exposing the API ``TrainingThread`` calls.

    Counts calls to ``get_labeled_features`` / ``get_multiclass_labeled_features``
    so tests can assert which mode path ran. ``save_classifier`` is a ``MagicMock``
    so call args can be checked.
    """

    _DEFAULT_BEHAVIOR_SETTINGS: ClassVar[dict] = {
        "window_size": 5,
        "balance_labels": False,
        "symmetric_behavior": False,
    }

    def __init__(
        self,
        tmp_path: Path,
        mode: ClassifierMode,
        binary_features: dict | None = None,
        multiclass_features: dict | None = None,
    ) -> None:
        self.project_paths = SimpleNamespace(training_log_dir=tmp_path)
        self.feature_manager = SimpleNamespace(distance_unit=ProjectDistanceUnit.PIXEL)
        self.session_tracker = SimpleNamespace(classifier_trained=MagicMock())
        self.settings_manager = SimpleNamespace(
            classifier_mode=mode,
            cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL,
            get_behavior=lambda _behavior: dict(self._DEFAULT_BEHAVIOR_SETTINGS),
        )
        self._binary_features = binary_features
        self._multiclass_features = multiclass_features
        self.binary_calls = 0
        self.multiclass_calls = 0
        self.save_classifier = MagicMock()

    def get_project_defaults(self) -> dict:
        """Return the default project settings used when classifier has none."""
        return dict(self._DEFAULT_BEHAVIOR_SETTINGS)

    def get_labeled_features(
        self,
        behavior: str,
        progress_callable=None,
        should_terminate_callable=None,
    ) -> tuple[dict, dict]:
        """Return the binary feature payload and a single-identity group mapping."""
        self.binary_calls += 1
        if should_terminate_callable is not None:
            should_terminate_callable()
        if progress_callable is not None:
            progress_callable()
        return self._binary_features, {0: {"video": "video.avi", "identity": 0}}

    def get_multiclass_labeled_features(
        self,
        progress_callable=None,
        should_terminate_callable=None,
        grouping_strategy=None,
        behavior_settings=None,
    ) -> tuple[dict, dict]:
        """Return the multi-class feature payload and an empty group mapping."""
        self.multiclass_calls += 1
        if should_terminate_callable is not None:
            should_terminate_callable()
        if progress_callable is not None:
            progress_callable()
        return self._multiclass_features, {}

    @staticmethod
    def counts(_behavior: str) -> dict:
        """Return a single-video, single-identity bout-count entry."""
        return {"video.avi": {0: {"unfragmented_bout_counts": (1, 0)}}}


class FakeClassifyingProject:
    """Project test double exposing the API ``ClassifyThread`` calls.

    ``save_predictions`` is a ``MagicMock`` so tests can assert how predictions
    were persisted; ``load_pose_est`` returns a fixed five-frame pose stand-in.
    """

    def __init__(self, mode: ClassifierMode) -> None:
        self.settings_manager = SimpleNamespace(
            classifier_mode=mode,
            get_behavior=lambda _behavior: {"window_size": 5, "postprocessing": []},
        )
        self.feature_manager = SimpleNamespace(distance_unit=ProjectDistanceUnit.PIXEL)
        self.video_manager = SimpleNamespace(
            videos=["video.avi"],
            video_path=lambda _video: "video.avi",
            num_videos=1,
        )
        self.feature_dir = "feature_dir"
        self.cache_format = "hdf5"
        self.save_predictions = MagicMock()
        self._pose = SimpleNamespace(
            identities=[0],
            num_identities=1,
            num_frames=5,
            fps=30,
        )

    def load_pose_est(self, _video_path) -> SimpleNamespace:
        """Return the fixed pose-estimation stand-in."""
        return self._pose

    @staticmethod
    def get_project_defaults() -> dict:
        """Return the default project settings used when classifier has none."""
        return {"window_size": 9}


# ---------------------------------------------------------------------------
# IdentityFeatures factory
# ---------------------------------------------------------------------------


def make_fake_identity_features() -> type:
    """Return a fresh ``IdentityFeatures`` stand-in class with isolated state.

    Tests typically ``monkeypatch.setattr(..., make_fake_identity_features())``
    onto the production import site. Each call returns a fresh class so the
    ``op_settings_seen`` record does not leak between tests.

    The returned class exposes the same surface ``ClassifyThread`` uses:
    ``__init__(*args, **kwargs)`` records the ``op_settings`` kwarg, and
    ``get_features(window_size)`` returns a fixed 5-frame feature payload.
    """

    class FakeIdentityFeatures:
        op_settings_seen: list[dict] = []  # noqa: RUF012 - per-class fresh state

        def __init__(self, *_args, **kwargs) -> None:
            self.__class__.op_settings_seen.append(kwargs.get("op_settings", {}))

        @staticmethod
        def get_features(_window_size: int) -> dict:
            return {
                "per_frame": {"a": np.arange(5, dtype=np.float32)},
                "window": {"b": np.arange(5, dtype=np.float32)},
                "frame_indexes": np.arange(5, dtype=np.intp),
            }

    return FakeIdentityFeatures
