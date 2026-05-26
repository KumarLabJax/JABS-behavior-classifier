"""Per-mode classification strategies used by :class:`ClassifyThread`.

Two strategies - :class:`BinaryClassifyStrategy` and
:class:`MultiClassClassifyStrategy` - implement the mode-specific pieces of
the classification pipeline (effective settings, the behavior key under which
predictions are persisted, the optional class-name list, the per-identity
probabilities to store, the zero-fill shape used when an identity has no
data, and the optional postprocessing pipeline). The orchestrator in
:class:`ClassifyThread` consumes the strategy and stays mode-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from jabs.behavior.postprocessing import PostprocessingPipeline
from jabs.classifier import Classifier, MultiClassClassifier
from jabs.project.prediction_manager import MULTICLASS_PREDICTION_KEY

if TYPE_CHECKING:
    from jabs.project import Project


class ClassifyStrategy:
    """Per-mode hooks for the classification pipeline."""

    def __init__(
        self,
        classifier: Classifier | MultiClassClassifier,
        project: Project,
        behavior: str,
    ) -> None:
        self._classifier = classifier
        self._project = project
        self._behavior = behavior

    def project_settings(self) -> dict:
        """Return the settings used for feature extraction and inference."""
        raise NotImplementedError

    def prediction_behavior(self) -> str:
        """Return the behavior key under which predictions are persisted."""
        raise NotImplementedError

    def class_names(self) -> list[str] | None:
        """Return the class-name list emitted on completion and saved to disk.

        ``None`` for binary mode, where the saved record uses the behavior key
        and there is no per-class probability matrix.
        """
        raise NotImplementedError

    def probabilities_for_storage(
        self,
        prob: npt.NDArray[np.float32],
        confidence: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Return the per-frame probabilities to persist for this identity.

        Args:
            prob: Full per-class probability matrix from ``predict_proba``,
                shape ``(n_frames, n_classes)``.
            confidence: Per-frame confidence of the chosen class, shape
                ``(n_frames,)``.

        Returns:
            The probabilities to persist - shape depends on mode.
        """
        raise NotImplementedError

    def empty_probabilities(self, num_frames: int) -> npt.NDArray[np.float32]:
        """Return a zero-filled probabilities array for an identity with no data."""
        raise NotImplementedError

    def postprocess_identity(
        self,
        predictions: npt.NDArray[np.int8],
        probabilities: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int8] | None:
        """Postprocessed predictions for this identity, or ``None`` if not applicable."""
        raise NotImplementedError


class BinaryClassifyStrategy(ClassifyStrategy):
    """Classification pipeline for the binary behavior-vs-not-behavior classifier."""

    def __init__(
        self,
        classifier: Classifier,
        project: Project,
        behavior: str,
    ) -> None:
        super().__init__(classifier, project, behavior)
        self._project_settings = project.settings_manager.get_behavior(behavior)
        self._postprocessing_pipeline = PostprocessingPipeline(
            self._project_settings.get("postprocessing", [])
        )

    def project_settings(self) -> dict:
        """Return the behavior-scoped settings from the project settings manager."""
        return self._project_settings

    def prediction_behavior(self) -> str:
        """Return the behavior label used as the prediction record key."""
        return self._behavior

    def class_names(self) -> list[str] | None:
        """Binary mode does not persist class names."""
        return None

    def probabilities_for_storage(
        self,
        prob: npt.NDArray[np.float32],
        confidence: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Binary mode persists the per-frame confidence of the chosen class."""
        return confidence

    def empty_probabilities(self, num_frames: int) -> npt.NDArray[np.float32]:
        """One-dimensional zero array sized to the video's frame count."""
        return np.zeros(num_frames, dtype=np.float32)

    def postprocess_identity(
        self,
        predictions: npt.NDArray[np.int8],
        probabilities: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int8] | None:
        """Run the configured postprocessing pipeline for this identity."""
        return self._postprocessing_pipeline.run(predictions, probabilities)


class MultiClassClassifyStrategy(ClassifyStrategy):
    """Classification pipeline for the multi-class behavior classifier."""

    def __init__(
        self,
        classifier: MultiClassClassifier,
        project: Project,
        behavior: str,
    ) -> None:
        super().__init__(classifier, project, behavior)
        multiclass_classifier = cast(MultiClassClassifier, classifier)
        # Multiclass shares one settings bundle across every behavior; fall back
        # to project defaults if the classifier was constructed without one.
        self._project_settings = (
            multiclass_classifier.project_settings or project.get_project_defaults()
        )
        self._class_names: list[str] = multiclass_classifier.get_class_names()

    def project_settings(self) -> dict:
        """Return the captured project settings used at construction time."""
        return self._project_settings

    def prediction_behavior(self) -> str:
        """Return the reserved multi-class prediction record key."""
        return MULTICLASS_PREDICTION_KEY

    def class_names(self) -> list[str] | None:
        """Return ``[MULTICLASS_NONE_BEHAVIOR, *behavior_names]``."""
        return self._class_names

    def probabilities_for_storage(
        self,
        prob: npt.NDArray[np.float32],
        confidence: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Multi-class mode persists the full per-class probability matrix."""
        return prob

    def empty_probabilities(self, num_frames: int) -> npt.NDArray[np.float32]:
        """Two-dimensional zero array shaped ``(num_frames, n_classes)``."""
        return np.zeros((num_frames, len(self._class_names)), dtype=np.float32)

    def postprocess_identity(
        self,
        predictions: npt.NDArray[np.int8],
        probabilities: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int8] | None:
        """Post-processing semantics are currently binary-only."""
        return None
