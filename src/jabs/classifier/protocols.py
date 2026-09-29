"""Protocol definitions for JABS classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class ClassifierProtocol(Protocol):
    """Protocol defining the shared interface for JABS classifiers.

    Both binary and multi-class classifiers implement this interface,
    allowing consumers to type-hint against either classifier type without
    isinstance checks.
    """

    @property
    def feature_names(self) -> list[str] | None:
        """Return the list of feature names used during training."""
        ...

    def train(self, data: dict, random_seed: int | None = None) -> None:
        """Train the classifier on labeled data.

        Args:
            data: Dictionary containing training_data, training_labels,
                and optionally feature_names.
            random_seed: Optional random seed for reproducibility.
        """
        ...

    def predict(
        self,
        features: pd.DataFrame,
        frame_indexes: npt.NDArray[np.intp] | None = None,
    ) -> npt.NDArray:
        """Predict class labels for the given features.

        Args:
            features: DataFrame of feature data to classify.
            frame_indexes: Optional frame indexes to restrict prediction to.

        Returns:
            Predicted class label array.
        """
        ...

    def predict_proba(
        self,
        features: pd.DataFrame,
        frame_indexes: npt.NDArray[np.intp] | None = None,
    ) -> npt.NDArray:
        """Predict class probabilities for the given features.

        Args:
            features: DataFrame of feature data to classify.
            frame_indexes: Optional frame indexes to restrict prediction to.

        Returns:
            Predicted probability matrix.
        """
        ...

    @staticmethod
    def combine_data(per_frame: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
        """Combine per-frame and window feature DataFrames into one.

        Args:
            per_frame: Per-frame feature DataFrame.
            window: Window feature DataFrame.

        Returns:
            The combined feature DataFrame.
        """
        ...

    @staticmethod
    def derive_predictions(
        probabilities: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.floating]]:
        """Derive class predictions and confidence from class probabilities.

        Args:
            probabilities: Array of shape ``(n_frames, n_classes)`` of predicted
                class probabilities.

        Returns:
            Tuple ``(predictions, confidence)``.
        """
        ...

    def save(self, path: Path) -> None:
        """Serialize the classifier to disk.

        Args:
            path: Destination file path.
        """
        ...

    def load(self, path: Path) -> None:
        """Deserialize the classifier from disk, updating this instance.

        Args:
            path: Source file path.
        """
        ...
