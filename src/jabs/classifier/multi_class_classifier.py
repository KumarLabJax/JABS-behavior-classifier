"""Multi-class behavior classifier for simultaneous N-behavior classification."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Generator
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierType

from . import classifier_utils
from .base import BaseClassifier

logger = logging.getLogger(__name__)


class MultiClassClassifier(BaseClassifier):
    """Multi-class behavior classifier for simultaneous classification of N behaviors.

    Trains a single classifier over all annotated behaviors, outputting one of N
    behavior classes (or background) per frame. Per-behavior TrackLabels arrays
    are merged into a single class-index array before training.

    The background class (index 0) is the multi-class analog of the binary
    classifier's "not {behavior}" class, except that it represents the absence
    of *all* annotated behaviors rather than the negation of a single one. It
    is populated exclusively via explicit ``MULTICLASS_NONE_BEHAVIOR`` labels;
    frames that are simply unlabeled are excluded from training entirely.

    Merging rules:
        - BEHAVIOR in ``MULTICLASS_NONE_BEHAVIOR`` TrackLabels → class 0 (background)
        - BEHAVIOR in behavior X's TrackLabels → class X's 1-based index
        - All other frames → excluded from training

    Cross-validation note:
        Leave-one-group-out CV uses a relaxed split criterion compared to
        binary mode. Because individual groups (videos or animals) are often
        labeled for only a subset of behaviors, requiring all classes in every
        test split would yield no valid splits. Instead, a test split is
        accepted when it contains at least 2 classes above ``LABEL_THRESHOLD``
        and the remaining training groups collectively contain all classes
        above ``LABEL_THRESHOLD``. A follow-up ticket should introduce a
        multi-video grouping strategy that aggregates groups to improve class
        coverage in test splits.

    Attributes:
        LABEL_THRESHOLD: Minimum number of labeled frames required per class.
    """

    LABEL_THRESHOLD: ClassVar[int] = classifier_utils.LABEL_THRESHOLD

    _VERSION: ClassVar[int] = 1
    _MULTICLASS: ClassVar[bool] = True
    _PERSISTED_REQUIRED: ClassVar[tuple[str, ...]] = (
        "_classifier",
        "_behavior_names",
        "_classifier_type",
        "_feature_names",
    )
    _PERSISTED_OPTIONAL: ClassVar[tuple[str, ...]] = ("_project_settings",)

    def __init__(
        self,
        behavior_names: list[str],
        classifier_type: ClassifierType = ClassifierType.RANDOM_FOREST,
        n_jobs: int = 1,
    ) -> None:
        """Initialize a MultiClassClassifier.

        Args:
            behavior_names: Ordered list of behavior names to classify (must
                not include ``MULTICLASS_NONE_BEHAVIOR``). Class index 0 is
                reserved for background; ``behavior_names[i]`` maps to class
                index ``i + 1``.
            classifier_type: Underlying algorithm to use.
            n_jobs: Number of parallel jobs for training and inference.

        Raises:
            ValueError: If ``behavior_names`` is empty, contains duplicates,
                or includes the reserved name ``MULTICLASS_NONE_BEHAVIOR``.
            ValueError: If ``classifier_type`` is not supported in the current
                environment.
        """
        if not behavior_names:
            raise ValueError("behavior_names must not be empty")
        if MULTICLASS_NONE_BEHAVIOR in behavior_names:
            raise ValueError(
                f"behavior_names must not include the reserved name {MULTICLASS_NONE_BEHAVIOR!r}"
            )
        if len(behavior_names) != len(set(behavior_names)):
            raise ValueError("behavior_names must not contain duplicate entries")

        super().__init__(classifier_type=classifier_type, n_jobs=n_jobs)
        self._behavior_names: list[str] = list(behavior_names)

    @property
    def behavior_names(self) -> list[str]:
        """Ordered list of behavior names (does not include background)."""
        return list(self._behavior_names)

    def get_class_names(self) -> list[str]:
        """Return the ordered class names, with background at index 0."""
        return ["background", *self._behavior_names]

    def set_project_settings(self, project) -> None:
        """Copy project defaults as classifier settings."""
        self._project_settings = dict(project.get_project_defaults())

    def train(self, data: dict, random_seed: int | None = None) -> None:
        """Train the multi-class classifier.

        Required keys in ``data``:
            - ``per_frame``: per-frame feature DataFrame (one row per labeled frame)
            - ``window``: window feature DataFrame (same shape as per_frame)
            - ``labels_by_behavior``: dict mapping behavior name → label array of
              ``TrackLabels.Label`` values, one element per row in ``per_frame``

        Optional keys:
            - ``settings``: dict with ``symmetric_behavior`` (bool) and
              ``balance_labels`` (bool), both default False
            - ``feature_names``: explicit list of feature column names

        Args:
            data: Training data dictionary (see above).
            random_seed: Optional random seed for reproducibility.

        Raises:
            ValueError: If required keys are missing from ``data``.
        """
        for key in ("per_frame", "window", "labels_by_behavior"):
            if key not in data:
                raise ValueError(f"Missing required key in training data: '{key}'")

        settings = data.get("settings", self._project_settings or {})
        # Persist the effective training settings so downstream classification
        # consistently reuses the same feature/postprocessing parameters.
        self._project_settings = dict(settings)

        multiclass_labels, include_mask = classifier_utils.merge_labels(
            data["labels_by_behavior"], self._behavior_names
        )

        n_classes_present = len(np.unique(multiclass_labels))
        if n_classes_present < 2:
            raise ValueError(
                f"Training requires at least 2 distinct classes, but only "
                f"{n_classes_present} found. Ensure that at least two behaviors have "
                f"BEHAVIOR-labeled frames, or combine one behavior with explicit "
                f"background labels via the '{MULTICLASS_NONE_BEHAVIOR}' behavior entry."
            )

        combined = classifier_utils.combine_data(data["per_frame"], data["window"])
        features = combined[include_mask].reset_index(drop=True)

        if "feature_names" in data:
            self._feature_names = data["feature_names"]
            features = features[self._feature_names]
        else:
            self._feature_names = features.columns.to_list()

        if settings.get("symmetric_behavior", False):
            features, multiclass_labels = classifier_utils.augment_symmetric(
                features, multiclass_labels
            )
        if settings.get("balance_labels", False):
            features, multiclass_labels = classifier_utils.downsample_balance(
                features, multiclass_labels, random_seed
            )

        clf = self._create_classifier(random_seed=random_seed)
        cleaned = self._clean_features(features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self._classifier = clf.fit(cleaned, multiclass_labels)

        self._classifier_file = None
        self._classifier_hash = None
        self._classifier_source = None
        logger.info("MultiClassClassifier trained on %d behaviors", len(self._behavior_names))

    def predict(
        self,
        features: pd.DataFrame,
        frame_indexes: npt.NDArray[np.intp] | None = None,
    ) -> npt.NDArray[np.int8]:
        """Predict class indices for the given features.

        Args:
            features: DataFrame of feature data.
            frame_indexes: Indexes of frames with valid pose data. Frames
                absent from this array receive a prediction of -1 (no pose).

        Returns:
            Integer array of shape ``(n_frames,)`` with class indices 0..N,
            or -1 for frames with no pose data.
        """
        cleaned = self._get_features_to_classify(self._clean_features(features))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            # ravel() normalizes CatBoost MultiClass, which returns (n, 1).
            result = np.ravel(self._classifier.predict(cleaned)).astype(np.int8)

        if frame_indexes is not None:
            result_adjusted = np.full(result.shape, -1, dtype=np.int8)
            result_adjusted[frame_indexes] = result[frame_indexes]
            result = result_adjusted

        return result

    def predict_proba(
        self,
        features: pd.DataFrame,
        frame_indexes: npt.NDArray[np.intp] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Predict class probabilities for the given features.

        Args:
            features: DataFrame of feature data.
            frame_indexes: Indexes of frames with valid pose data. Frames
                absent from this array receive zero probability across all
                classes.

        Returns:
            Float array of shape ``(n_frames, N+1)`` where N is the number of
            behaviors. Column 0 is the background class.
        """
        cleaned = self._get_features_to_classify(self._clean_features(features))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            result = self._classifier.predict_proba(cleaned).astype(np.float32)

        if frame_indexes is not None:
            result_adjusted = np.zeros(result.shape, dtype=np.float32)
            result_adjusted[frame_indexes] = result[frame_indexes]
            result = result_adjusted

        return result

    def save(self, path):
        """Serialize the classifier to disk and log the destination."""
        super().save(path)
        logger.info("MultiClassClassifier saved to %s", path)

    def load(self, path):
        """Deserialize a classifier from disk and log the source."""
        super().load(path)
        logger.info("MultiClassClassifier loaded from %s", path)

    @staticmethod
    def leave_one_group_out(
        per_frame_features: pd.DataFrame,
        window_features: pd.DataFrame,
        labels: npt.NDArray,
        groups: npt.NDArray,
    ) -> Generator[dict, None, None]:
        """Yield leave-one-group-out splits for multi-class cross-validation.

        Uses a relaxed acceptance criterion: a split is valid when the test
        group has at least 2 classes above ``LABEL_THRESHOLD`` and the
        training portion has all classes above ``LABEL_THRESHOLD``. See the
        class docstring for rationale.

        Args:
            per_frame_features: Per-frame feature DataFrame for labeled data.
            window_features: Window feature DataFrame for labeled data.
            labels: Multi-class label array (class indices).
            groups: Group ID array corresponding to each feature row.

        Yields:
            Split dictionaries with keys: training_data, training_labels,
            test_data, test_labels, test_group, feature_names.

        Raises:
            ValueError: If no valid split can be found.
        """
        yield from classifier_utils.leave_one_group_out(
            per_frame_features,
            window_features,
            labels,
            groups,
            label_threshold=MultiClassClassifier.LABEL_THRESHOLD,
            min_test_classes=2,
        )

    @staticmethod
    def get_leave_one_group_out_max(
        labels: npt.NDArray,
        groups: npt.NDArray,
    ) -> int:
        """Count the number of valid LOGO splits for multi-class CV.

        A group is counted as a valid test split when it contains at least 2
        distinct classes above ``LABEL_THRESHOLD`` and the remaining training
        groups collectively contain all classes above ``LABEL_THRESHOLD``.

        Args:
            labels: Multi-class label array (class indices).
            groups: Group ID array corresponding to each label.

        Returns:
            Number of groups that can serve as a valid test split.
        """
        all_classes = np.unique(labels)
        unique_groups = np.unique(groups)
        count = 0
        for g in unique_groups:
            test_mask = np.asarray(groups) == g
            test_labels = np.asarray(labels)[test_mask]
            train_labels = np.asarray(labels)[~test_mask]

            n_test_classes = sum(
                np.count_nonzero(test_labels == cls) >= MultiClassClassifier.LABEL_THRESHOLD
                for cls in all_classes
            )
            train_has_all = all(
                np.count_nonzero(train_labels == cls) >= MultiClassClassifier.LABEL_THRESHOLD
                for cls in all_classes
            )
            if n_test_classes >= 2 and train_has_all:
                count += 1
        return count
