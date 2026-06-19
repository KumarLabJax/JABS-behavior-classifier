"""Multi-class behavior classifier for simultaneous N-behavior classification."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import (
    DEFAULT_CV_GROUPING_STRATEGY,
    ClassifierType,
    CrossValidationGroupingStrategy,
    compile_grouping_regex,
    filename_group_key,
)
from jabs.core.utils import hash_file
from jabs.project import load_multiclass_training_data

from . import classifier_utils
from .base import BaseClassifier

logger = logging.getLogger(__name__)


class MultiClassClassifier(BaseClassifier):
    """Multi-class behavior classifier for simultaneous classification of N behaviors.

    Trains a single classifier over all annotated behaviors, outputting one of N
    behavior classes (or ``MULTICLASS_NONE_BEHAVIOR``) per frame. Per-behavior
    TrackLabels arrays are merged into a single class-index array before training.

    The ``MULTICLASS_NONE_BEHAVIOR`` class (index 0) is the multi-class analog of the
    binary classifier's "not {behavior}" class, except that it represents the absence
    of *all* annotated behaviors rather than the negation of a single one. It is
    populated exclusively via explicit ``MULTICLASS_NONE_BEHAVIOR`` labels; frames
    that are simply unlabeled are excluded from training entirely.

    Merging rules:
        - BEHAVIOR in ``MULTICLASS_NONE_BEHAVIOR`` TrackLabels → class 0
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
            behavior_names: Ordered list of behavior names to classify (must not include
                ``MULTICLASS_NONE_BEHAVIOR``). Class index 0 is reserved for
                ``MULTICLASS_NONE_BEHAVIOR``; ``behavior_names[i]`` maps to class index ``i + 1``.
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
        """Ordered list of behavior names (does not include ``MULTICLASS_NONE_BEHAVIOR``)."""
        return list(self._behavior_names)

    def get_class_names(self) -> list[str]:
        """Return the ordered list of class names for this classifier.

        Returns:
            List with ``MULTICLASS_NONE_BEHAVIOR`` at index 0 followed by behavior names.
        """
        return [MULTICLASS_NONE_BEHAVIOR, *self._behavior_names]

    def rename_behavior(self, old_name: str, new_name: str) -> None:
        """Rename a behavior class in place, preserving class-index order.

        The behavior keeps its existing class index, so a previously trained
        model and any saved predictions referencing that index remain valid.

        Args:
            old_name: Current behavior name. Must be one of ``behavior_names``.
            new_name: Replacement name. Must not already be present and must not
                collide with the reserved ``MULTICLASS_NONE_BEHAVIOR`` name.

        Raises:
            ValueError: If ``old_name`` is not a known behavior, if ``new_name``
                is already present, or if ``new_name`` is the reserved name.
        """
        if old_name not in self._behavior_names:
            raise ValueError(f"behavior {old_name!r} is not known to this classifier")
        if new_name == MULTICLASS_NONE_BEHAVIOR:
            raise ValueError(
                f"new behavior name must not be the reserved name {MULTICLASS_NONE_BEHAVIOR!r}"
            )
        if new_name in self._behavior_names:
            raise ValueError(f"behavior {new_name!r} already exists")

        self._behavior_names[self._behavior_names.index(old_name)] = new_name

    def set_project_settings(self, project, behavior: str | None = None) -> None:
        """Copy project defaults as classifier settings.

        Args:
            project: Project to copy default settings from.
            behavior: Unused; accepted only for signature parity with
                ``Classifier.set_project_settings``. Multi-class mode trains a
                single shared classifier with no behavior to scope to, so
                project-level defaults are always used.
        """
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
            behaviors. Column 0 is the ``MULTICLASS_NONE_BEHAVIOR`` class.
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

    @classmethod
    def from_pickle(cls, path: Path) -> MultiClassClassifier:
        """Load a MultiClassClassifier from a pickle file and log the source.

        Thin wrapper over :meth:`BaseClassifier.from_pickle` that adds an
        informational log entry; all validation lives in the base class.
        """
        classifier = super().from_pickle(path)
        logger.info("MultiClassClassifier loaded from %s", path)
        return classifier

    @classmethod
    def from_training_file(
        cls, path: Path, classifier_type: ClassifierType | None = None
    ) -> MultiClassClassifier:
        """Train a new MultiClassClassifier from an exported training file.

        Args:
            path: Path to a multi-class training HDF5 file produced by
                ``export_training_data_multiclass()``.
            classifier_type: Override the classifier algorithm stored in the training
                file. If ``None``, the type recorded in the file is used.

        Returns:
            A freshly trained ``MultiClassClassifier`` instance.

        Raises:
            ValueError: If the file is not a valid multi-class training export or
                the effective classifier type is unsupported in the current environment.
        """
        loaded, _ = load_multiclass_training_data(path)

        effective_type = (
            classifier_type if classifier_type is not None else loaded["classifier_type"]
        )
        classifier = cls(
            behavior_names=loaded["behavior_names"],
            classifier_type=effective_type,
        )
        classifier.set_dict_settings(loaded["settings"])
        classifier.train(
            {
                "per_frame": loaded["per_frame"],
                "window": loaded["window"],
                "labels_by_behavior": loaded["labels_by_behavior"],
            },
            random_seed=loaded["training_seed"],
        )
        classifier._classifier_file = Path(path).name
        classifier._classifier_hash = hash_file(Path(path))
        classifier._classifier_source = "training_file"
        return classifier

    @staticmethod
    def leave_one_group_out(
        per_frame_features: pd.DataFrame,
        window_features: pd.DataFrame,
        labels: npt.NDArray,
        groups: npt.NDArray,
        excluded_groups: set[int] | None = None,
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
            excluded_groups: Group ids held out of training (eligible as the test
                group, but never part of a training fold).

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
            excluded_groups=excluded_groups,
        )

    @staticmethod
    def get_leave_one_group_out_max(
        labels: npt.NDArray,
        groups: npt.NDArray,
        excluded_groups: set[int] | None = None,
    ) -> int:
        """Count the number of valid LOGO splits for multi-class CV.

        A group is counted as a valid test split when it contains at least 2
        distinct classes above ``LABEL_THRESHOLD`` and the remaining training
        groups collectively contain all classes above ``LABEL_THRESHOLD``.

        Args:
            labels: Multi-class label array (class indices).
            groups: Group ID array corresponding to each label.
            excluded_groups: Group ids held out of training (eligible as the test
                group, but never part of a training fold).

        Returns:
            Number of groups that can serve as a valid test split.
        """
        return classifier_utils.count_valid_logo_splits(
            labels,
            groups,
            label_threshold=MultiClassClassifier.LABEL_THRESHOLD,
            min_test_classes=2,
            excluded_groups=excluded_groups,
        )

    @staticmethod
    def count_label_threshold(
        counts_by_behavior: dict[str, dict],
        behavior_names: list[str],
        cv_grouping_strategy: CrossValidationGroupingStrategy = DEFAULT_CV_GROUPING_STRATEGY,
        cv_grouping_regex: str | None = None,
    ) -> int:
        """Count multi-class LOGO groups that satisfy the relaxed acceptance rule.

        A group is counted as a valid test split when it contains at least 2
        distinct classes above ``LABEL_THRESHOLD`` and the remaining training
        groups collectively contain all classes above ``LABEL_THRESHOLD``.

        Args:
            counts_by_behavior: Maps each class name to its labeled-frame count
                dict (the structure returned by ``Project.counts(name)``),
                shaped as ``dict[video_name][identity]`` of fragmented and
                unfragmented frame/bout count tuples.
            behavior_names: Ordered class names whose counts appear in
                ``counts_by_behavior``. Typically includes
                ``MULTICLASS_NONE_BEHAVIOR``.
            cv_grouping_strategy: Cross-validation grouping strategy.
            cv_grouping_regex: Regex used to extract a grouping key from each video
                filename when ``cv_grouping_strategy`` is ``FILENAME_PATTERN``. An
                empty or invalid regex yields a count of 0 (no trainable groups).

        Returns:
            Number of groups that can serve as a valid multi-class LOGO test split.

        Note:
            Uses "fragmented" label counts since these reflect labels usable
            for training.
        """
        if not behavior_names:
            return 0

        # FILENAME_PATTERN aggregates like VIDEO grouping, but keys each group by
        # the regex-extracted filename key instead of the video name (so several
        # videos can share one group). An empty/invalid regex means no groups.
        pattern = None
        if cv_grouping_strategy == CrossValidationGroupingStrategy.FILENAME_PATTERN:
            try:
                pattern = compile_grouping_regex(cv_grouping_regex or "")
            except ValueError:
                return 0

        threshold = MultiClassClassifier.LABEL_THRESHOLD
        group_class_counts: dict[tuple[str, int] | str, dict[str, int]] = {}
        for behavior_name in behavior_names:
            behavior_counts = counts_by_behavior.get(behavior_name, {})
            for video_name, video_counts in behavior_counts.items():
                if cv_grouping_strategy in (
                    CrossValidationGroupingStrategy.VIDEO,
                    CrossValidationGroupingStrategy.FILENAME_PATTERN,
                ):
                    key: tuple[str, int] | str = (
                        filename_group_key(video_name, pattern)
                        if pattern is not None
                        else video_name
                    )
                    group_entry = group_class_counts.setdefault(key, {})
                    group_entry[behavior_name] = group_entry.get(behavior_name, 0) + sum(
                        identity_counts["fragmented_frame_counts"][0]
                        for identity_counts in video_counts.values()
                    )
                else:
                    for identity, identity_counts in video_counts.items():
                        key = (video_name, int(identity))
                        group_entry = group_class_counts.setdefault(key, {})
                        group_entry[behavior_name] = identity_counts["fragmented_frame_counts"][0]

        if not group_class_counts:
            return 0

        total_by_class = {
            class_name: sum(
                group_counts.get(class_name, 0) for group_counts in group_class_counts.values()
            )
            for class_name in behavior_names
        }

        valid_groups = 0
        for group_counts in group_class_counts.values():
            n_test_classes = sum(
                group_counts.get(class_name, 0) >= threshold for class_name in behavior_names
            )
            train_has_all_classes = all(
                (total_by_class[class_name] - group_counts.get(class_name, 0)) >= threshold
                for class_name in behavior_names
            )
            if n_test_classes >= 2 and train_has_all_classes:
                valid_groups += 1

        return valid_groups

    @staticmethod
    def label_threshold_met(
        counts_by_behavior: dict[str, dict],
        behavior_names: list[str],
        min_groups: int,
        cv_grouping_strategy: CrossValidationGroupingStrategy = DEFAULT_CV_GROUPING_STRATEGY,
        cv_grouping_regex: str | None = None,
    ) -> bool:
        """Determine whether multi-class labels support ``min_groups`` LOGO splits.

        Args:
            counts_by_behavior: Maps each class name to its labeled-frame count
                dict (see :meth:`count_label_threshold`).
            behavior_names: Ordered class names whose counts appear in
                ``counts_by_behavior``. Returns ``False`` when fewer than two
                class names are supplied.
            min_groups: Minimum number of valid LOGO splits required. Floored
                at 1, since multi-class training requires at least one valid
                split.
            cv_grouping_strategy: Cross-validation grouping strategy.
            cv_grouping_regex: Regex used for ``FILENAME_PATTERN`` grouping (see
                :meth:`count_label_threshold`).

        Returns:
            True if the count of valid splits meets ``max(1, min_groups)``.
        """
        if len(behavior_names) < 2:
            return False
        valid_splits = MultiClassClassifier.count_label_threshold(
            counts_by_behavior=counts_by_behavior,
            behavior_names=behavior_names,
            cv_grouping_strategy=cv_grouping_strategy,
            cv_grouping_regex=cv_grouping_regex,
        )
        return valid_splits >= max(1, min_groups)
