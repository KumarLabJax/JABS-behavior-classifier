"""Multi-class behavior classifier for simultaneous N-behavior classification."""

from __future__ import annotations

import logging
import typing
import warnings
from collections.abc import Generator
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierType
from jabs.core.utils import hash_file
from jabs.project import TrackLabels

from . import classifier_utils
from .factories import make_catboost, make_random_forest, make_xgboost

logger = logging.getLogger(__name__)

_VERSION = 1

_CLASSIFIER_FACTORIES: dict[ClassifierType, typing.Callable[[int, int | None], typing.Any]] = {
    ClassifierType.RANDOM_FOREST: make_random_forest,
    ClassifierType.CATBOOST: make_catboost,
}

try:
    import xgboost  # noqa: F401
except ImportError:
    logger.warning(
        "Unable to import xgboost. XGBoost support will be unavailable. "
        "You may need to install xgboost and/or libomp."
    )
else:
    _CLASSIFIER_FACTORIES[ClassifierType.XGBOOST] = make_xgboost


class MultiClassClassifier:
    """Multi-class behavior classifier for simultaneous classification of N behaviors.

    Trains a single classifier over all annotated behaviors, outputting one of N
    behavior classes (or background) per frame. Per-behavior TrackLabels arrays are
    merged into a single class-index array before training.

    The background class (index 0) is the multi-class analog of the binary
    classifier's "not {behavior}" class, except that it represents the absence of
    *all* annotated behaviors rather than the negation of a single one. It is
    populated exclusively via explicit ``MULTICLASS_NONE_BEHAVIOR`` labels; frames
    that are simply unlabeled are excluded from training entirely.

    Merging rules:
        - BEHAVIOR in ``MULTICLASS_NONE_BEHAVIOR`` TrackLabels → class 0 (background)
        - BEHAVIOR in behavior X's TrackLabels → class X's 1-based index
        - All other frames → excluded from training

    Cross-validation note:
        Leave-one-group-out CV uses a relaxed split criterion compared to binary
        mode. Because individual groups (videos or animals) are often labeled for
        only a subset of behaviors, requiring all classes in every test split would
        yield no valid splits. Instead, a test split is accepted when it contains at
        least 2 classes above ``LABEL_THRESHOLD`` and the remaining training groups
        collectively contain all classes above ``LABEL_THRESHOLD``. A follow-up
        ticket should introduce a multi-video grouping strategy that aggregates
        groups to improve class coverage in test splits.

    Attributes:
        LABEL_THRESHOLD: Minimum number of labeled frames required per class.
    """

    LABEL_THRESHOLD: int = classifier_utils.LABEL_THRESHOLD

    def __init__(
        self,
        behavior_names: list[str],
        classifier_type: ClassifierType = ClassifierType.RANDOM_FOREST,
        n_jobs: int = 1,
    ) -> None:
        """Initialize a MultiClassClassifier.

        Args:
            behavior_names: Ordered list of behavior names to classify (must not include
                ``MULTICLASS_NONE_BEHAVIOR``). Class index 0 is reserved for background;
                ``behavior_names[i]`` maps to class index ``i + 1``.
            classifier_type: Underlying algorithm to use.
            n_jobs: Number of parallel jobs for training and inference.

        Raises:
            ValueError: If ``classifier_type`` is not supported in the current environment.
        """
        self._behavior_names: list[str] = list(behavior_names)
        self._classifier_type = classifier_type
        self._n_jobs = n_jobs
        self._classifier = None
        self._feature_names: list[str] | None = None
        self._version = _VERSION
        self._classifier_file: str | None = None
        self._classifier_hash: str | None = None
        self._classifier_source: str | None = None

        if classifier_type not in self._supported_classifier_choices():
            raise ValueError("Invalid classifier type")

    @property
    def behavior_names(self) -> list[str]:
        """Ordered list of behavior names (does not include background)."""
        return list(self._behavior_names)

    @property
    def feature_names(self) -> list[str] | None:
        """Feature names used when training this classifier."""
        return self._feature_names

    @property
    def classifier_type(self) -> ClassifierType:
        """Underlying classifier algorithm."""
        return self._classifier_type

    def get_class_names(self) -> list[str]:
        """Return the ordered list of class names for this classifier.

        Returns:
            List with ``"background"`` at index 0 followed by behavior names.
        """
        return ["background", *self._behavior_names]

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

        settings = data.get("settings", {})

        multiclass_labels, include_mask = self.merge_labels(
            data["labels_by_behavior"], self._behavior_names
        )

        combined = classifier_utils.combine_data(data["per_frame"], data["window"])
        features = combined[include_mask].reset_index(drop=True)

        if "feature_names" in data:
            self._feature_names = data["feature_names"]
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
        cleaned = classifier_utils.clean_features(features, self._classifier_type)
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
            frame_indexes: Indexes of frames with valid pose data. Frames absent
                from this array receive a prediction of -1 (no pose).

        Returns:
            Integer array of shape ``(n_frames,)`` with class indices 0..N,
            or -1 for frames with no pose data.
        """
        cleaned = self._get_features_to_classify(
            classifier_utils.clean_features(features, self._classifier_type)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            result = self._classifier.predict(cleaned).astype(np.int8)

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
            frame_indexes: Indexes of frames with valid pose data. Frames absent
                from this array receive zero probability across all classes.

        Returns:
            Float array of shape ``(n_frames, N+1)`` where N is the number of
            behaviors. Column 0 is the background class.
        """
        cleaned = self._get_features_to_classify(
            classifier_utils.clean_features(features, self._classifier_type)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            result = self._classifier.predict_proba(cleaned).astype(np.float32)

        if frame_indexes is not None:
            result_adjusted = np.zeros(result.shape, dtype=np.float32)
            result_adjusted[frame_indexes] = result[frame_indexes]
            result = result_adjusted

        return result

    def save(self, path: Path) -> None:
        """Serialize the classifier to disk using joblib.

        Args:
            path: Destination file path.
        """
        joblib.dump(self, path)
        if self._classifier_file is None:
            self._classifier_file = Path(path).name
            self._classifier_hash = hash_file(Path(path))
            self._classifier_source = "serialized"
        logger.info("MultiClassClassifier saved to %s", path)

    def load(self, path: Path) -> None:
        """Deserialize a classifier from disk, updating this instance in place.

        Args:
            path: Source file path.

        Raises:
            ValueError: If the file is not a ``MultiClassClassifier``, was saved
                with a different version, or uses an unsupported classifier type.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            c = joblib.load(path)
            for warning in caught_warnings:
                if issubclass(warning.category, InconsistentVersionWarning):
                    raise ValueError("Classifier trained with different version of sklearn.")
                else:
                    warnings.warn(warning.message, warning.category, stacklevel=2)

        if not isinstance(c, MultiClassClassifier):
            raise ValueError(f"{path} is not an instance of MultiClassClassifier")

        if c._version != _VERSION:
            raise ValueError(
                f"Unable to deserialize pickled classifier. "
                f"File version {c._version}, expected {_VERSION}."
            )

        if c._classifier_type not in self._supported_classifier_choices():
            raise ValueError("Invalid classifier type")

        self._classifier = c._classifier
        self._behavior_names = c._behavior_names
        self._classifier_type = c._classifier_type
        self._feature_names = c._feature_names
        if c._classifier_file is not None:
            self._classifier_file = c._classifier_file
            self._classifier_hash = c._classifier_hash
            self._classifier_source = c._classifier_source
        else:
            self._classifier_file = Path(path).name
            self._classifier_hash = hash_file(Path(path))
            self._classifier_source = "pickle"

        logger.info("MultiClassClassifier loaded from %s", path)

    @staticmethod
    def leave_one_group_out(
        per_frame_features: pd.DataFrame,
        window_features: pd.DataFrame,
        labels: npt.NDArray,
        groups: npt.NDArray,
    ) -> Generator[dict, None, None]:
        """Generate leave-one-group-out splits for multi-class cross-validation.

        Uses a relaxed acceptance criterion: a split is valid when the test group
        has at least 2 classes above ``LABEL_THRESHOLD`` and the training portion
        has all classes above ``LABEL_THRESHOLD``. See the class docstring for
        rationale.

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
        """Count the number of valid LOGO splits for multi-class cross-validation.

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

    @staticmethod
    def merge_labels(
        labels_by_behavior: dict[str, npt.NDArray[np.int8]],
        behavior_names: list[str],
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
        """Merge per-behavior label arrays into a single multi-class label array.

        Merging rules:
            - ``TrackLabels.Label.BEHAVIOR`` in ``MULTICLASS_NONE_BEHAVIOR`` entry
              → class 0 (background)
            - ``TrackLabels.Label.BEHAVIOR`` in behavior X's entry
              → class index (1-based, by position in ``behavior_names``)
            - All other frames → excluded (not in the returned mask)

        Args:
            labels_by_behavior: dict mapping behavior name to a label array of
                ``TrackLabels.Label`` integer values, one element per frame.
            behavior_names: Ordered list of N behavior names (must not include
                ``MULTICLASS_NONE_BEHAVIOR``).

        Returns:
            Tuple of ``(multiclass_labels, include_mask)`` where:

            - ``multiclass_labels``: integer array of class indices (0..N) for
              the included frames only, length M ≤ n_frames.
            - ``include_mask``: boolean array of length n_frames; True where the
              frame is included in training.
        """
        n_frames = next(iter(labels_by_behavior.values())).shape[0]
        class_indices = np.full(n_frames, -1, dtype=np.intp)

        # MULTICLASS_NONE_BEHAVIOR BEHAVIOR frames → class 0 (explicit background)
        if MULTICLASS_NONE_BEHAVIOR in labels_by_behavior:
            none_arr = labels_by_behavior[MULTICLASS_NONE_BEHAVIOR]
            class_indices[none_arr == TrackLabels.Label.BEHAVIOR] = 0

        # Named behavior BEHAVIOR frames → class 1..N
        for i, behavior in enumerate(behavior_names, start=1):
            if behavior in labels_by_behavior:
                beh_arr = labels_by_behavior[behavior]
                class_indices[beh_arr == TrackLabels.Label.BEHAVIOR] = i

        include_mask = class_indices >= 0
        return class_indices[include_mask], include_mask

    def _create_classifier(self, random_seed: int | None = None) -> typing.Any:
        """Instantiate the underlying sklearn/xgboost/catboost classifier."""
        try:
            factory = _CLASSIFIER_FACTORIES[self._classifier_type]
        except KeyError:
            raise ValueError(f"Unsupported classifier type: {self._classifier_type!r}") from None
        return factory(self._n_jobs, random_seed)

    def _get_features_to_classify(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reorder/select feature columns to match the trained model's expectations."""
        if self._classifier_type == ClassifierType.XGBOOST:
            classifier_columns = self._classifier.get_booster().feature_names
        elif hasattr(self._classifier, "feature_names_in_"):
            classifier_columns = list(self._classifier.feature_names_in_)
        elif hasattr(self._classifier, "feature_names_"):
            classifier_columns = list(self._classifier.feature_names_)
        else:
            raise RuntimeError("Error obtaining feature names from classifier.")
        return features[classifier_columns]

    @staticmethod
    def _supported_classifier_choices() -> set[ClassifierType]:
        """Determine supported classifier types in the current environment."""
        return set(_CLASSIFIER_FACTORIES.keys())
