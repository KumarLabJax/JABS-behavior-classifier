"""Factory functions and registry for behavior classifiers.

``XGBOOST_AVAILABLE`` is set at import time by probing for the ``xgboost``
package. The factory registry below maps each ``ClassifierType`` to its
constructor for both binary and multi-class modes; mode-specific lookup is
exposed via :func:`get_factory` and :func:`supported_classifier_types`.
"""

import logging
import typing

from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from jabs.core.enums import ClassifierType

logger = logging.getLogger(__name__)

try:
    import xgboost as _xgboost  # noqa: F401

    XGBOOST_AVAILABLE: bool = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning(
        "Unable to import xgboost. XGBoost support will be unavailable. "
        "You may need to install xgboost and/or libomp."
    )


ClassifierFactory = typing.Callable[[int, int | None], typing.Any]


def make_random_forest(n_jobs: int, random_seed: int | None) -> RandomForestClassifier:
    """Construct a RandomForest classifier.

    Args:
        n_jobs: Number of parallel jobs.
        random_seed: Random seed for reproducibility.

    Returns:
        A configured ``RandomForestClassifier``.
    """
    return RandomForestClassifier(n_jobs=n_jobs, random_state=random_seed)


def make_catboost(n_jobs: int, random_seed: int | None) -> CatBoostClassifier:
    """Construct a CatBoost classifier for binary classification.

    Args:
        n_jobs: Number of parallel jobs.
        random_seed: Random seed for reproducibility.

    Returns:
        A configured ``CatBoostClassifier``.
    """
    return CatBoostClassifier(
        thread_count=n_jobs,
        random_state=random_seed,
        verbose=False,
        allow_writing_files=False,
    )


def make_catboost_multiclass(n_jobs: int, random_seed: int | None) -> CatBoostClassifier:
    """Construct a CatBoost classifier for multi-class classification.

    Uses ``loss_function="MultiClass"`` (softmax over all classes), required
    when the label set has more than two classes.

    Args:
        n_jobs: Number of parallel jobs.
        random_seed: Random seed for reproducibility.

    Returns:
        A configured ``CatBoostClassifier`` with multi-class loss.
    """
    return CatBoostClassifier(
        loss_function="MultiClass",
        thread_count=n_jobs,
        random_state=random_seed,
        verbose=False,
        allow_writing_files=False,
    )


def make_xgboost(n_jobs: int, random_seed: int | None) -> ClassifierMixin:
    """Construct an XGBoost classifier.

    XGBoost may not be available in all environments (e.g., macOS without
    libomp), so the import is deferred to call time.

    Args:
        n_jobs: Number of parallel jobs.
        random_seed: Random seed for reproducibility.

    Returns:
        A configured ``XGBClassifier``. Typed as ``ClassifierMixin`` to avoid a
        hard dependency on xgboost in type hints.

    Raises:
        RuntimeError: If XGBoost is not available in the current environment.
    """
    try:
        import xgboost
    except ImportError as e:
        raise RuntimeError(
            "XGBoost classifier requested but 'xgboost' is not available in this environment."
        ) from e
    return xgboost.XGBClassifier(n_jobs=n_jobs, random_state=random_seed)


_BINARY_FACTORIES: dict[ClassifierType, ClassifierFactory] = {
    ClassifierType.RANDOM_FOREST: make_random_forest,
    ClassifierType.CATBOOST: make_catboost,
}

_MULTICLASS_FACTORIES: dict[ClassifierType, ClassifierFactory] = {
    ClassifierType.RANDOM_FOREST: make_random_forest,
    ClassifierType.CATBOOST: make_catboost_multiclass,
}

if XGBOOST_AVAILABLE:
    _BINARY_FACTORIES[ClassifierType.XGBOOST] = make_xgboost
    _MULTICLASS_FACTORIES[ClassifierType.XGBOOST] = make_xgboost


def get_factory(classifier_type: ClassifierType, *, multiclass: bool) -> ClassifierFactory:
    """Look up the factory function for a classifier type and mode.

    Args:
        classifier_type: Which underlying algorithm to construct.
        multiclass: True for multi-class mode, False for binary mode.

    Returns:
        The factory callable producing an instance of the requested type.

    Raises:
        ValueError: If the classifier type is not supported in the current
            environment for the requested mode.
    """
    table = _MULTICLASS_FACTORIES if multiclass else _BINARY_FACTORIES
    try:
        return table[classifier_type]
    except KeyError:
        raise ValueError(f"Unsupported classifier type: {classifier_type!r}") from None


def supported_classifier_types(*, multiclass: bool) -> set[ClassifierType]:
    """Return the set of classifier types available in the current environment.

    Args:
        multiclass: True for multi-class mode, False for binary mode.

    Returns:
        Set of supported ``ClassifierType`` values.
    """
    table = _MULTICLASS_FACTORIES if multiclass else _BINARY_FACTORIES
    return set(table.keys())
