"""Factory functions and registry for behavior classifiers.

``XGBOOST_AVAILABLE``, ``CATBOOST_AVAILABLE`` and ``LIGHTGBM_AVAILABLE`` are set
at import time by probing for the ``xgboost``, ``catboost`` and ``lightgbm``
packages. CatBoost and LightGBM are optional dependencies, installed with the
``extra-classifiers`` extra. The factory registry below maps each ``ClassifierType`` to its
constructor for both binary and multi-class modes; mode-specific lookup is
exposed via :func:`get_factory` and :func:`supported_classifier_types`.
"""

import logging
import typing

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


try:
    import catboost as _catboost  # noqa: F401

    CATBOOST_AVAILABLE: bool = True
except (ImportError, OSError):
    # catboost is optional, and the extra skips it on Python versions it has no wheels
    # for, so a missing package is expected and not worth a warning.
    CATBOOST_AVAILABLE = False
    logger.debug("catboost is not available. CatBoost support will be unavailable.")

try:
    import lightgbm as _lightgbm  # noqa: F401

    LIGHTGBM_AVAILABLE: bool = True
except (ImportError, OSError):
    # lightgbm is optional, so a missing package is expected and not worth a warning.
    # OSError covers an installed package whose native library cannot load (e.g. macOS
    # without libomp), which would otherwise only fail when a classifier is trained.
    LIGHTGBM_AVAILABLE = False
    logger.debug("lightgbm is not available. LightGBM support will be unavailable.")

# Hyperparameters for LightGBM, from an optimization run on a single dataset. They may
# not generalize to every behavior, and JABS does not yet expose them for tuning.
_LIGHTGBM_PARAMS: dict[str, int | float] = {
    "n_estimators": 187,
    "learning_rate": 0.013940346079873234,
    "max_depth": 11,
    "num_leaves": 138,
    "min_child_samples": 16,
    "colsample_bytree": 0.5171942605576092,
    "reg_alpha": 1.527156759251193,
    "reg_lambda": 2.133142332373004e-06,
}


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


def _import_catboost() -> typing.Any:
    """Import and return the optional ``catboost`` module.

    Raises:
        RuntimeError: If CatBoost is not available in the current environment.
    """
    try:
        import catboost
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "CatBoost classifier requested but 'catboost' is not available in this "
            "environment. Install it with the 'extra-classifiers' extra."
        ) from e
    return catboost


def make_catboost(n_jobs: int, random_seed: int | None) -> ClassifierMixin:
    """Construct a CatBoost classifier for binary classification.

    CatBoost is an optional dependency, so the import is deferred to call time.

    Args:
        n_jobs: Number of parallel jobs.
        random_seed: Random seed for reproducibility.

    Returns:
        A configured ``CatBoostClassifier``. Typed as ``ClassifierMixin`` to avoid a
        hard dependency on catboost in type hints.

    Raises:
        RuntimeError: If CatBoost is not available in the current environment.
    """
    return _import_catboost().CatBoostClassifier(
        thread_count=n_jobs,
        random_state=random_seed,
        verbose=False,
        allow_writing_files=False,
    )


def make_catboost_multiclass(n_jobs: int, random_seed: int | None) -> ClassifierMixin:
    """Construct a CatBoost classifier for multi-class classification.

    Uses ``loss_function="MultiClass"`` (softmax over all classes), required
    when the label set has more than two classes.

    Args:
        n_jobs: Number of parallel jobs.
        random_seed: Random seed for reproducibility.

    Returns:
        A configured ``CatBoostClassifier`` with multi-class loss, typed as
        ``ClassifierMixin`` to avoid a hard dependency on catboost in type hints.

    Raises:
        RuntimeError: If CatBoost is not available in the current environment.
    """
    return _import_catboost().CatBoostClassifier(
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


def make_lightgbm(n_jobs: int, random_seed: int | None) -> ClassifierMixin:
    """Construct a LightGBM classifier.

    LightGBM is an optional dependency, so the import is deferred to call time. The
    same estimator serves binary and multi-class mode: LightGBM chooses its objective
    from the number of classes in the training labels.

    Args:
        n_jobs: Number of parallel jobs.
        random_seed: Random seed for reproducibility.

    Returns:
        A configured ``LGBMClassifier``. Typed as ``ClassifierMixin`` to avoid a
        hard dependency on lightgbm in type hints.

    Raises:
        RuntimeError: If LightGBM is not available in the current environment.
    """
    try:
        import lightgbm
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "LightGBM classifier requested but 'lightgbm' is not available in this "
            "environment. Install it with the 'extra-classifiers' extra."
        ) from e
    return lightgbm.LGBMClassifier(
        n_jobs=n_jobs, random_state=random_seed, verbose=-1, **_LIGHTGBM_PARAMS
    )


_BINARY_FACTORIES: dict[ClassifierType, ClassifierFactory] = {
    ClassifierType.RANDOM_FOREST: make_random_forest,
}

_MULTICLASS_FACTORIES: dict[ClassifierType, ClassifierFactory] = {
    ClassifierType.RANDOM_FOREST: make_random_forest,
}

if CATBOOST_AVAILABLE:
    _BINARY_FACTORIES[ClassifierType.CATBOOST] = make_catboost
    _MULTICLASS_FACTORIES[ClassifierType.CATBOOST] = make_catboost_multiclass

if XGBOOST_AVAILABLE:
    _BINARY_FACTORIES[ClassifierType.XGBOOST] = make_xgboost
    _MULTICLASS_FACTORIES[ClassifierType.XGBOOST] = make_xgboost

if LIGHTGBM_AVAILABLE:
    _BINARY_FACTORIES[ClassifierType.LIGHTGBM] = make_lightgbm
    _MULTICLASS_FACTORIES[ClassifierType.LIGHTGBM] = make_lightgbm


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
