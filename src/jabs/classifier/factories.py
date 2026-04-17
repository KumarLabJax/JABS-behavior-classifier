"""Factory functions for various classifiers.

``XGBOOST_AVAILABLE`` is set at import time by probing for the ``xgboost``
package.  Both ``Classifier`` and ``MultiClassClassifier`` import this flag
to conditionally register XGBoost support, so the availability check and
warning are emitted exactly once regardless of how many classifier modules
are imported.
"""

import logging

from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

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


def make_random_forest(n_jobs: int, random_seed: int | None) -> RandomForestClassifier:
    """Factory function to construct a RandomForest classifier.

    Args:
        n_jobs (int): Number of parallel jobs.
        random_seed (int | None): Random seed for reproducibility.

    Returns:
        RandomForestClassifier: An instance of RandomForestClassifier.
    """
    return RandomForestClassifier(n_jobs=n_jobs, random_state=random_seed)


def make_catboost(n_jobs: int, random_seed: int | None) -> CatBoostClassifier:
    """Factory function to construct a CatBoost classifier.

    Args:
        n_jobs (int): Number of parallel jobs.
        random_seed (int | None): Random seed for reproducibility.

    Returns:
        CatBoostClassifier: An instance of CatBoostClassifier.
    """
    return CatBoostClassifier(
        thread_count=n_jobs,
        random_state=random_seed,
        verbose=False,  # Suppress training output
        allow_writing_files=False,  # Don't write intermediate files
    )


def make_catboost_multiclass(n_jobs: int, random_seed: int | None) -> CatBoostClassifier:
    """Factory function to construct a CatBoost classifier for multi-class problems.

    Uses ``loss_function="MultiClass"`` (softmax over all classes), which is
    required when the label set has more than two classes.

    Args:
        n_jobs: Number of parallel jobs.
        random_seed: Random seed for reproducibility.

    Returns:
        CatBoostClassifier configured for multi-class classification.
    """
    return CatBoostClassifier(
        loss_function="MultiClass",
        thread_count=n_jobs,
        random_state=random_seed,
        verbose=False,
        allow_writing_files=False,
    )


def make_xgboost(n_jobs: int, random_seed: int | None) -> ClassifierMixin:
    """Factory function to construct an XGBoost classifier.

    XGBoost might not be available in all environments (such as macOS without
    libomp installed), so we try to import here.

    Args:
        n_jobs (int): Number of parallel jobs.
        random_seed (int | None): Random seed for reproducibility.

    Returns:
        An instance of XGBClassifier. Note: type hint is ClassifierMixin to avoid
        direct dependency on xgboost in type hints.

    Raises:
        RuntimeError: If XGBoost is not available.
    """
    try:
        import xgboost
    except ImportError as e:
        raise RuntimeError(
            "XGBoost classifier requested but 'xgboost' is not available in this environment."
        ) from e
    return xgboost.XGBClassifier(n_jobs=n_jobs, random_state=random_seed)
