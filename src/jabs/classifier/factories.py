"""Factory functions for various classifiers."""

from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


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
