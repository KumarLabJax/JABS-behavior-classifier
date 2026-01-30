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


def make_lightgbm(n_jobs: int, random_seed: int | None) -> ClassifierMixin:
    """Factory function to construct a LightGBM classifier.

    LightGBM is optional and not installed by default. We try to import here.

    Args:
        n_jobs (int): Number of parallel jobs.
        random_seed (int | None): Random seed for reproducibility.

    Raises:
        RuntimeError: If LightGBM is not available.

    Note:
        Currently uses hyperparameters obtained from optimization on a
        specific dataset. These may not generalize well to other datasets.
    """
    params = {
        "n_estimators": 187,
        "learning_rate": 0.013940346079873234,
        "max_depth": 11,
        "num_leaves": 138,
        "min_child_samples": 16,
        "subsample": 0.7475884550556351,
        "colsample_bytree": 0.5171942605576092,
        "reg_alpha": 1.527156759251193,
        "reg_lambda": 2.133142332373004e-06,
    }

    try:
        import lightgbm
    except ImportError as e:
        raise RuntimeError(
            "LightGBM classifier requested but 'lightgbm' is not available in this environment."
        ) from e
    return lightgbm.LGBMClassifier(n_jobs=n_jobs, random_state=random_seed, verbose=-1, **params)
