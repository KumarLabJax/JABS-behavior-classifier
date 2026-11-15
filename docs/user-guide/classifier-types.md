# Choosing a Classifier Type

JABS supports four machine learning classifier types: **Random Forest**, **XGBoost**, **CatBoost**, and **LightGBM**. Random Forest and XGBoost are installed with JABS. CatBoost and LightGBM are optional and are installed with the `extra-classifiers` extra:

```bash
pip install "jabs-behavior-classifier[extra-classifiers]"
```

See [Optional Extras](optional-extras.md) for how to add an extra to an existing installation, including with uv or pipx. An optional classifier that is not installed, or cannot be loaded, is left out of the classifier choices in JABS. CatBoost does not yet publish packages for Python 3.15, so on Python 3.15 the extra installs LightGBM only.

Each has different characteristics that may make it more suitable for your specific use case.

> **Classifier type vs. classifier mode:** This page covers the machine learning *algorithm*. Separately, JABS can train one binary classifier per behavior (the default) or a single classifier across all behaviors at once. See [Multi-Class Classification (Preview)](multi-class.md) for the experimental multi-class mode.

## Random Forest (Default)

Random Forest is the default classifier and a good starting point for most users.

**Pros:**

- ✅ **Fast training** - Trains quickly, even on large datasets
- ✅ **Well-established** - Mature algorithm with extensive validation
- ✅ **Good baseline performance** - Reliable results across many behavior types
- ✅ **Low memory footprint** - Efficient with system resources

**Cons:**

- ⚠️ **May plateau** - Sometimes reaches a performance ceiling compared to gradient boosting methods
- ⚠️ **Less flexible** - Fewer tuning options than boosting methods
- ⚠️ **Does not handle missing data** - Random Forest does not natively handle missing (NaN) values, so JABS currently replaces all NaNs with 0 during training and classification. This might not be a good choice if your data has many missing values

**Best for:** Quick iterations, initial exploration, behaviors with simpler decision boundaries, or when training time is a priority.

## CatBoost

CatBoost is a gradient boosting algorithm that can achieve excellent performance, particularly for complex behaviors. It is installed with the `extra-classifiers` extra.

**Pros:**

- ✅ **High accuracy** - Often achieves the best classification performance
- ✅ **Handles missing data natively** - No imputation needed for NaN values
- ✅ **Robust to overfitting** - Built-in regularization techniques
- ✅ **No external dependencies** - Installs cleanly without additional libraries (on Python 3.10-3.14)

**Cons:**

- ⚠️ **Slower training** - Takes significantly longer to train than Random Forest
- ⚠️ **Higher memory usage** - May require more RAM during training
- ⚠️ **Longer to classify** - Prediction can be slower on very large datasets
- ⚠️ **Optional install** - Requires installing the `extra-classifiers` extra, and is not yet available on Python 3.15

**Best for:** Final production classifiers where accuracy is paramount, complex behaviors with subtle patterns, or when you have time for longer training sessions.

## XGBoost

XGBoost is another gradient boosting algorithm known for winning machine learning competitions.

**Pros:**

- ✅ **Excellent performance** - Typically matches or exceeds Random Forest accuracy
- ✅ **Handles missing data natively** - Like CatBoost, works with NaN values
- ✅ **Faster than CatBoost** - Better training speed than CatBoost
- ✅ **Widely used** - Extensive community support and documentation

**Cons:**

- ⚠️ **Dependency on libomp** - On macOS, may require separate installation of OpenMP library
- ⚠️ **Slower than Random Forest** - Training takes longer than Random Forest
- ⚠️ **May be unavailable** - If libomp is not installed, XGBoost won't be available as a choice in JABS

**Best for:** When you need better accuracy than Random Forest but faster training than CatBoost, or when you're familiar with gradient boosting methods.

## LightGBM

LightGBM is a gradient boosting algorithm designed for fast training and low memory use. It is installed with the `extra-classifiers` extra.

**Pros:**

- ✅ **Fast training** - Faster than XGBoost and CatBoost
- ✅ **Good accuracy** - Typically comparable to XGBoost
- ✅ **Handles missing data natively** - Like CatBoost and XGBoost, works with NaN values
- ✅ **Low memory usage** - Efficient with system resources

**Cons:**

- ⚠️ **Optional install** - Requires installing the `extra-classifiers` extra
- ⚠️ **Fixed hyperparameters** - JABS uses hyperparameters tuned on a single dataset, which may not suit every behavior. Tuning them is not currently supported in JABS
- ⚠️ **Dependency on libomp** - Like XGBoost, may require a separate OpenMP installation on macOS

**Best for:** When you want gradient boosting accuracy with faster training, especially on large projects.

## Quick Comparison

| Feature | Random Forest | CatBoost | XGBoost | LightGBM |
|---------|--------------|----------|---------|----------|
| **Training Speed** | ⚡⚡⚡ Fast | 🐌 Slow | ⚡⚡ Moderate | ⚡⚡⚡ Fast |
| **Accuracy** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐ Very Good |
| **Missing Data Handling** | Imputation to 0 | Native support | Native support | Native support |
| **Setup Complexity** | ✅ Simple | ⚠️ Optional extra | ⚠️ May need libomp | ⚠️ Optional extra; may need libomp |
| **Best Use Case** | Quick iterations | Production accuracy | Balanced performance | Fast gradient boosting |

## Recommendations

**Getting Started:** Start with **Random Forest** to quickly iterate and establish a baseline. It trains fast, allowing you to experiment with different labeling strategies and window sizes.

**Optimizing Performance:** Once you've refined your labels and parameters, try **CatBoost** for a potential accuracy boost. The longer training time is worthwhile for your final classifier.

**Alternative:** If CatBoost is too slow or you want something between Random Forest and CatBoost, try **XGBoost** (if available on your system).

**Note:** The actual performance difference between classifiers varies by behavior type and dataset. We recommend testing multiple classifiers on your specific data to find the best option for your use case.