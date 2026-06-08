"""Tests for excluded-group handling in leave-one-group-out cross-validation.

Excluded videos (groups) may serve as the held-out test group but are never
part of a training fold.
"""

import numpy as np
import pandas as pd

from jabs.classifier import classifier_utils


def _binary_logo_data():
    """Three groups (0, 1, 2), each with 25 positive and 25 negative labels.

    Each group individually meets the per-class LABEL_THRESHOLD (20), so every
    group is a valid test split when no groups are excluded.
    """
    groups = np.repeat([0, 1, 2], 50)
    labels = np.tile(np.array([1] * 25 + [0] * 25, dtype=np.int8), 3)
    n = len(labels)
    per_frame = pd.DataFrame({"f1": np.arange(n, dtype=float)})
    window = pd.DataFrame({"f2": np.arange(n, dtype=float)})
    return per_frame, window, labels, groups


def test_count_valid_logo_splits_without_exclusion():
    """All three groups are valid test splits when none are excluded."""
    _, _, labels, groups = _binary_logo_data()
    assert classifier_utils.count_valid_logo_splits(labels, groups) == 3


def test_count_valid_logo_splits_excluded_groups_drop_invalid_folds():
    """Excluding groups removes their rows from training folds.

    With groups 1 and 2 excluded, holding out group 0 leaves an empty training
    pool (its only other rows came from the excluded groups), so that fold is no
    longer valid; holding out group 1 or 2 still trains on group 0.
    """
    _, _, labels, groups = _binary_logo_data()
    assert classifier_utils.count_valid_logo_splits(labels, groups, excluded_groups={1, 2}) == 2


def test_leave_one_group_out_excludes_rows_from_training_but_allows_test():
    """An excluded group never appears in a training fold but can be the holdout."""
    per_frame, window, labels, groups = _binary_logo_data()
    splits = list(
        classifier_utils.leave_one_group_out(
            per_frame, window, labels, groups, excluded_groups={2}
        )
    )

    # the excluded group is still eligible as the held-out test group
    test_groups = {s["test_group"] for s in splits}
    assert 2 in test_groups

    # the excluded group's rows never appear in any training fold
    for split in splits:
        train_groups = set(np.unique(groups[split["training_idx"]]).tolist())
        assert 2 not in train_groups
