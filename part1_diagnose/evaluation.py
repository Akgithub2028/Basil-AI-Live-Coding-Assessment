"""Subject-disjoint evaluation and preprocessing (no model dependencies)."""
import numpy as np


def split_subjects(groups, seed=0):
    """Hold out the highest 20% of IDs; validate on 20% of the remaining IDs.

    With the prescribed subjects 1–10, test subjects are always 9 and 10.
    Validation subjects are sampled using the experiment seed, without labels.
    The remaining six subjects train the model; all three sets are disjoint.
    """
    subjects = np.unique(groups)
    if len(subjects) < 3:
        raise ValueError("Need at least three distinct subjects for train/val/test.")
    n_test = max(1, int(np.ceil(0.2 * len(subjects))))
    test_subjects = subjects[-n_test:]
    remaining = subjects[:-n_test]
    n_val = min(len(remaining) - 1, max(1, int(np.ceil(0.2 * len(remaining)))))
    val_subjects = np.random.default_rng(seed).choice(remaining, n_val, replace=False)
    te = np.flatnonzero(np.isin(groups, test_subjects))
    va = np.flatnonzero(np.isin(groups, val_subjects))
    tr = np.flatnonzero(~np.isin(groups, np.r_[test_subjects, val_subjects]))
    return tr, va, te


def standardize_from_train(X, tr):
    """Fit channel statistics on training trials only; apply to every split."""
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    return ((X - mean) / std).astype(np.float32), mean, std
