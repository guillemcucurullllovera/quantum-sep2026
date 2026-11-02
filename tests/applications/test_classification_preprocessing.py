"""Tests for `applications.classification.preprocessing`: the composable
fracdiff + train-only-standardization step, and the purged walk-forward
split helper."""
from __future__ import annotations

import numpy as np
import pytest

from applications.classification.preprocessing import (
    Preprocessor,
    PreprocessingConfig,
    purged_walk_forward_splits,
)
from core.errors import ValidationError


def test_preprocessor_standardizes_using_train_only_stats():
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
    X_test = rng.normal(loc=50.0, scale=2.0, size=(20, 3))  # very different distribution

    pre = Preprocessor(PreprocessingConfig(standardize=True)).fit(X_train)
    Z_train = pre.transform(X_train)
    Z_test = pre.transform(X_test)

    # train transform should be ~standard normal (fit on itself)
    assert np.abs(Z_train.mean(axis=0)).max() < 0.3
    assert np.abs(Z_train.std(axis=0) - 1.0).max() < 0.3
    # test transform uses the SAME (train) mean/std, so an out-of-distribution
    # test set produces a large, not-near-zero, standardized mean
    assert np.abs(Z_test.mean(axis=0)).min() > 5.0


def test_preprocessor_requires_fit_before_transform():
    pre = Preprocessor()
    with pytest.raises(ValidationError):
        pre.transform(np.zeros((5, 2)))


def test_preprocessor_fracdiff_changes_the_series_and_preserves_shape():
    rng = np.random.default_rng(1)
    X = np.cumsum(rng.normal(size=(200, 2)), axis=0)  # random-walk-like, non-stationary
    pre_raw = Preprocessor(PreprocessingConfig(standardize=False))
    pre_fd = Preprocessor(PreprocessingConfig(fracdiff_d=0.4, standardize=False))
    Z_raw = pre_raw.fit_transform(X)
    Z_fd = pre_fd.fit_transform(X)
    assert Z_raw.shape == X.shape
    assert Z_fd.shape == X.shape
    assert not np.allclose(Z_raw, Z_fd)


def test_purged_walk_forward_splits_are_disjoint_and_embargoed():
    splits = purged_walk_forward_splits(n_samples=100, n_splits=5, embargo_td=3, label_horizon=5)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(test_idx) > 0
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
        # purge: no training row's [t0, t0+label_horizon] window can overlap the
        # test fold's start
        if len(train_idx) > 0 and len(test_idx) > 0:
            test_start = test_idx.min()
            overlapping = train_idx[(train_idx + 5) >= test_start]
            assert len(overlapping) == 0
