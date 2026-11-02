"""
Tests for core.feature_engineering — fracdiff, purged_cv, triple_barrier.

All tests run with numpy + pandas only.
"""

from math import comb

import numpy as np
import pandas as pd
import pytest

from core.feature_engineering.fracdiff import fracdiff, _ffd_weights, _adf_stationary
from core.feature_engineering.purged_cv import (
    PurgedKFold,
    CombinatorialPurgedKFold,
    purged_train_test_split,
)
from core.feature_engineering.triple_barrier import (
    triple_barrier_labels,
    events_from_cusum,
    label_stats,
)


# ── Synthetic data helpers ───────────────────────────────────────────────────

def make_price_series(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(log_ret))
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="close")


def make_random_walk(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n))
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(walk, index=idx, name="rw")


def make_feature_df(n: int = 200, d: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame(rng.standard_normal((n, d)),
                         index=idx,
                         columns=[f"f{i}" for i in range(d)])


# ── fracdiff ─────────────────────────────────────────────────────────────────

class TestFracdiff:
    def test_output_length_matches_input(self):
        s = make_random_walk()
        fd = fracdiff(s, d=0.35)
        assert len(fd) == len(s)

    def test_leading_nans_present(self):
        s = make_random_walk(n=200)
        fd = fracdiff(s, d=0.35, threshold=1e-3)
        assert fd.isna().sum() > 0, "FFD must produce leading NaNs for warm-up."

    def test_d1_approximates_first_difference(self):
        s = make_random_walk(n=100)
        fd1 = fracdiff(s, d=1.0, threshold=1e-8).dropna()
        diff1 = s.diff().dropna()
        common = fd1.index.intersection(diff1.index)
        np.testing.assert_allclose(
            fd1.loc[common].values, diff1.loc[common].values, atol=1e-6
        )

    def test_stationary_at_d035_random_walk(self):
        s = make_random_walk(n=500)
        fd = fracdiff(s, d=0.35, threshold=1e-3).dropna()
        assert len(fd) > 50, (
            f"Expected at least 50 valid values after fracdiff+dropna, got {len(fd)}."
        )
        assert _adf_stationary(fd.values, pvalue_threshold=0.05), (
            "fracdiff(d=0.35) on random walk should be ADF-stationary."
        )

    def test_weights_sum_to_finite(self):
        w = _ffd_weights(d=0.35, threshold=1e-5)
        assert np.isfinite(w).all()
        assert len(w) > 1

    def test_invalid_d_raises(self):
        s = make_random_walk(n=50)
        with pytest.raises(ValueError, match="d must be in"):
            fracdiff(s, d=0.0)
        with pytest.raises(ValueError, match="d must be in"):
            fracdiff(s, d=3.0)


# ── purged_cv ────────────────────────────────────────────────────────────────

class TestPurgedKFold:
    def test_split_yields_n_splits(self):
        X = make_feature_df(n=200)
        pkf = PurgedKFold(n_splits=5, embargo_td=10)
        splits = list(pkf.split(X))
        assert len(splits) == 5

    def test_train_test_disjoint(self):
        X = make_feature_df(n=200)
        pkf = PurgedKFold(n_splits=5, embargo_td=10)
        for train_idx, test_idx in pkf.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_all_test_indices_covered(self):
        X = make_feature_df(n=200)
        pkf = PurgedKFold(n_splits=5, embargo_td=0)
        all_test = []
        for _, test_idx in pkf.split(X):
            all_test.extend(test_idx.tolist())
        assert len(set(all_test)) == len(all_test), "Duplicate test indices across folds."
        assert set(all_test) == set(range(len(X))), "Not all indices covered by test folds."

    def test_no_temporal_leakage_with_t1(self):
        n = 100
        X = make_feature_df(n=n)
        t1 = pd.Series(
            [X.index[min(i + 20, n - 1)] for i in range(n)],
            index=X.index,
        )
        pkf = PurgedKFold(n_splits=4, embargo_td=5)
        for train_idx, test_idx in pkf.split(X, t1=t1):
            if len(test_idx) == 0:
                continue
            test_start_time = X.index[test_idx[0]]
            train_t1 = t1.iloc[train_idx]
            assert not (train_t1 >= test_start_time).any(), (
                "Temporal leakage: training t1 overlaps test start."
            )

    def test_get_n_splits(self):
        pkf = PurgedKFold(n_splits=7, embargo_td=0)
        assert pkf.get_n_splits() == 7

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits"):
            PurgedKFold(n_splits=1)


class TestCombinatorialPurgedKFold:
    def test_number_of_splits_is_choose(self):
        X = make_feature_df(n=240)
        cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo_td=0)
        splits = list(cpcv.split(X))
        assert len(splits) == comb(6, 2)
        assert cpcv.get_n_splits() == comb(6, 2)

    def test_number_of_backtest_paths(self):
        cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
        assert cpcv.get_n_paths() == comb(6, 2) * 2 // 6

    def test_train_test_disjoint(self):
        X = make_feature_df(n=240)
        cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo_td=5)
        for train_idx, test_idx in cpcv.split(X):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_test_size_is_k_over_n(self):
        n = 240
        X = make_feature_df(n=n)
        cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo_td=0)
        for _, test_idx in cpcv.split(X):
            assert abs(len(test_idx) - n * 2 / 6) <= 6

    def test_every_group_tested_equally(self):
        n = 180
        X = make_feature_df(n=n)
        N, k = 6, 2
        cpcv = CombinatorialPurgedKFold(n_splits=N, n_test_splits=k, embargo_td=0)
        groups = np.array_split(np.arange(n), N)
        counts = {gi: 0 for gi in range(N)}
        for _, test_idx in cpcv.split(X):
            tested_groups = {
                gi for gi, g in enumerate(groups) if g[0] in set(test_idx)
            }
            for gi in tested_groups:
                counts[gi] += 1
        assert set(counts.values()) == {comb(N - 1, k - 1)}

    def test_purge_removes_overlapping_train_rows(self):
        n = 120
        X = make_feature_df(n=n)
        t1 = pd.Series(
            [X.index[min(i + 15, n - 1)] for i in range(n)], index=X.index
        )
        N, k = 6, 2
        cpcv = CombinatorialPurgedKFold(n_splits=N, n_test_splits=k, embargo_td=0)
        groups = np.array_split(np.arange(n), N)
        for train_idx, test_idx in cpcv.split(X, t1=t1):
            test_row_set = set(test_idx.tolist())
            group_spans = [
                (X.index[g[0]], X.index[g[-1]])
                for g in groups
                if g[0] in test_row_set
            ]
            for i in train_idx:
                start_i, end_i = X.index[i], t1.iloc[i]
                for g_lo, g_hi in group_spans:
                    assert end_i < g_lo or start_i > g_hi, (
                        f"leakage: train row {i} overlaps a test group"
                    )

    def test_embargo_removes_following_rows(self):
        n = 120
        X = make_feature_df(n=n)
        embargo = 5
        cpcv = CombinatorialPurgedKFold(
            n_splits=6, n_test_splits=1, embargo_td=embargo
        )
        for train_idx, test_idx in cpcv.split(X):
            test_end = max(test_idx)
            banned = set(range(test_end + 1, min(test_end + 1 + embargo, n)))
            assert banned.isdisjoint(set(train_idx)), (
                "embargoed rows after a test group leaked into train"
            )

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError, match="n_splits"):
            CombinatorialPurgedKFold(n_splits=1, n_test_splits=1)
        with pytest.raises(ValueError, match="n_test_splits"):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=6)
        with pytest.raises(ValueError, match="n_test_splits"):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=0)
        with pytest.raises(ValueError, match="embargo"):
            CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo_td=-1)


class TestPurgedSplit:
    def test_train_test_sizes(self):
        X = make_feature_df(n=200)
        y = pd.Series(np.zeros(200), index=X.index)
        X_tr, X_te, _, _ = purged_train_test_split(X, y, test_size=0.2, embargo_td=10)
        assert len(X_te) == 40
        assert len(X_tr) <= 200 - 40


# ── triple_barrier ───────────────────────────────────────────────────────────

class TestTripleBarrier:
    def test_labels_in_valid_set(self):
        close = make_price_series(n=300)
        events = pd.Series(close.index[::10], index=close.index[::10])
        labels = triple_barrier_labels(close, events, pt=0.02, sl=0.01)
        assert set(labels.unique()).issubset({-1, 0, 1}), (
            f"Unexpected label values: {labels.unique()}"
        )

    def test_labels_not_all_zero(self):
        close = make_price_series(n=300)
        events = pd.Series(close.index[::5], index=close.index[::5])
        labels = triple_barrier_labels(close, events, pt=0.005, sl=0.005)
        assert labels.nunique() > 1, "Labels are all zero — barriers never triggered."

    def test_label_stats_sum_to_total(self):
        close = make_price_series(n=300)
        events = pd.Series(close.index[::10], index=close.index[::10])
        labels = triple_barrier_labels(close, events, pt=0.02, sl=0.01)
        stats = label_stats(labels)
        assert stats["buy"] + stats["hold"] + stats["sell"] == stats["total"]

    def test_cusum_events_generated(self):
        close = make_price_series(n=300)
        events = events_from_cusum(close, threshold=0.02)
        assert len(events) > 0, "CUSUM filter should generate at least one event."

    def test_dynamic_width_requires_vol(self):
        close = make_price_series(n=100)
        events = pd.Series(close.index[::10], index=close.index[::10])
        with pytest.raises(ValueError, match="vol is required"):
            triple_barrier_labels(close, events, dynamic_width=True)
