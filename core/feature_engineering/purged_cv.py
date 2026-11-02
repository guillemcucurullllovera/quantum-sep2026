"""
Purged K-Fold Cross-Validator with Embargo.

Reference: López de Prado (2018) "Advances in Financial Machine Learning"
           Chapter 7 — Cross-Validation in Finance; Chapter 12 (CPCV) —
           Backtesting through Cross-Validation.

Standard k-fold leaks information when training returns overlap with test
returns in time. This validator:
    1. PURGES: removes training observations whose event windows overlap
       with the test fold's event windows.
    2. EMBARGOES: removes the `embargo_td` rows immediately after the test
       fold to prevent forward-looking bias from slow-moving features.

No sklearn required — implements the CV split protocol from scratch.
For sklearn integration, pass the splits into cross_val_score manually.

See core.feature_engineering (package docstring) for how this relates to
core.dataset_interfaces.FinancialDataset.split_walk_forward.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """
    Purged k-fold cross-validator with embargo.

    Usage (pandas index style):
        pkf = PurgedKFold(n_splits=5, embargo_td=60)
        for train_idx, test_idx in pkf.split(X, t1=t1):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            model.predict(X.iloc[test_idx])

    Args:
        n_splits    : number of folds (default 5)
        embargo_td  : embargo period in ROWS after each test fold (default 60).
                      Use 60 for monthly data with a 60-day forward window;
                      scale accordingly for daily or weekly data.
    """

    def __init__(self, n_splits: int = 5, embargo_td: int = 60):
        if n_splits < 2:
            raise ValueError(f"n_splits must be ≥ 2; got {n_splits}")
        if embargo_td < 0:
            raise ValueError(f"embargo_td must be ≥ 0; got {embargo_td}")
        self.n_splits = n_splits
        self.embargo_td = embargo_td

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        t1: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index pairs.

        Args:
            X  : feature DataFrame (uses its index for time ordering)
            y  : ignored (present for sklearn API compatibility)
            t1 : pd.Series of event end times aligned to X's index.
                 If None, uses X.index (assumes each row is a point event).
        Yields:
            (train_indices, test_indices) as 1-D integer arrays
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if t1 is None:
            t1 = pd.Series(X.index, index=X.index)

        indices = np.arange(len(X))
        test_starts = np.array_split(indices, self.n_splits)

        for fold_indices in test_starts:
            test_start = fold_indices[0]
            test_end = fold_indices[-1]

            # Embargo: rows immediately after test fold
            embargo_end = min(test_end + self.embargo_td, len(X) - 1)

            # Test mask
            test_mask = np.zeros(len(X), dtype=bool)
            test_mask[fold_indices] = True

            # Purge: remove training rows whose t1 overlaps test window
            # t1[i] >= X.index[test_start] means the event spans into the test fold
            test_start_time = X.index[test_start]
            overlap = t1 >= test_start_time

            # Also remove embargo window
            embargo_mask = np.zeros(len(X), dtype=bool)
            if embargo_end > test_end:
                embargo_mask[test_end + 1 : embargo_end + 1] = True

            train_mask = ~test_mask & ~overlap & ~embargo_mask
            yield indices[train_mask], indices[test_mask]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class CombinatorialPurgedKFold:
    """Combinatorial Purged Cross-Validation (CPCV).

    Reference: López de Prado (2018) "Advances in Financial Machine Learning",
    Chapter 12 — Backtesting through Cross-Validation.

    Where :class:`PurgedKFold` holds out ONE contiguous group per split, CPCV
    partitions the observations into ``n_splits`` (= N) groups and, for every
    combination, holds out ``n_test_splits`` (= k) of them as the test set. This
    yields ``C(N, k)`` train/test splits and — because each group is tested in
    several combinations — reconstructs ``φ = C(N,k)·k / N`` distinct, leakage-
    controlled backtest paths.

    Purge + embargo are applied around EVERY test group (test groups may be
    non-contiguous), so no training label window overlaps a test window and the
    ``embargo_td`` rows immediately after each test group are dropped.

    Args:
        n_splits      : number of groups N to partition the observations into
                        (default 6). Must be ≥ 2.
        n_test_splits : number of groups k held out as test per combination
                        (default 2). Must satisfy 1 ≤ k < N.
        embargo_td    : embargo period in ROWS after each test group (default 0).

    Usage:
        cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo_td=5)
        for train_idx, test_idx in cpcv.split(X, t1=t1):
            ...
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_td: int = 0,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be ≥ 2; got {n_splits}")
        if not (1 <= n_test_splits < n_splits):
            raise ValueError(
                f"n_test_splits must satisfy 1 ≤ k < n_splits; "
                f"got k={n_test_splits}, n_splits={n_splits}"
            )
        if embargo_td < 0:
            raise ValueError(f"embargo_td must be ≥ 0; got {embargo_td}")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_td = embargo_td

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        t1: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate CPCV train/test index pairs.

        Args:
            X  : feature DataFrame (uses its index for time ordering)
            y  : ignored (sklearn API compatibility)
            t1 : pd.Series of event end times aligned to X's index. If None,
                 each row is treated as a point event (t1 == index).
        Yields:
            (train_indices, test_indices) as 1-D integer arrays.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if t1 is None:
            t1 = pd.Series(X.index, index=X.index)

        n = len(X)
        indices = np.arange(n)
        groups = np.array_split(indices, self.n_splits)
        index_values = X.index

        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_group_arrays = [groups[g] for g in test_combo]
            test_idx = np.concatenate(test_group_arrays)

            test_mask = np.zeros(n, dtype=bool)
            test_mask[test_idx] = True

            purge_mask = np.zeros(n, dtype=bool)
            embargo_mask = np.zeros(n, dtype=bool)

            for grp in test_group_arrays:
                g_start, g_end = grp[0], grp[-1]
                g_start_time = index_values[g_start]
                g_end_time = index_values[g_end]

                # PURGE: drop any (train) row whose label interval
                # [index[i], t1[i]] overlaps this test group's time span.
                # Overlap ⇔ NOT (t1[i] < g_start_time OR index[i] > g_end_time).
                overlap = ~((t1.values < g_start_time) | (index_values.values > g_end_time))
                purge_mask |= overlap

                # EMBARGO: drop the embargo_td rows immediately after the group.
                if self.embargo_td > 0 and g_end < n - 1:
                    emb_end = min(g_end + self.embargo_td, n - 1)
                    embargo_mask[g_end + 1 : emb_end + 1] = True

            train_mask = ~test_mask & ~purge_mask & ~embargo_mask
            yield indices[train_mask], indices[test_mask]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of combinatorial train/test splits: C(N, k)."""
        return comb(self.n_splits, self.n_test_splits)

    def get_n_paths(self) -> int:
        """Number of distinct backtest paths reconstructed: φ = C(N,k)·k / N."""
        return comb(self.n_splits, self.n_test_splits) * self.n_test_splits // self.n_splits


def purged_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    embargo_td: int = 60,
    t1: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Simple walk-forward split with embargo (no cross-validation).

    Args:
        X          : feature DataFrame
        y          : label Series
        test_size  : fraction of data to use as test (default 0.2 = 20%)
        embargo_td : embargo rows between train and test
        t1         : event end times (None → use X.index)
    Returns:
        X_train, X_test, y_train, y_test
    """
    n = len(X)
    n_test = max(1, int(n * test_size))
    n_embargo = embargo_td
    n_train = n - n_test - n_embargo

    if n_train <= 0:
        raise ValueError(
            f"Not enough data for split: n={n}, test={n_test}, embargo={n_embargo}"
        )

    if t1 is not None:
        test_start_time = X.index[n_train + n_embargo]
        overlap_mask = t1.iloc[:n_train + n_embargo] >= test_start_time
        train_indices = np.where(~overlap_mask.values)[0]
    else:
        train_indices = np.arange(n_train)

    test_indices = np.arange(n_train + n_embargo, n)

    return (
        X.iloc[train_indices],
        X.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[test_indices],
    )


__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    "purged_train_test_split",
]
