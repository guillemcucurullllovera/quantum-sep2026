"""Data pipeline: turns a realized-returns panel into price levels, derives
a fracdiff-stationarized trend feature per asset (reusing
`core.feature_engineering.fracdiff` -- fractional differencing is not
re-derived here), and checks whether that feature actually predicts the
next period's return via `core.feature_engineering.purged_cv.PurgedKFold`
so any signal a caller might use to tilt `PortfolioSpec.expected_returns`
has an honest, leakage-controlled check behind it rather than an in-sample
rubber stamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.dataset_interfaces import FinancialDataset
from core.errors import ValidationError
from core.feature_engineering.fracdiff import fracdiff
from core.feature_engineering.purged_cv import PurgedKFold
from core.metrics import mse
from core.schemas import DatasetDescriptor


def returns_to_prices(realized_returns: np.ndarray, start_price: float = 100.0) -> np.ndarray:
    """[T, N] realized simple returns -> [T+1, N] price levels (row 0 is
    the common start price every asset is normalized to)."""
    returns = np.asarray(realized_returns, dtype=float)
    prices = np.empty((returns.shape[0] + 1, returns.shape[1]))
    prices[0] = start_price
    for t in range(returns.shape[0]):
        prices[t + 1] = prices[t] * (1 + returns[t])
    return prices


def fracdiff_trend_features(
    prices: np.ndarray, asset_names: list[str], d: float = 0.4, threshold: float = 1e-2
) -> pd.DataFrame:
    """One fracdiff-stationarized trend feature per asset, each computed
    independently via `core.feature_engineering.fracdiff.fracdiff` -- this
    module only orchestrates, it does not re-derive fractional
    differencing. `threshold` defaults looser (1e-2, vs. fracdiff's own
    1e-5 default) than a research-grade fracdiff call would use: the FFD
    weight window's width grows fast as threshold shrinks (empirically
    ~11 lags at 1e-2 vs. ~1450 at 1e-5 for d=0.4), and this application's
    multi-period scenarios (`scenario.py`) intentionally use short panels
    (tens, not thousands, of periods) where a research-grade threshold
    would burn in more rows than exist.
    """
    df = pd.DataFrame(np.asarray(prices, dtype=float), columns=asset_names)
    return pd.DataFrame({name: fracdiff(df[name], d=d, threshold=threshold) for name in asset_names})


def evaluate_trend_signal_purged(
    realized_returns: np.ndarray,
    asset_names: list[str],
    *,
    d: float = 0.4,
    threshold: float = 1e-2,
    n_splits: int = 4,
    embargo_td: int = 1,
) -> dict:
    """Does the cross-sectional-average fracdiff trend feature predict the
    NEXT period's cross-sectional-average realized return? Evaluated via
    `PurgedKFold` (train folds never overlap a test fold's event window,
    plus an embargo) rather than a plain train/test split -- a real,
    leakage-controlled predictive check, not a rubber-stamp. A trivial
    one-feature OLS fit per fold is deliberately the whole model here: the
    point of this function is the leak-free evaluation protocol, not the
    predictor.
    """
    prices = returns_to_prices(realized_returns)
    features = fracdiff_trend_features(prices, asset_names, d=d, threshold=threshold).iloc[1:].reset_index(drop=True)
    signal = features.mean(axis=1)
    target = pd.Series(np.asarray(realized_returns, dtype=float).mean(axis=1), index=signal.index)

    valid = signal.dropna().index
    signal, target = signal.loc[valid], target.loc[valid]
    if len(signal) < n_splits * 3:
        raise ValidationError(
            f"evaluate_trend_signal_purged: only {len(signal)} rows survive the fracdiff burn-in "
            f"window -- not enough for {n_splits} purged folds. Use more periods or a smaller d."
        )

    X = pd.DataFrame({"trend": signal.values}, index=signal.index)
    pkf = PurgedKFold(n_splits=n_splits, embargo_td=embargo_td)
    fold_mses: list[float] = []
    for train_idx, test_idx in pkf.split(X, y=target):
        if len(train_idx) < 2 or len(test_idx) == 0:
            continue
        x_train, y_train = X.iloc[train_idx]["trend"].values, target.iloc[train_idx].values
        x_test, y_test = X.iloc[test_idx]["trend"].values, target.iloc[test_idx].values
        beta = float(np.cov(x_train, y_train)[0, 1] / (np.var(x_train) + 1e-12))
        alpha = float(y_train.mean() - beta * x_train.mean())
        pred = alpha + beta * x_test
        fold_mses.append(mse(y_test, pred))

    naive_mse = mse(target.values, np.full(len(target), target.mean()))
    return {
        "fold_mses": fold_mses,
        "mean_fold_mse": float(np.mean(fold_mses)) if fold_mses else None,
        "naive_mean_baseline_mse": naive_mse,
        "n_folds_evaluated": len(fold_mses),
        "beats_naive_baseline": (float(np.mean(fold_mses)) < naive_mse) if fold_mses else None,
    }


class PortfolioFinancialDataset(FinancialDataset):
    """Exposes a realized-returns panel's fracdiff trend features through
    the platform's standard `FinancialDataset` interface, so this
    application layer's data reuses the same split/cross-validation tooling
    every other dataset in this platform does, instead of a bespoke path."""

    def __init__(self, realized_returns: np.ndarray, asset_names: list[str], d: float = 0.4, threshold: float = 1e-2) -> None:
        self._realized = np.asarray(realized_returns, dtype=float)
        self._names = list(asset_names)
        prices = returns_to_prices(self._realized)
        self._features = fracdiff_trend_features(prices, self._names, d=d, threshold=threshold).iloc[1:].reset_index(drop=True)

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="portfolio_fracdiff_features", n_samples=len(self._features),
            n_features=self._features.shape[1], domain="portfolio_construction",
        )

    def features(self) -> np.ndarray:
        return self._features.to_numpy()

    def targets(self) -> np.ndarray:
        return self._realized.mean(axis=1)
