"""Rolling / walk-forward tail-risk evaluation: fit VaR/CVaR on a training
window, backtest breach performance on the following, never-seen test
window, over a rolling series of splits -- showing how a VaR/CVaR estimate
would have evolved and held up out-of-sample, not just what it is on the
full sample.

Reuses `core.dataset_interfaces.FinancialDataset.split_walk_forward` (the
platform's one leakage-safe splitter) rather than reimplementing index
arithmetic -- `_PnLSeriesDataset` below exists solely to adapt a 1-D P&L
series to that interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.metrics import expected_shortfall, value_at_risk
from core.schemas import DatasetDescriptor

from applications.tail_risk.backtest import BreachTestResult, backtest_var_breaches
from applications.tail_risk.errors import validate_pnl_sample

__all__ = ["WalkForwardFold", "WalkForwardReport", "rolling_tail_risk"]


class _PnLSeriesDataset(FinancialDataset):
    """Minimal `FinancialDataset` adapter over a 1-D PnL series, existing
    solely so walk-forward tail-risk evaluation reuses the platform's one
    leakage-safe splitter instead of reimplementing index arithmetic."""

    def __init__(self, pnl: np.ndarray) -> None:
        self._pnl = np.asarray(pnl, dtype=float)

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="pnl_series", n_samples=len(self._pnl), n_features=1, domain="tail_risk_pnl",
        )

    def features(self) -> np.ndarray:
        return self._pnl.reshape(-1, 1)

    def targets(self) -> np.ndarray:
        return self._pnl


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    n_train: int
    n_test: int
    var_estimate: float
    cvar_estimate: float
    breach_test: BreachTestResult


@dataclass(frozen=True)
class WalkForwardReport:
    alpha: float
    n_splits: int
    folds: list[WalkForwardFold]

    def var_series(self) -> list[float]:
        return [f.var_estimate for f in self.folds]

    def cvar_series(self) -> list[float]:
        return [f.cvar_estimate for f in self.folds]

    def total_breaches(self) -> int:
        return sum(f.breach_test.n_breaches for f in self.folds)


def rolling_tail_risk(
    pnl_series: np.ndarray, *, n_splits: int = 4, alpha: float = 0.95, min_train_fraction: float = 0.5,
) -> WalkForwardReport:
    """Walk-forward VaR/CVaR: for each fold, fit historical VaR/CVaR
    (`core.metrics`) on the training window and backtest breaches
    (`applications.tail_risk.backtest`) on the following test window. Every
    fold's train indices strictly precede its test indices (guaranteed by
    `split_walk_forward`'s contract), so no fold's VaR estimate ever sees
    the data it is evaluated against.
    """
    pnl = validate_pnl_sample(pnl_series)
    dataset = _PnLSeriesDataset(pnl)
    splits = dataset.split_walk_forward(n_splits, min_train_fraction=min_train_fraction)

    folds: list[WalkForwardFold] = []
    for i, (train_idx, test_idx) in enumerate(splits):
        train_pnl = pnl[train_idx]
        test_pnl = pnl[test_idx]
        var = value_at_risk(train_pnl, alpha)
        cvar = expected_shortfall(train_pnl, alpha)
        breach = backtest_var_breaches(test_pnl, var, alpha=alpha)
        folds.append(
            WalkForwardFold(
                fold=i, n_train=len(train_idx), n_test=len(test_idx),
                var_estimate=var, cvar_estimate=cvar, breach_test=breach,
            )
        )
    return WalkForwardReport(alpha=alpha, n_splits=n_splits, folds=folds)
