"""Multi-period rebalancing scenarios and walk-forward backtesting.

A single-shot `solve_portfolio` call proves a solver can pick a portfolio;
it says nothing about whether that portfolio would actually have performed
well over time under realistic rebalancing constraints. This module:

1. Generates a genuinely correlated multi-period returns panel (explicit
   sector-block correlation structure, not i.i.d. draws).
2. Wraps it as a `core.dataset_interfaces.FinancialDataset` so walk-forward
   splitting goes through the platform's own leak-free
   `split_walk_forward` rather than hand-rolled slicing.
3. Rebalances period by period, with each period's turnover constraint
   referencing the PRIOR period's *actually realized* holdings (not a
   target, not an idealized carry-over).
4. Scores each period's realized return using a LATER return draw than the
   one used to estimate that period's expected-return/covariance inputs --
   the expected-return estimate for period t is fit only on data strictly
   before period t's own realized outcome is used to score it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioResult, PortfolioSpec, SolverChoice
from applications.portfolio.multi_solver import solve_portfolio
from applications.portfolio.synthetic_market import generate_synthetic_market
from core.dataset_interfaces import FinancialDataset
from core.errors import ValidationError
from core.metrics import expected_shortfall, max_drawdown, sharpe_ratio, value_at_risk
from core.schemas import DatasetDescriptor


class MultiPeriodReturnsDataset(FinancialDataset):
    """A synthetic multi-period realized-returns panel (T periods x N assets)
    exposed through the platform's standard `FinancialDataset` interface, so
    `split_walk_forward`'s leak-free-by-construction split logic applies
    here exactly as it would to any other dataset in this platform."""

    def __init__(self, realized_returns: np.ndarray, asset_names: list[str]) -> None:
        self._realized = np.asarray(realized_returns, dtype=float)
        self._names = list(asset_names)

    @property
    def descriptor(self) -> DatasetDescriptor:
        t, n = self._realized.shape
        return DatasetDescriptor(
            name="synthetic_multi_period_portfolio_returns", n_samples=t, n_features=n,
            domain="portfolio_construction",
        )

    def features(self) -> np.ndarray:
        return self._realized

    def targets(self) -> np.ndarray:
        # per-period cross-sectional average realized return -- used only as
        # a same-length series for split_walk_forward's index bookkeeping,
        # not fed to any solver.
        return self._realized.mean(axis=1)


def generate_multi_period_market(
    n_assets: int,
    n_periods: int,
    *,
    seed: int = 0,
    avg_correlation: float = 0.25,
    n_sectors: int = 3,
    sector_correlation_boost: float = 0.15,
) -> tuple[np.ndarray, list[str], list[int]]:
    """A realized-returns panel [n_periods, n_assets] drawn from ONE
    covariance structure with explicit sector blocks (assets in the same
    sector get an extra correlation boost on top of the base single-factor
    structure `synthetic_market.generate_synthetic_market` already builds),
    so a walk-forward backtest has genuine cross-sectional and sector
    structure to exploit or be fooled by -- not independent per-asset noise.

    Returns (realized_returns, asset_names, sector_id_per_asset).
    """
    if n_periods < 4:
        raise ValidationError("generate_multi_period_market: n_periods must be >= 4 for a meaningful backtest")
    rng = np.random.default_rng(seed)
    sector_id = [i % n_sectors for i in range(n_assets)]
    _mu0, cov0, names = generate_synthetic_market(n_assets, seed=seed, avg_correlation=avg_correlation)

    cov = cov0.copy()
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j and sector_id[i] == sector_id[j]:
                cov[i, j] += sector_correlation_boost * np.sqrt(cov0[i, i] * cov0[j, j])
    eigvals, eigvecs = np.linalg.eigh((cov + cov.T) / 2)
    eigvals = np.clip(eigvals, 1e-8, None)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    base_mu = rng.uniform(-0.01, 0.04, size=n_assets)
    realized = rng.multivariate_normal(base_mu, cov, size=n_periods)
    return realized, names, sector_id


def _estimate_period_inputs(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected-return/covariance inputs estimated ONLY from a training
    window strictly preceding the period being traded."""
    mu = window.mean(axis=0)
    cov = np.cov(window, rowvar=False) if window.shape[0] > 1 else np.eye(window.shape[1]) * 1e-4
    eigvals, eigvecs = np.linalg.eigh((cov + cov.T) / 2)
    eigvals = np.clip(eigvals, 1e-8, None)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return mu, cov


@dataclass
class RebalancePeriod:
    period: int
    spec: PortfolioSpec
    result: PortfolioResult
    realized_return: float
    turnover_violation: bool


@dataclass
class WalkForwardResult:
    solver: str
    periods: list[RebalancePeriod] = field(default_factory=list)
    period_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    n_infeasible_periods: int = 0
    n_turnover_violations: int = 0


def run_walk_forward_backtest(
    realized_returns: np.ndarray,
    asset_names: list[str],
    *,
    cardinality_k: int,
    solver: SolverChoice,
    n_splits: int = 4,
    turnover_limit: int | None = None,
    lookback: int = 20,
    risk_aversion: float = 0.5,
    seed: int = 42,
    extra_constraints: list[ConstraintSpec] | None = None,
) -> WalkForwardResult:
    """Walk-forward rebalancing backtest for one solver. Each split's TRAIN
    window (via `MultiPeriodReturnsDataset.split_walk_forward`, this
    platform's shared leak-free splitter) estimates that period's
    expected-return/covariance inputs; the realized return charged to that
    period comes from the first row of the corresponding TEST window --
    strictly later data than anything the period's `PortfolioSpec` was
    built from.
    """
    realized_returns = np.asarray(realized_returns, dtype=float)
    n = len(asset_names)
    dataset = MultiPeriodReturnsDataset(realized_returns, asset_names)
    splits = dataset.split_walk_forward(n_splits=n_splits, min_train_fraction=0.4)

    holdings = np.zeros(n)
    periods: list[RebalancePeriod] = []
    period_returns: list[float] = []
    n_infeasible = 0
    n_turnover_violations = 0

    for period_idx, (train_idx, test_idx) in enumerate(splits):
        window_idx = train_idx[-lookback:] if len(train_idx) > lookback else train_idx
        mu, cov = _estimate_period_inputs(realized_returns[window_idx])

        constraints = [ConstraintSpec(kind="cardinality", params={"k": cardinality_k})]
        constraints.extend(extra_constraints or [])
        if turnover_limit is not None and period_idx > 0:
            constraints.append(ConstraintSpec(
                kind="turnover", params={"previous_holdings": holdings.tolist(), "limit": turnover_limit},
            ))

        spec = PortfolioSpec(
            assets=[AssetSpec(name=nm, expected_return=float(mu[i])) for i, nm in enumerate(asset_names)],
            cov_matrix=cov.tolist(), risk_aversion=risk_aversion, constraints=constraints,
        )
        result = solve_portfolio(spec, solver, seed=seed + period_idx)
        if not result.feasible:
            n_infeasible += 1

        turnover_violation = False
        if turnover_limit is not None and period_idx > 0:
            changed = int(np.sum(np.abs(result.bitstring - holdings) > 0.5))
            turnover_violation = changed > turnover_limit
            n_turnover_violations += int(turnover_violation)

        test_period_returns = realized_returns[test_idx[0]]
        held = result.bitstring > 0.5
        realized_ret = float(test_period_returns[held].mean()) if held.any() else 0.0

        periods.append(RebalancePeriod(
            period=period_idx, spec=spec, result=result,
            realized_return=realized_ret, turnover_violation=turnover_violation,
        ))
        period_returns.append(realized_ret)
        holdings = result.bitstring.copy()

    returns_arr = np.array(period_returns)
    cumulative = np.cumprod(1 + returns_arr) if len(returns_arr) else np.array([1.0])
    return WalkForwardResult(
        solver=solver.value,
        periods=periods,
        period_returns=returns_arr,
        sharpe=sharpe_ratio(returns_arr, periods_per_year=max(len(returns_arr), 1)) if len(returns_arr) else 0.0,
        max_drawdown=max_drawdown(cumulative),
        var_95=value_at_risk(returns_arr) if len(returns_arr) else 0.0,
        cvar_95=expected_shortfall(returns_arr) if len(returns_arr) else 0.0,
        n_infeasible_periods=n_infeasible,
        n_turnover_violations=n_turnover_violations,
    )


def compare_walk_forward(
    realized_returns: np.ndarray,
    asset_names: list[str],
    *,
    cardinality_k: int,
    solvers: list[SolverChoice],
    n_splits: int = 4,
    turnover_limit: int | None = None,
    lookback: int = 20,
    risk_aversion: float = 0.5,
    seed: int = 42,
    extra_constraints: list[ConstraintSpec] | None = None,
) -> dict[str, WalkForwardResult]:
    """Run the identical walk-forward scenario (same realized-returns panel,
    same splits, same turnover carry-over rule) through every requested
    solver, so their out-of-sample Sharpe ratios are directly comparable --
    the honest "which generation actually won out-of-sample" comparison."""
    return {
        s.value: run_walk_forward_backtest(
            realized_returns, asset_names, cardinality_k=cardinality_k, solver=s,
            n_splits=n_splits, turnover_limit=turnover_limit, lookback=lookback,
            risk_aversion=risk_aversion, seed=seed, extra_constraints=extra_constraints,
        )
        for s in solvers
    }
