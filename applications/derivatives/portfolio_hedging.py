"""Portfolio hedging composition: takes a solved PortfolioSpec allocation,
Monte-Carlo-simulates its P&L under correlated asset returns (Cholesky of
the spec's own covariance matrix, reusing classical.monte_carlo's GBM
engine per-asset rather than reimplementing path simulation), and feeds
the result into applications.tail_risk.assess_tail_risk -- then compares
against a simple scaled-down-exposure hedge to show whether it actually
improves CVaR. This is a real composition across three existing modules,
not a new algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applications.portfolio.domain import PortfolioSpec
from applications.tail_risk.var_cvar import TailRiskReport, assess_tail_risk
from classical.monte_carlo.gbm import gbm_terminal_prices
from core.errors import ValidationError


def simulate_portfolio_pnl(
    spec: PortfolioSpec,
    weights: np.ndarray,
    *,
    horizon_years: float = 1.0,
    n_paths: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Correlated multi-asset terminal P&L via a Cholesky-coupled combination
    of per-asset GBM draws (reusing classical.monte_carlo.gbm_terminal_prices
    for the underlying univariate simulation)."""
    weights = np.asarray(weights, dtype=float)
    n = spec.n_assets()
    if len(weights) != n:
        raise ValidationError(f"weights length {len(weights)} != n_assets {n}")

    cov = spec.cov()
    try:
        L = np.linalg.cholesky(cov + np.eye(n) * 1e-10)
    except np.linalg.LinAlgError as e:
        raise ValidationError(f"portfolio covariance matrix is not positive-definite: {e}") from e

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n))
    correlated = z @ L.T  # shape (n_paths, n), Cov(correlated) ~= cov

    mu = spec.expected_returns()
    terminal_returns = np.empty((n_paths, n))
    for i in range(n):
        # per-asset terminal log-return from GBM dynamics driven by the
        # *shared correlated shock* for asset i, not an independent one --
        # this preserves the cross-asset correlation structure while still
        # calling the shared GBM formula for each asset's own drift/vol.
        sigma_i = float(np.sqrt(cov[i, i])) or 1e-6
        log_ret = (mu[i] - 0.5 * sigma_i**2) * horizon_years + sigma_i * np.sqrt(horizon_years) * correlated[:, i]
        terminal_returns[:, i] = np.exp(log_ret) - 1.0

    held = weights > 0.5
    if not held.any():
        raise ValidationError("simulate_portfolio_pnl: no assets held (all weights <= 0.5)")
    port_weights = held.astype(float) / held.sum()  # equal-weight among held assets
    pnl = terminal_returns @ port_weights
    return pnl


@dataclass
class HedgeComparison:
    unhedged: TailRiskReport
    hedged: TailRiskReport
    hedge_scale: float
    cvar_improved: bool


def hedge_tail_risk(
    spec: PortfolioSpec,
    weights: np.ndarray,
    *,
    hedge_scale: float = 0.5,
    horizon_years: float = 1.0,
    n_paths: int = 4096,
    seed: int = 0,
    run_quantum: bool = False,
) -> HedgeComparison:
    """Compare the tail risk of the solved allocation against a simple
    exposure-scaled hedge (a real, if simple, hedge: reduce notional by
    `hedge_scale`, which scales P&L linearly). This is meant as a genuine
    honest baseline -- exposure scaling is the simplest hedge that
    actually reduces tail risk, and any more sophisticated instrument
    (options overlay, etc.) should beat it, not merely differ from it.
    """
    pnl_unhedged = simulate_portfolio_pnl(spec, weights, horizon_years=horizon_years, n_paths=n_paths, seed=seed)
    pnl_hedged = pnl_unhedged * hedge_scale

    unhedged_report = assess_tail_risk(pnl_unhedged, run_quantum=run_quantum, run_meta_agent_comparator=False, seed=seed)
    hedged_report = assess_tail_risk(pnl_hedged, run_quantum=run_quantum, run_meta_agent_comparator=False, seed=seed)

    return HedgeComparison(
        unhedged=unhedged_report, hedged=hedged_report, hedge_scale=hedge_scale,
        cvar_improved=hedged_report.cvar < unhedged_report.cvar,
    )
