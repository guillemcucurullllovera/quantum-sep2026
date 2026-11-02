"""Feasibility checking + repair for QAOA/QUBO results.

Reuses `algorithms.qaoa_qubo.classical_baseline.simulated_annealing_qubo`'s
existing `x0=`-seeded local-search mode (the same SA-polish pattern
`applications/asset_allocation/allocator.py` already uses) rather than
building a second penalty-aware local search from scratch. What's new here
is (1) an explicit, constraint-type-aware feasibility CHECK (SA-polish
alone has no notion of "is this actually feasible," it just minimizes QUBO
cost, which usually but not always yields feasibility) and (2) a
deterministic greedy fallback repair for the one case SA-polish handles
worst: an exact cardinality target, where flipping bits to minimize QUBO
cost can get stuck one bit short of/over the exact count.
"""
from __future__ import annotations

import numpy as np

from algorithms.qaoa_qubo.classical_baseline import simulated_annealing_qubo
from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints


def is_feasible(x: np.ndarray, constraints: PortfolioConstraints) -> dict[str, bool]:
    """Check each active constraint independently; returns a per-constraint
    feasibility report (not just a single bool) so callers can see exactly
    what's violated."""
    x = np.asarray(x, dtype=float)
    report: dict[str, bool] = {}

    if constraints.cardinality_k is not None:
        report["cardinality"] = int(round(x.sum())) == constraints.cardinality_k

    if constraints.previous_holdings is not None and constraints.turnover_limit is not None:
        p = np.asarray(constraints.previous_holdings, dtype=float)
        trades = int(np.sum(np.abs(x - p) > 0.5))
        report["turnover"] = trades <= constraints.turnover_limit

    for lim in constraints.sector_limits:
        idx = lim.asset_indices
        count = int(round(x[idx].sum()))
        ok = True
        if lim.max_count is not None:
            ok = ok and count <= lim.max_count
        if lim.min_count:
            ok = ok and count >= lim.min_count
        report[f"sector:{lim.name}"] = ok

    if constraints.liquidity_scores is not None:
        liq = np.asarray(constraints.liquidity_scores, dtype=float)
        selected = x > 0.5
        report["liquidity"] = bool(np.all(liq[selected] >= constraints.liquidity_floor)) if selected.any() else True

    return report


def repair_infeasible(
    Q: np.ndarray,
    x: np.ndarray,
    constraints: PortfolioConstraints,
    *,
    seed: int = 0,
    sa_polish_sweeps: int = 500,
) -> tuple[np.ndarray, dict[str, bool]]:
    """Attempt to repair an infeasible result. Two stages:

    1. SA-polish seeded from x (cheap local search, may already fix minor
       violations as a side effect of cost minimization).
    2. If cardinality is still wrong after stage 1, a deterministic greedy
       fix: if too many bits are set, drop the ones contributing the least
       favorable marginal cost first; if too few, add the most favorable.
       This targets the specific failure mode SA-polish handles worst
       (landing exactly on a hard count target), not a general-purpose
       constraint solver.
    """
    report = is_feasible(x, constraints)
    if all(report.values()):
        return x, report

    x_polished, _cost, _hist = simulated_annealing_qubo(Q, n_sweeps=sa_polish_sweeps, seed=seed, x0=x)
    report = is_feasible(x_polished, constraints)

    if constraints.cardinality_k is not None and not report.get("cardinality", True):
        x_polished = _greedy_cardinality_fix(Q, x_polished, constraints.cardinality_k)
        report = is_feasible(x_polished, constraints)

    return x_polished, report


def _greedy_cardinality_fix(Q: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    diag_only_cost = Q.diagonal()  # marginal single-asset cost proxy for tie-breaking

    while int(x.sum()) > k:
        selected = np.where(x > 0.5)[0]
        worst = selected[np.argmax(diag_only_cost[selected])]
        x[worst] = 0.0
    while int(x.sum()) < k:
        unselected = np.where(x <= 0.5)[0]
        if len(unselected) == 0:
            break
        best = unselected[np.argmin(diag_only_cost[unselected])]
        x[best] = 1.0

    return x
