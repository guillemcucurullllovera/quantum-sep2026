"""Shared metric definitions -- Track A and Track B MUST use these exact
formulas (not independently re-derive equivalent-looking ones) so that
"objective gap" and "approximation ratio" mean the same thing regardless
of which solver produced the compared selection. See
docs/quantum/QUANTUM1B_PREREGISTRATION.md EXP-B1 for the metric list this
implements.
"""
from __future__ import annotations

from dataclasses import dataclass

from experiments.quantum1b.instance import OptionSelectionInstance, SelectionEvaluation, evaluate_selection


def objective_gap(optimal_objective: float, candidate_objective: float) -> float:
    """(opt - candidate) / |opt|, 0 when candidate matches optimal, > 0
    when candidate is worse. Undefined (returns None) when opt == 0 (the
    degenerate case where NO_TRADE is itself optimal) -- callers must
    handle that explicitly rather than dividing by zero."""
    if abs(optimal_objective) < 1e-12:
        return None  # type: ignore[return-value]
    return (optimal_objective - candidate_objective) / abs(optimal_objective)


def approximation_ratio(optimal_objective: float, candidate_objective: float) -> float:
    """candidate / opt, 1.0 at optimum, < 1.0 for a feasible worse
    solution. Undefined (returns None) when opt == 0, same as
    `objective_gap`. Meaningful only for feasible candidate selections --
    callers must check `SelectionEvaluation.feasible` first."""
    if abs(optimal_objective) < 1e-12:
        return None  # type: ignore[return-value]
    return candidate_objective / optimal_objective


@dataclass
class SampleSetMetrics:
    n_samples: int
    feasible_fraction: float
    best_objective: float
    best_feasible_objective: float | None
    expected_objective: float
    cvar_objective: float
    constraint_violation_rate: float
    n_distinct_top_decile: int
    mean_risk_budget_utilization: float


def evaluate_sample_set(
    instance: OptionSelectionInstance,
    samples: list[tuple[int, ...]],
    *,
    cvar_alpha: float = 1.0,
) -> SampleSetMetrics:
    """Evaluate a batch of sampled 0/1 selections (e.g. QAOA measurement
    outcomes) against the true constrained objective. `cvar_alpha` is the
    fraction of samples (by minimized cost = -objective, i.e. best
    fraction by objective) averaged for the CVaR objective -- alpha=1.0
    reduces to the plain expectation.

    This is the SINGLE place CVaR is computed from a sample set, used
    identically by the QAOA optimizer's own training-time objective
    (Track B) and by any post-hoc reporting, so "CVaR objective" always
    means the same thing."""
    if not samples:
        raise ValueError("evaluate_sample_set: samples must be non-empty")
    if not (0.0 < cvar_alpha <= 1.0):
        raise ValueError(f"cvar_alpha must be in (0, 1], got {cvar_alpha}")

    evals: list[SelectionEvaluation] = [evaluate_selection(instance, s) for s in samples]
    n = len(evals)

    feasible_evals = [e for e in evals if e.feasible]
    feasible_fraction = len(feasible_evals) / n

    objectives = [e.objective for e in evals]
    best_objective = max(objectives)
    best_feasible_objective = max((e.objective for e in feasible_evals), default=None)
    expected_objective = sum(objectives) / n

    # CVaR: mean of the best alpha-fraction by objective (we are
    # MAXIMIZING objective, so CVaR-alpha averages the top alpha-fraction
    # of objective values -- the "good tail," matching Barkoutsos &
    # Nannicini 2020's convention translated from their cost-minimization
    # framing to this module's utility-maximization framing).
    k = max(1, round(cvar_alpha * n))
    top_k = sorted(objectives, reverse=True)[:k]
    cvar_objective = sum(top_k) / len(top_k)

    violation_count = sum(len(e.violations) for e in evals)
    constraint_violation_rate = violation_count / n

    top_decile_k = max(1, round(0.1 * n))
    top_decile_selections = sorted(evals, key=lambda e: -e.objective)[:top_decile_k]
    n_distinct_top_decile = len({e.selection for e in top_decile_selections})

    mean_rbu = sum(e.risk_budget_utilization for e in evals) / n

    return SampleSetMetrics(
        n_samples=n,
        feasible_fraction=feasible_fraction,
        best_objective=best_objective,
        best_feasible_objective=best_feasible_objective,
        expected_objective=expected_objective,
        cvar_objective=cvar_objective,
        constraint_violation_rate=constraint_violation_rate,
        n_distinct_top_decile=n_distinct_top_decile,
        mean_risk_budget_utilization=mean_rbu,
    )
