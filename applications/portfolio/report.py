"""Human-readable reporting for a portfolio solve: chosen weights,
objective value, constraint satisfaction, which solver was used and why."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applications.portfolio.domain import PortfolioSpec
from applications.portfolio.multi_solver import MultiSolverComparison, SingleSolveResult
from applications.portfolio.validation import ValidationReport, validate_bitstring


@dataclass
class PortfolioReport:
    solver: str
    selected_assets: list[str]
    cost: float
    wall_time_s: float
    validation: ValidationReport
    summary: str


def build_report(spec: PortfolioSpec, result: SingleSolveResult) -> PortfolioReport:
    validation = validate_bitstring(spec, result.bitstring)
    selected = [spec.assets[i].name for i in range(spec.n_assets()) if result.bitstring[i] > 0.5]
    status = "OK, all constraints satisfied" if validation.ok else f"{len(validation.violations)} constraint(s) violated"
    summary = (
        f"Solver: {result.solver} | selected {len(selected)}/{spec.n_assets()} assets in "
        f"{result.wall_time_s * 1000:.1f}ms | cost={result.cost:.6f} | {status}"
    )
    return PortfolioReport(
        solver=result.solver, selected_assets=selected, cost=result.cost,
        wall_time_s=result.wall_time_s, validation=validation, summary=summary,
    )


def build_comparison_report(spec: PortfolioSpec, comparison: MultiSolverComparison) -> str:
    lines = [f"Multi-solver comparison ({len(comparison.results)} backends), best by QUBO cost: {comparison.best}"]
    if comparison.best_feasible is not None and comparison.best_feasible != comparison.best:
        lines[0] += f" (best FEASIBLE: {comparison.best_feasible})"
    for name, r in comparison.results.items():
        v = validate_bitstring(spec, r.bitstring)
        flag = "OK" if v.ok else f"VIOLATED: {v.violations}"
        lines.append(f"  {name:16s} cost={r.cost:12.6f}  time={r.wall_time_s * 1000:8.1f}ms  {flag}")
    return "\n".join(lines)


def build_walk_forward_report(results: dict[str, "object"]) -> str:
    """Honest cross-generation out-of-sample summary: which solver actually
    produced the best realized Sharpe ratio over a walk-forward backtest,
    not which one claims the best single-shot QUBO cost. `results` is the
    dict `applications.portfolio.scenario.compare_walk_forward` returns
    (solver name -> WalkForwardResult); typed loosely here to avoid a
    reporting-module -> scenario-module import cycle.
    """
    if not results:
        return "No walk-forward results to report."
    lines = ["Walk-forward comparison (out-of-sample):"]
    ranked = sorted(results.items(), key=lambda kv: kv[1].sharpe, reverse=True)
    best_name = ranked[0][0]
    lines[0] += f" best realized Sharpe: {best_name}"
    for name, r in ranked:
        lines.append(
            f"  {name:20s} sharpe={r.sharpe:8.3f}  max_dd={r.max_drawdown:8.3f}  "
            f"VaR95={r.var_95:8.4f}  CVaR95={r.cvar_95:8.4f}  "
            f"infeasible_periods={r.n_infeasible_periods}/{len(r.periods)}  "
            f"turnover_violations={r.n_turnover_violations}"
        )
    return "\n".join(lines)
