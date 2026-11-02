"""Composition point: run a scenario grid through
`applications.tail_risk.comparison.compare_methods`, so every scenario in
the grid gets a real multi-method VaR/CVaR assessment, not just a summary
statistic.

Scope note, mirroring `applications/derivatives/scenario_analysis.py`'s
`quantum_scenarios` pattern: the QAE method inside `compare_methods` is a
full simulator run (several amplitude-estimation calls) per scenario, so it
is only enabled for `quantum_scenarios` (a caller-selected subset of
scenario names), not the whole grid, by default.
"""
from __future__ import annotations

import numpy as np

from applications.scenario_analysis.domain import ScenarioSpec
from applications.scenario_analysis.generator import ScenarioGenerator
from applications.scenario_analysis.runner import ScenarioGridReport, ScenarioResultRow
from applications.tail_risk.comparison import compare_methods
from applications.tail_risk.domain import TailRiskSpec
from core.registry import APPLICATION_REGISTRY

__all__ = ["run_tail_risk_scenarios"]


def run_tail_risk_scenarios(
    base_pnl: np.ndarray,
    scenarios: list[ScenarioSpec],
    *,
    spec: TailRiskSpec | None = None,
    quantum_scenarios: tuple[str, ...] = (),
    seed: int = 42,
) -> ScenarioGridReport:
    """Shock `base_pnl` under every scenario, then run the full multi-method
    tail-risk comparison on each shocked series.

    If `quantum_scenarios` is empty (the default), `spec.run_quantum`
    applies uniformly to the whole grid (False if `spec` is omitted, to
    keep a full grid fast by default). If `quantum_scenarios` is non-empty,
    it *overrides* `spec.run_quantum` per scenario: only the named
    scenarios run the QAE method, every other scenario runs without it --
    the same explicit-subset-selection pattern
    `applications/derivatives/scenario_analysis.py`'s `quantum_scenarios`
    parameter uses for its own per-cell quantum opt-in.
    """
    base_spec = spec or TailRiskSpec(run_quantum=False)
    gen = ScenarioGenerator(seed=seed)
    grid = gen.generate_grid(base_pnl, scenarios, seed=seed)

    rows: list[ScenarioResultRow] = []
    for scenario in scenarios:
        shocked = grid[scenario.name]
        use_quantum = (scenario.name in quantum_scenarios) if quantum_scenarios else base_spec.run_quantum
        this_spec = (
            base_spec.model_copy(update={"run_quantum": use_quantum})
            if use_quantum != base_spec.run_quantum
            else base_spec
        )
        report = compare_methods(shocked, this_spec)
        rows.append(
            ScenarioResultRow(
                scenario=scenario.name, description=scenario.describe(),
                n_samples=int(shocked.shape[0]), risk=report,
            )
        )
    return ScenarioGridReport(base_n_samples=int(np.asarray(base_pnl).shape[0]), rows=rows)


def _register() -> None:
    if "scenario_analysis_tail_risk" not in APPLICATION_REGISTRY.list():
        APPLICATION_REGISTRY.register("scenario_analysis_tail_risk")(run_tail_risk_scenarios)


_register()
