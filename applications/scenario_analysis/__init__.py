"""Cross-application scenario analysis: a `ScenarioGenerator` producing real,
distinct parametrized shocks (vol spike, correlation breakdown, mean shift,
jump), a grid runner that plugs any downstream risk pipeline in as
`risk_fn`, and a tail-risk composition adapter wiring it directly into
`applications.tail_risk`'s multi-method comparison.

`run_spot_vol_sweep` (options-specific, owned by
`applications/derivatives/scenario_analysis.py`) is re-exported here for
discoverability, not reimplemented -- this package is the general
framework, not an options-only one.
"""
from applications.derivatives.scenario_analysis import ScenarioSweepResult, run_spot_vol_sweep
from applications.scenario_analysis.domain import ScenarioSpec, ShockKind, ShockSpec
from applications.scenario_analysis.generator import ScenarioGenerator, default_scenario_grid
from applications.scenario_analysis.runner import ScenarioGridReport, ScenarioResultRow, run_scenario_grid
from applications.scenario_analysis.tail_risk_adapter import run_tail_risk_scenarios

__all__ = [
    "ShockKind",
    "ShockSpec",
    "ScenarioSpec",
    "ScenarioGenerator",
    "default_scenario_grid",
    "ScenarioResultRow",
    "ScenarioGridReport",
    "run_scenario_grid",
    "run_tail_risk_scenarios",
    "ScenarioSweepResult",
    "run_spot_vol_sweep",
]
