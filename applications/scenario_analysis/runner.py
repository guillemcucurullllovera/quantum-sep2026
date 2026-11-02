"""Run a scenario grid through any downstream risk callable and aggregate
into one report -- the general cross-application composition point:
tail-risk, options, and portfolio pipelines all plug in as the `risk_fn`
argument, so this module owns none of the risk-metric logic itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from applications.scenario_analysis.domain import ScenarioSpec
from applications.scenario_analysis.generator import ScenarioGenerator

__all__ = ["ScenarioResultRow", "ScenarioGridReport", "run_scenario_grid"]


@dataclass(frozen=True)
class ScenarioResultRow:
    scenario: str
    description: str
    n_samples: int
    risk: Any  # whatever risk_fn returns for this scenario -- typed by the caller


@dataclass(frozen=True)
class ScenarioGridReport:
    base_n_samples: int
    rows: list[ScenarioResultRow]

    def by_name(self, name: str) -> ScenarioResultRow:
        for row in self.rows:
            if row.scenario == name:
                return row
        raise KeyError(f"no scenario named {name!r} in this grid")

    def scenario_names(self) -> list[str]:
        return [row.scenario for row in self.rows]


def run_scenario_grid(
    base: np.ndarray,
    scenarios: list[ScenarioSpec],
    risk_fn: Callable[[np.ndarray], Any],
    *,
    generator: ScenarioGenerator | None = None,
    seed: int = 42,
) -> ScenarioGridReport:
    """Shock `base` under every scenario in `scenarios` (via `generator`,
    default a fresh `ScenarioGenerator(seed=seed)`), run `risk_fn` on each
    shocked series, and collect results into one report.

    `risk_fn` is the composition point: pass
    `applications.tail_risk.comparison.compare_methods`, an options pricer,
    a portfolio risk function, or any `callable(np.ndarray) -> result`.
    """
    gen = generator or ScenarioGenerator(seed=seed)
    grid = gen.generate_grid(base, scenarios, seed=seed)
    rows = []
    for scenario in scenarios:
        shocked = grid[scenario.name]
        risk = risk_fn(shocked)
        rows.append(
            ScenarioResultRow(
                scenario=scenario.name, description=scenario.describe(),
                n_samples=int(np.asarray(shocked).shape[0]), risk=risk,
            )
        )
    return ScenarioGridReport(base_n_samples=int(np.asarray(base).shape[0]), rows=rows)
