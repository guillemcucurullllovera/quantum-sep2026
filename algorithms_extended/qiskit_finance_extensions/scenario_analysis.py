"""Spot/vol/barrier-level scenario sweeps for the barrier payoff integrand.

Kept inside this package rather than applications/derivatives/ (out of this
task's owned paths) — applications/derivatives/scenario_analysis.py
already covers vanilla European spot/vol sweeps; this is barrier-specific
and composes build_barrier_payoff_integrand directly, so a caller wanting a
single application-layer entry point can wrap both.
"""
from __future__ import annotations

from dataclasses import dataclass

from algorithms_extended.qiskit_finance_extensions.barrier_payoff import (
    BarrierKind,
    build_barrier_payoff_integrand,
)


@dataclass(frozen=True)
class BarrierScenarioRow:
    s0: float
    sigma: float
    barrier: float
    exact_grid_price: float
    monte_carlo_terminal_only: float
    monte_carlo_path_monitored: float
    vanilla_upper_bound: float


def run_barrier_scenario_sweep(
    kind: BarrierKind,
    *,
    spots: list[float],
    vols: list[float],
    barriers: list[float],
    strike: float = 100.0,
    r: float = 0.04,
    t: float = 0.25,
    n: int = 4,
    n_mc_paths: int = 20_000,
    seed: int = 42,
) -> list[BarrierScenarioRow]:
    from algorithms_extended.qiskit_finance_extensions.barrier_payoff import vanilla_reference_price

    rows = []
    for s0 in spots:
        for sigma in vols:
            for barrier in barriers:
                integrand = build_barrier_payoff_integrand(
                    kind, s0=s0, strike=strike, barrier=barrier, r=r, sigma=sigma, t=t, n=n,
                    n_mc_paths=n_mc_paths, seed=seed,
                )
                upper = vanilla_reference_price(s0, strike, r, sigma, t)
                rows.append(
                    BarrierScenarioRow(
                        s0=s0, sigma=sigma, barrier=barrier,
                        exact_grid_price=integrand.exact_grid_price,
                        monte_carlo_terminal_only=integrand.monte_carlo_terminal_only,
                        monte_carlo_path_monitored=integrand.monte_carlo_path_monitored,
                        vanilla_upper_bound=upper,
                    )
                )
    return rows
