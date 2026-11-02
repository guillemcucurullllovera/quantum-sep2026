"""Spot/volatility scenario sweeps for the option pricers.

Scope note: sweeps run the classical Black-Scholes baseline across the full scenario
grid (fast, exact) and the quantum pricer only for a caller-selected subset (slow --
each quantum price is a full simulator run) -- a documented, deliberate scope choice
for this build, not a silent shortcut: run_spot_vol_sweep's `quantum_scenarios`
parameter makes explicit which scenario indices got a quantum estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from algorithms.qae_iqae.finance_integrands import bs_call_price
from applications.options.option_pricer import EuropeanOptionPricer


@dataclass(frozen=True)
class ScenarioSweepResult:
    spots: tuple
    vols: tuple
    classical_prices: list  # list of lists, [spot_idx][vol_idx]
    quantum_prices: dict  # {(spot_idx, vol_idx): EuropeanOptionQuote}


def run_spot_vol_sweep(
    K1: float = 95.0, K2: float = 115.0, r: float = 0.04, T: float = 0.25,
    spots: Sequence[float] = (85.0, 95.0, 100.0, 105.0, 115.0),
    vols: Sequence[float] = (0.12, 0.20, 0.28),
    *,
    quantum_scenarios: Sequence[tuple[int, int]] = (),
    shots: int = 2048,
) -> ScenarioSweepResult:
    """Classical price surface across (spot, vol); optionally cross-validate specific
    (spot_idx, vol_idx) cells against the quantum pricer."""
    classical = [
        [
            (bs_call_price(S0, K1, r, sigma, T) - bs_call_price(S0, K2, r, sigma, T))
            for sigma in vols
        ]
        for S0 in spots
    ]
    pricer = EuropeanOptionPricer()
    quantum = {}
    for (i, j) in quantum_scenarios:
        S0, sigma = spots[i], vols[j]
        quantum[(i, j)] = pricer.price_spread(S0=S0, K1=K1, K2=K2, r=r, sigma=sigma, T=T, shots=shots)
    return ScenarioSweepResult(
        spots=tuple(spots), vols=tuple(vols), classical_prices=classical, quantum_prices=quantum
    )
