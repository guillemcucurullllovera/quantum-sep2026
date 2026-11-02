"""Two-asset basket call payoff — "where practical" per the build spec.

Feasibility note (stated up front, not discovered after the fact): a basket
option on N assets needs a joint index register of sum(n_i) qubits, and the
per-index controlled-rotation architecture used here (and in
barrier_payoff.py) is O(2**sum(n_i)) gates to build the A-circuit. For 2
assets at n_a=n_b=2 qubits each (4x4=16 joint grid points) this is a small,
genuinely runnable circuit (4-qubit index register + 1 payoff ancilla, 16
controlled-RY gates). At 3 assets or >3 qubits per asset it stops being
practical with this construction (64+ gates and growing exponentially) —
that regime needs a proper quantum weighted-adder/QFT-based payoff circuit,
which is a real further-research item, not built here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from algorithms_extended.qiskit_finance_extensions.barrier_payoff import _flip_or_rotate_on_index
from algorithms_extended.qiskit_finance_extensions.uncertainty_models import (
    DiscretizedBivariateLogNormal,
)


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


@dataclass(frozen=True)
class BasketCallIntegrand:
    strike: float
    weight_a: float
    weight_b: float
    dist: DiscretizedBivariateLogNormal
    grid_a: tuple
    grid_b: tuple
    probs: tuple  # row-major, index = ib * 2**n_a + ia
    payoff_angle: tuple
    norm_h: float
    exact_grid_price: float
    monte_carlo_price: float
    monte_carlo_stderr: float


def build_basket_call_integrand(
    *,
    s0_a: float = 100.0,
    s0_b: float = 100.0,
    strike: float = 100.0,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    r: float = 0.04,
    sigma_a: float = 0.20,
    sigma_b: float = 0.25,
    rho: float = 0.3,
    t: float = 0.25,
    n_a: int = 2,
    n_b: int = 2,
    n_mc_paths: int = 200_000,
    seed: int = 42,
) -> BasketCallIntegrand:
    if n_a > 3 or n_b > 3:
        raise ValueError(
            "n_a, n_b > 3 makes the per-index controlled-rotation A-circuit "
            "impractically large (see module docstring) — not supported here."
        )
    dist = DiscretizedBivariateLogNormal.from_sigma_window(
        s0_a=s0_a, s0_b=s0_b, r=r, sigma_a=sigma_a, sigma_b=sigma_b, rho=rho, t=t,
        num_qubits_a=n_a, num_qubits_b=n_b,
    )
    grid_a, grid_b = dist.joint_grid()
    probs = dist.joint_probabilities()
    disc = math.exp(-r * t)

    raw_payoffs = []
    for sb in grid_b:
        for sa in grid_a:
            basket_value = weight_a * sa + weight_b * sb
            raw_payoffs.append(max(basket_value - strike, 0.0))

    h = max(raw_payoffs) or 1.0
    g = [_clip01(p / h) for p in raw_payoffs]
    angles = tuple(2.0 * math.asin(math.sqrt(x)) for x in g)
    exact_grid_price = disc * h * sum(p * x for p, x in zip(probs, g))

    mc_price, mc_stderr = _monte_carlo_basket_price(
        s0_a=s0_a, s0_b=s0_b, strike=strike, weight_a=weight_a, weight_b=weight_b,
        r=r, sigma_a=sigma_a, sigma_b=sigma_b, rho=rho, t=t, n_paths=n_mc_paths, seed=seed,
    )

    return BasketCallIntegrand(
        strike=strike, weight_a=weight_a, weight_b=weight_b, dist=dist,
        grid_a=tuple(grid_a), grid_b=tuple(grid_b), probs=tuple(probs),
        payoff_angle=angles, norm_h=h, exact_grid_price=exact_grid_price,
        monte_carlo_price=mc_price, monte_carlo_stderr=mc_stderr,
    )


def build_A_basket_payoff(integrand: BasketCallIntegrand):
    from qiskit import QuantumCircuit

    n_a = integrand.dist.num_qubits_a
    n_b = integrand.dist.num_qubits_b
    n = n_a + n_b
    qc = QuantumCircuit(n + 1, name="A_basket_call")
    qc.append(integrand.dist.state_preparation(), list(range(n)))
    anc = n
    for j, theta in enumerate(integrand.payoff_angle):
        if abs(theta) > 1e-12:
            _flip_or_rotate_on_index(qc, j, n, anc, theta)
    return qc


def price_from_amplitude(integrand: BasketCallIntegrand, a: float) -> float:
    disc = math.exp(-integrand.dist.r * integrand.dist.t)
    return disc * integrand.norm_h * a


def _monte_carlo_basket_price(
    *, s0_a, s0_b, strike, weight_a, weight_b, r, sigma_a, sigma_b, rho, t, n_paths, seed
) -> tuple[float, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z2 = rng.standard_normal(n_paths)
    za = z1
    zb = rho * z1 + math.sqrt(1 - rho**2) * z2

    s_a = sigma_a * math.sqrt(t)
    s_b = sigma_b * math.sqrt(t)
    drift_a = (r - 0.5 * sigma_a**2) * t
    drift_b = (r - 0.5 * sigma_b**2) * t

    terminal_a = s0_a * np.exp(drift_a + s_a * za)
    terminal_b = s0_b * np.exp(drift_b + s_b * zb)

    basket = weight_a * terminal_a + weight_b * terminal_b
    payoff = math.exp(-r * t) * np.clip(basket - strike, 0, None)
    return float(payoff.mean()), float(payoff.std(ddof=1) / (n_paths**0.5))
