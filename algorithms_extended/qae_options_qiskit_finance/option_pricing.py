"""European call-spread pricing via MLAE, and digital-call pricing via the IQAE
barrier estimator. See THIRD_PARTY_NOTICES.md for exactly what's adapted from qiskit-finance's
conceptual pattern versus original to this platform.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from algorithms.qae_iqae.backends import ideal_sim, kingston_sim, run_counts, ancilla_p1
from algorithms.qae_iqae.finance_integrands import (
    BarrierIntegrand,
    bs_call_price,
    bs_digital_call_price,
    build_crash_barrier,
    call_spread_price_from_amplitude,
)
from algorithms.qae_iqae.register import estimate_barrier_probability
from algorithms_extended.qae_spinqit.state_prep import angle_table, build_qae_circuit
from algorithms_extended.qae_spinqit.toy_integrands import Integrand
from algorithms_extended.qae_spinqit.mlae import estimate as mlae_estimate


@dataclass(frozen=True)
class CallSpreadPricingResult:
    quantum_price_ideal: float
    quantum_price_noisy: float
    grid_price_at_exact_a: float
    true_bs_spread_price: float
    a_hat_ideal: float
    a_hat_noisy: float
    ci_fisher_ideal: tuple
    n_grid_points: int
    note: str


def run_call_spread_mlae(
    integrand: Integrand,
    *,
    ks=(0, 1, 2),
    shots: int = 4096,
    seed: int = 42,
    n_bootstrap: int = 500,
    skip_noisy: bool = False,
) -> CallSpreadPricingResult:
    """Price a European call spread by MLAE over the folded n=2 grid: run Q^k A for
    each k, feed the observed success counts to the MLAE estimator, map the recovered
    amplitude back to a price, and compare against the exact Black-Scholes spread
    price recorded in the integrand's metadata.

    `skip_noisy=True` skips the FakeKingston noisy-simulator pass (which costs ~10s
    just to build the noise model) and reuses the ideal result for the noisy fields --
    useful for fast iteration/tests that only care about the ideal-simulator pipeline
    correctness, not the noise-robustness comparison.
    """
    at = angle_table(integrand)
    ideal = ideal_sim()

    def _run(backend) -> tuple[list[int], list[int]]:
        successes, totals = [], []
        for k in ks:
            circ = build_qae_circuit(at, k)
            counts = run_counts(backend, circ, shots=shots, seed=seed)
            s, n = ancilla_p1(counts, ancilla_clbit=2)
            successes.append(s)
            totals.append(n)
        return successes, totals

    s_ideal, n_ideal = _run(ideal)
    res_ideal = mlae_estimate(list(ks), s_ideal, n_ideal, n_bootstrap=n_bootstrap, seed=seed)

    if skip_noisy:
        res_noisy = res_ideal
    else:
        noisy = kingston_sim()
        s_noisy, n_noisy = _run(noisy)
        res_noisy = mlae_estimate(list(ks), s_noisy, n_noisy, n_bootstrap=n_bootstrap, seed=seed)

    meta = integrand.meta
    return CallSpreadPricingResult(
        quantum_price_ideal=call_spread_price_from_amplitude(meta, res_ideal.a_hat),
        quantum_price_noisy=call_spread_price_from_amplitude(meta, res_noisy.a_hat),
        grid_price_at_exact_a=float(meta["grid_price_estimate"]),
        true_bs_spread_price=float(meta["true_bs_spread_price"]),
        a_hat_ideal=res_ideal.a_hat, a_hat_noisy=res_noisy.a_hat,
        ci_fisher_ideal=res_ideal.ci_fisher,
        n_grid_points=2 ** int(meta["n"]),
        note=str(meta["note"]),
    )


@dataclass(frozen=True)
class DigitalCallPricingResult:
    quantum_price_ideal: float
    true_bs_digital_price: float
    prob_above_strike_ideal: float
    exact_grid_prob_above: float
    discretisation_gap: float
    note: str


def digital_call_probability(
    S0: float, K: float, r: float, sigma: float, T: float, payout: float = 1.0,
    *, n: int = 3, epsilon: float = 0.01, shots: int = 2048,
) -> DigitalCallPricingResult:
    """Price a cash-or-nothing digital call, payout*e^{-rT}*P(S_T > K), by reusing the
    IQAE barrier estimator for P(S_T <= K) and taking the complement -- an honest reuse
    of already-validated infrastructure rather than a third circuit construction."""
    barrier: BarrierIntegrand = build_crash_barrier(
        S0=S0, r=r, sigma=sigma, T=T, crash_pct=1.0 - K / S0, n=n,
    )
    result = estimate_barrier_probability(barrier, epsilon=epsilon, shots=shots, log_to_registry=False)
    prob_below_ideal = result.metrics["a_hat_ideal"]
    prob_above_ideal = 1.0 - prob_below_ideal
    disc = math.exp(-r * T)

    true_price = bs_digital_call_price(S0, K, r, sigma, T, payout=payout)
    exact_grid_prob_above = 1.0 - barrier.exact_grid_a

    return DigitalCallPricingResult(
        quantum_price_ideal=disc * payout * prob_above_ideal,
        true_bs_digital_price=true_price,
        prob_above_strike_ideal=prob_above_ideal,
        exact_grid_prob_above=exact_grid_prob_above,
        discretisation_gap=abs(exact_grid_prob_above - (1.0 - barrier.exact_analytic)),
        note=barrier.meta["note"],
    )
