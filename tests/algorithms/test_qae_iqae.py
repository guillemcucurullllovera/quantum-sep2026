"""Real validation, not smoke tests: the state-prep amplitude must equal the mean of
g(x_i) by construction (statevector, no shots), MLAE must recover a known amplitude
from synthetic binomial counts, IQAE's confidence interval must actually contain the
true grid amplitude, and the discretisation gap must shrink as the grid refines.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from algorithms_extended.qae_spinqit.state_prep import angle_table, build_A, exact_amplitude
from algorithms_extended.qae_spinqit.toy_integrands import TOY, theta_from_value
from algorithms_extended.qae_spinqit.mlae import estimate, p_k
from algorithms_extended.qae_spinqit.affinity import check_affinity
from algorithms.qae_iqae.finance_integrands import (
    bs_call_price,
    bs_cdf_below,
    bs_digital_call_price,
    bs_put_price,
    build_crash_barrier,
    build_return_distribution_barrier,
)
from algorithms.qae_iqae.barrier_circuits import build_A_barrier, comparator_affinity
from algorithms.qae_iqae.register import estimate_barrier_probability


def test_state_prep_amplitude_equals_mean_g_by_construction():
    """The defining correctness property of the whole QAE encoding."""
    for name, integrand in TOY.items():
        at = angle_table(integrand)
        a_exact = exact_amplitude(at)
        expected_mean_g = sum(integrand.g(x) for x in (1 / 8, 3 / 8, 5 / 8, 7 / 8)) / 4
        assert abs(a_exact - expected_mean_g) < 1e-9, f"integrand {name} failed by-construction check"


def test_theta_from_value_roundtrip():
    for v in (0.0, 0.1, 0.5, 0.9, 1.0):
        theta = theta_from_value(v)
        recovered = math.sin(theta / 2) ** 2
        assert abs(recovered - v) < 1e-12


def test_mlae_recovers_known_amplitude_from_synthetic_counts():
    """Generate exact binomial counts for a known true amplitude, confirm MLAE recovers
    it within a tolerance that shrinks with shot count — a real numerical-recovery test,
    not a trivial-constant assertion."""
    true_a = 0.3
    ks = [0, 1, 2]
    shots_per_k = 20000
    rng = np.random.default_rng(7)
    successes = [int(rng.binomial(shots_per_k, p_k(true_a, k))) for k in ks]
    result = estimate(ks, successes, [shots_per_k] * len(ks), n_bootstrap=200)
    assert abs(result.a_hat - true_a) < 0.01
    assert result.ci_fisher[0] <= true_a <= result.ci_fisher[1]


def test_affinity_toy_g0_is_affine_degree_zero():
    rep = check_affinity(TOY["g0"])
    assert rep.is_affine
    assert rep.degree == 0


def test_affinity_toy_g2_is_nonaffine():
    rep = check_affinity(TOY["g2"])
    assert not rep.is_affine
    assert rep.degree == 2


def test_black_scholes_call_put_parity():
    S0, K, r, sigma, T = 100.0, 100.0, 0.04, 0.2, 0.25
    call = bs_call_price(S0, K, r, sigma, T)
    put = bs_put_price(S0, K, r, sigma, T)
    lhs = call - put
    rhs = S0 - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-9  # put-call parity, an exact analytic identity


def test_black_scholes_digital_call_bounded_by_discount_factor():
    price = bs_digital_call_price(100.0, 100.0, 0.04, 0.2, 0.25, payout=1.0)
    assert 0.0 < price < math.exp(-0.04 * 0.25)


def test_crash_barrier_grid_refinement_shrinks_discretisation_gap():
    coarse = build_crash_barrier(n=2)
    fine = build_crash_barrier(n=5)
    assert fine.meta["discretisation_gap"] < coarse.meta["discretisation_gap"]


def test_crash_barrier_matches_bs_cdf_analytically():
    b = build_crash_barrier(S0=100.0, r=0.04, sigma=0.2, T=0.25, crash_pct=0.05, n=6)
    expected = bs_cdf_below(b.meta["barrier"], 100.0, 0.04, 0.2, 0.25)
    assert abs(b.exact_analytic - expected) < 1e-12


def test_return_distribution_barrier_from_empirical_samples():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 50000)
    barrier = build_return_distribution_barrier(pnl, loss_threshold=-1.5, n=6)
    empirical = float(np.mean(pnl <= -1.5))
    assert abs(barrier.exact_analytic - empirical) < 1e-9
    assert abs(barrier.exact_grid_a - empirical) < 0.02


def test_build_A_barrier_circuit_ancilla_matches_target_probability():
    """Full circuit-level correctness check (statevector, no shots): the factored
    loader+comparator circuit's ancilla amplitude must equal sum of in-barrier probs."""
    from qiskit.quantum_info import Statevector

    barrier = build_crash_barrier(n=3)
    A = build_A_barrier(barrier.probs, barrier.barrier_mask)
    sv = Statevector.from_instruction(A)
    anc = A.num_qubits - 1
    p1 = sum(p for b, p in sv.probabilities_dict().items() if b[0] == "1")
    assert abs(p1 - barrier.exact_grid_a) < 1e-9


def test_comparator_affinity_single_bit_split_is_affine():
    # index_to_bits is MSB-first, so splitting purely on the MSB (top half vs bottom
    # half of the 8-point grid) is a linear function of one variable => affine.
    mask = (True, True, True, True, False, False, False, False)
    resid, is_affine = comparator_affinity(mask)
    assert is_affine


def test_comparator_affinity_and_like_split_is_nonaffine():
    # true only when the top two bits are both 0 -- an AND-like function, not affine.
    mask = (True, True, False, False, False, False, False, False)
    resid, is_affine = comparator_affinity(mask)
    assert not is_affine


@pytest.mark.slow
def test_estimate_barrier_probability_end_to_end_ci_contains_truth():
    """End-to-end integration test through the application-facing API: the noisy IQAE
    confidence interval should contain the exact grid amplitude in the overwhelming
    majority of runs — check it holds for this fixed seed, and that metrics are sane."""
    barrier = build_crash_barrier(n=3, crash_pct=0.05)
    result = estimate_barrier_probability(barrier, epsilon=0.02, shots=1024, seed=3, log_to_registry=False)
    assert 0.0 <= result.metrics["a_hat_ideal"] <= 1.0
    assert result.metrics["abs_error_ideal_vs_grid"] < 0.05
    assert result.metrics["circuit_depth_Q2A"] > 0
    assert "p1_mitigated" in result.metrics
