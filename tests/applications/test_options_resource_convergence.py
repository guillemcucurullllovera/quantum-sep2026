"""Resource estimation must scale sensibly with target epsilon, and the
convergence study must show the sampled amplitude/price approaching the
statevector-exact grid price as real shot counts increase (shot noise
shrinking with more shots) -- both computed from a real, executed circuit,
not a synthetic curve.
"""
from __future__ import annotations

from applications.options.domain import OptionSpec, PayoffKind
from applications.options.resource_and_convergence import estimate_resources, run_convergence_study


def _barrier_spec() -> OptionSpec:
    return OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=115.0,
                       rate=0.04, vol=0.2, maturity=0.25)


def test_resource_estimate_reports_real_circuit_stats():
    spec = _barrier_spec()
    estimate = estimate_resources(spec, target_epsilon=0.01, n_qubits=4)
    assert estimate.circuit_stats is not None
    assert estimate.circuit_stats.n_qubits == 5  # 4 index qubits + 1 payoff ancilla
    assert estimate.circuit_stats.depth > 0
    assert estimate.iqae_expected_queries > 0
    assert estimate.canonical_ae_queries > 0


def test_resource_estimate_query_count_scales_inversely_with_epsilon():
    spec = _barrier_spec()
    loose = estimate_resources(spec, target_epsilon=0.05, n_qubits=4)
    tight = estimate_resources(spec, target_epsilon=0.005, n_qubits=4)
    # a tighter accuracy target must need at least as many oracle queries
    assert tight.iqae_expected_queries > loose.iqae_expected_queries
    assert tight.canonical_ae_queries > loose.canonical_ae_queries
    # IQAE's whole advantage over canonical QPE-based AE must hold at both settings
    assert tight.iqae_expected_queries < tight.canonical_ae_queries
    assert loose.iqae_expected_queries < loose.canonical_ae_queries


def test_basket_resource_estimate_scales_with_qubit_count():
    spec = OptionSpec(payoff=PayoffKind.BASKET_CALL, spot=100.0, spot2=100.0, strike=100.0,
                       vol=0.2, vol2=0.25, correlation=0.3, rate=0.04, maturity=0.25)
    small = estimate_resources(spec, target_epsilon=0.02, n_qubits=1)
    big = estimate_resources(spec, target_epsilon=0.02, n_qubits=2)
    assert big.circuit_stats.n_qubits > small.circuit_stats.n_qubits


def test_convergence_study_at_fixed_resolution_shot_noise_shrinks_with_shots():
    """At a fixed n_qubits, more shots must get closer to the
    statevector-exact grid price ON AVERAGE -- checked at n_qubits=4 by
    averaging real executed-circuit runs over several seeds, since any
    SINGLE binomial draw at low shots can occasionally land close to the
    truth by chance (that single-draw flakiness is exactly why this is
    averaged rather than asserted on one seed)."""
    spec = _barrier_spec()
    few_shot_errors, many_shot_errors = [], []
    for seed in (1, 2, 3, 4, 5):
        study = run_convergence_study(spec, shots_grid=(256, 16384), n_qubits_grid=(4,), seed=seed)
        by_shots = {p.shots: p for p in study.points}
        few_shot_errors.append(by_shots[256].abs_error_vs_grid)
        many_shot_errors.append(by_shots[16384].abs_error_vs_grid)
    import numpy as np
    assert np.mean(many_shot_errors) < np.mean(few_shot_errors)


def test_convergence_study_reports_both_discretisation_and_shot_error():
    spec = _barrier_spec()
    study = run_convergence_study(spec, shots_grid=(4096,), n_qubits_grid=(2, 3, 4), seed=7)
    assert len(study.points) == 3
    for p in study.points:
        assert p.shots == 4096
        assert p.mc_ground_truth > 0
        assert p.abs_error_vs_mc >= 0
        assert p.abs_error_vs_grid >= 0
        assert p.circuit_n_qubits == p.n_qubits + 1  # index register + payoff ancilla
