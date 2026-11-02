"""Tests for the factored crash-barrier construction, IQAE, comparator affinity, and the erosion
audit. Require qiskit / qiskit-algorithms (skipped where absent so the shared-venv run doesn't
error).
"""
import math

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_algorithms")

from algorithms.qae_iqae.finance_integrands import build_crash_barrier  # noqa: E402
from algorithms.qae_iqae.barrier_circuits import build_A_barrier, comparator_affinity  # noqa: E402
from algorithms_extended.qae_spinqit.quadrature import index_to_bits # noqa: E402
from algorithms_extended.qae_spinqit.state_prep import _ancilla_prob # noqa: E402


# ---- factored barrier construction -------------------------------------------------


@pytest.mark.parametrize("cp", [0.05, 0.10, 0.15])
def test_barrier_amplitude_equals_grid_probability(cp):
    b = build_crash_barrier(crash_pct=cp, n=3)
    assert abs(sum(b.probs) - 1.0) < 1e-9
    a = _ancilla_prob(build_A_barrier(b.probs, b.barrier_mask))
    assert abs(a - b.exact_grid_a) < 1e-9  # a = Σ_{j in mask} p_j, exactly


def test_barrier_analytic_cdf_is_monotone_in_crash_depth():
    shallow = build_crash_barrier(crash_pct=0.05).exact_analytic
    deep = build_crash_barrier(crash_pct=0.20).exact_analytic
    assert 0.0 < deep < shallow < 1.0


def test_barrier_probs_in_range():
    b = build_crash_barrier(crash_pct=0.05, n=3)
    assert all(0.0 <= p <= 1.0 for p in b.probs)
    assert b.meta["n_below_barrier"] == sum(b.barrier_mask)


# ---- comparator affinity -----------------------------------------------------------


def test_comparator_affine_on_bit_boundary_but_not_general():
    # a barrier that splits on the top bit (MSB=0 in-barrier) is an affine indicator
    mask_affine = tuple(index_to_bits(j, 3)[0] == 0 for j in range(8))
    _, is_affine = comparator_affinity(mask_affine)
    assert is_affine
    # a generic prefix threshold (i <= 2) is NOT affine
    mask_general = tuple(j <= 2 for j in range(8))
    _, is_general_affine = comparator_affinity(mask_general)
    assert not is_general_affine


# ---- IQAE recovers the amplitude ----------------------------------------------------


def test_iqae_ideal_recovers_barrier_grid_amplitude():
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from algorithms.qae_iqae.iqae import run_iqae

    b = build_crash_barrier(crash_pct=0.05, n=3)
    A = build_A_barrier(b.probs, b.barrier_mask)
    pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
    res = run_iqae(A, 3, sampler=SamplerV2(default_shots=1024, seed=5), transpiler=pm,
                   epsilon=0.03, alpha=0.05)
    assert abs(res["a_hat"] - b.exact_grid_a) < 0.03
    assert res["ci"][0] <= res["a_hat"] <= res["ci"][1]
    assert res["oracle_queries"] > 0


# ---- erosion audit ------------------------------------------------------------------


def test_erosion_importance_sampling_beats_plain_mc_on_tail():
    from algorithms.qae_iqae.erosion_audit import audit_importance_sampling
    B = audit_importance_sampling(crash_pct=0.20, N=2048, reps=30, seed=3)
    assert B["p_true"] < 0.05  # genuinely a tail event
    assert B["variance_reduction_factor"] > 5.0  # IS is much better on the tail


def test_erosion_herbert_loading_same_order():
    from algorithms.qae_iqae.erosion_audit import audit_herbert_loading
    C = audit_herbert_loading()
    ratios = [r["ratio_same_order"] for r in C["rows"]]
    assert max(ratios) / min(ratios) < 2.0  # both scale as 1/eps -> ~constant ratio
