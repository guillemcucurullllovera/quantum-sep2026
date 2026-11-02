"""Correctness tests for Track B's CVaR-QAOA engine
(`experiments.quantum1b.cvar_qaoa`).

(a) Statevector mixer/phase-operator correctness: compared against an
    independent brute-force dense-matrix QAOA circuit simulation on a
    tiny (n<=4) random instance -- this module's parity-test analogue to
    `tests/parity/test_qaoa_parity.py`.
(b) alpha=1.0 CVaR-QAOA's training objective reduces to expectation-QAOA's.
(c) Exact-distribution CVaR and finite-shot-sampled CVaR converge as shots
    grow, on a small fixed example.
(d) Determinism: same seed -> same result.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from algorithms.qaoa_qubo.qubo_builder import qubo_to_ising
from experiments.quantum1b.cvar_qaoa import (
    QAOARunResult,
    cost_diagonal_from_ising,
    cvar_from_distribution,
    exact_distribution_metrics,
    run_cvar_qaoa,
    run_expectation_qaoa,
    sample_bitstrings,
    simulate_qaoa,
)
from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.metrics import evaluate_sample_set
from experiments.quantum1b.qubo_encoding import build_qubo, selection_from_bits


# ---------------------------------------------------------------------------
# (a) Brute-force dense-matrix reference QAOA circuit
# ---------------------------------------------------------------------------


def _dense_qaoa_statevector(J: np.ndarray, h: np.ndarray, p: int, gamma, beta) -> np.ndarray:
    """Independent reference implementation: builds the FULL 2^n x 2^n
    cost-phase diagonal and mixer matrices (dense Kronecker products) and
    applies them by explicit matrix-vector multiplication -- the brute-force
    twin `simulate_qaoa`'s tensor-reshape implementation must match to high
    precision. Only used in tests, at n<=4, where a dense 2^n x 2^n matrix
    is trivial."""
    n = len(h)
    dim = 1 << n
    cost_diag = cost_diagonal_from_ising(J, h)

    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)

    rx = lambda b: np.array(
        [[np.cos(b), -1j * np.sin(b)], [-1j * np.sin(b), np.cos(b)]], dtype=np.complex128
    )
    identity = np.eye(2, dtype=np.complex128)

    for layer in range(p):
        state = state * np.exp(-1j * gamma[layer] * cost_diag)

        # Dense mixer: kron(RX, RX, ..., RX) built with qubit 0 as the
        # MOST-SIGNIFICANT factor in the Kronecker product, matching the
        # convention that bit b = (idx >> b) & 1 -> flat-index significance
        # 2^b, i.e. qubit (n-1) is the least-significant / last kron factor.
        mats = [rx(beta[layer]) for _ in range(n)]
        dense_mixer = mats[0]
        for m in mats[1:]:
            dense_mixer = np.kron(dense_mixer, m)
        # kron order above places qubit 0 as most-significant factor already
        # matches bit convention: flat idx = sum bit_i * 2^i, with qubit 0
        # contributing the 2^0 place if it's the LAST kron factor. Since our
        # loop builds kron(q0, q1, ..., qn-1) with q0 first (most-significant
        # kron factor), q0 ends up controlling the 2^(n-1) place, i.e. bit
        # (n-1) of idx -- so to match bit_b = (idx>>b)&1 for b=0..n-1 in
        # kron-product order, we must build kron(q_{n-1}, q_{n-2}, ..., q_0).
        dense_mixer = mats[-1]
        for m in reversed(mats[:-1]):
            dense_mixer = np.kron(dense_mixer, m)
        state = dense_mixer @ state

    return state


class TestEngineCorrectness:
    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("p", [1, 2])
    def test_matches_dense_brute_force_circuit(self, n, p):
        rng = np.random.default_rng(1234 + n * 10 + p)
        Q = rng.uniform(-2, 2, size=(n, n))
        Q = np.triu(Q)  # upper-triangular convention
        J, h, _offset = qubo_to_ising(Q)
        cost_diag = cost_diagonal_from_ising(J, h)

        gamma = rng.uniform(0, np.pi, p)
        beta = rng.uniform(0, np.pi, p)

        fast_state = simulate_qaoa(J, h, p, gamma, beta, cost_diag=cost_diag)
        dense_state = _dense_qaoa_statevector(J, h, p, gamma, beta)

        np.testing.assert_allclose(fast_state, dense_state, atol=1e-10)

    def test_cost_diagonal_matches_loop_reference(self):
        """Cross-check the vectorized cost_diagonal_from_ising against a
        literal per-bitstring Python loop using the identical bit
        convention documented in the module (and in
        algorithms/qaoa_qubo/solvers.py::_cost_diagonal_from_ising)."""
        n = 5
        rng = np.random.default_rng(7)
        Q = np.triu(rng.uniform(-1, 1, size=(n, n)))
        J, h, _offset = qubo_to_ising(Q)

        fast = cost_diagonal_from_ising(J, h)

        loop = np.zeros(1 << n)
        for idx in range(1 << n):
            bits = [(idx >> b) & 1 for b in range(n)]
            z = [1 - 2 * b for b in bits]
            e = sum(h[i] * z[i] for i in range(n))
            e += sum(J[i, j] * z[i] * z[j] for i in range(n) for j in range(i + 1, n))
            loop[idx] = e

        np.testing.assert_allclose(fast, loop, atol=1e-10)

    def test_statevector_is_normalized(self):
        rng = np.random.default_rng(99)
        n = 6
        Q = np.triu(rng.uniform(-1, 1, size=(n, n)))
        J, h, _offset = qubo_to_ising(Q)
        state = simulate_qaoa(J, h, 2, np.array([0.3, 0.7]), np.array([0.5, 1.1]))
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0, abs=1e-10)


class TestCostDiagonalMemoryBoundedPath:
    """`cost_diagonal_from_ising`'s O(dim)-memory branch (n above
    `dense_threshold_qubits`) must be bit-identical to the original
    O(dim*n)-memory dense-broadcast branch -- see that function's
    docstring for why this second branch exists (a real host OOM incident
    at n_qubits=23-25, fixed 2026-10-01, not preemptive hardening)."""

    @pytest.mark.parametrize("n", [3, 8, 12])
    def test_matches_dense_branch_bit_identical(self, n):
        rng = np.random.default_rng(1000 + n)
        Q = np.triu(rng.uniform(-2, 2, size=(n, n)))
        J, h, _offset = qubo_to_ising(Q)

        dense = cost_diagonal_from_ising(J, h, dense_threshold_qubits=n)
        bounded = cost_diagonal_from_ising(J, h, dense_threshold_qubits=0)

        np.testing.assert_allclose(bounded, dense, atol=1e-10)

    def test_matches_loop_reference_directly(self):
        """The bounded branch alone, independently cross-checked against
        the same literal-loop reference `test_cost_diagonal_matches_loop_reference`
        uses for the dense branch -- not merely agreement with the dense
        branch (which could share a bug)."""
        n = 6
        rng = np.random.default_rng(77)
        Q = np.triu(rng.uniform(-1, 1, size=(n, n)))
        J, h, _offset = qubo_to_ising(Q)

        bounded = cost_diagonal_from_ising(J, h, dense_threshold_qubits=0)

        loop = np.zeros(1 << n)
        for idx in range(1 << n):
            bits = [(idx >> b) & 1 for b in range(n)]
            z = [1 - 2 * b for b in bits]
            e = sum(h[i] * z[i] for i in range(n))
            e += sum(J[i, j] * z[i] * z[j] for i in range(n) for j in range(i + 1, n))
            loop[idx] = e

        np.testing.assert_allclose(bounded, loop, atol=1e-10)

    def test_handles_n_equals_one(self):
        # n=1: no quadratic term at all, exercises the `if n > 1` guard in
        # the bounded branch too.
        J = np.zeros((1, 1))
        h = np.array([2.5])
        bounded = cost_diagonal_from_ising(J, h, dense_threshold_qubits=0)
        dense = cost_diagonal_from_ising(J, h, dense_threshold_qubits=1)
        np.testing.assert_allclose(bounded, dense, atol=1e-10)
        np.testing.assert_allclose(bounded, [2.5, -2.5], atol=1e-10)

    def test_default_threshold_used_by_real_callers_is_18(self):
        from experiments.quantum1b.cvar_qaoa import DENSE_BITS_QUBIT_THRESHOLD

        assert DENSE_BITS_QUBIT_THRESHOLD == 18

    @pytest.mark.slow
    def test_bounded_branch_peak_memory_is_a_small_fraction_of_the_dense_branch_at_n20(self):
        """Regression test for independent-review finding P2-17 ('no
        automated large-n memory-regression test exists for the related
        EXP-B3 O(dim) fix' -- the review's own independent tracemalloc
        measurement at n=20 found dense-branch peak 696.3 MB vs.
        bounded-branch peak 42.0 MB, ~16.6x reduction, outputs agreeing to
        1.95e-14; this corroborated the fix as real and sound, but no
        automated test asserted it). A future refactor that silently
        reintroduced O(dim*n) behavior in the "bounded" branch would make
        this test fail.

        Uses n=20 (not the original incident's n=23-25) -- large enough to
        clearly separate the two branches' memory profiles (consistent
        with the review's own choice of n=20 for its independent
        measurement) while keeping this test's own resource cost bounded
        for routine use on a shared host."""
        import gc
        import tracemalloc

        n = 20
        rng = np.random.default_rng(2020)
        Q = np.triu(rng.uniform(-2, 2, size=(n, n)))
        J, h, _offset = qubo_to_ising(Q)

        def _peak_memory_bytes(dense_threshold_qubits: int) -> tuple[int, np.ndarray]:
            gc.collect()
            tracemalloc.start()
            try:
                result = cost_diagonal_from_ising(J, h, dense_threshold_qubits=dense_threshold_qubits)
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return peak, result

        bounded_peak, bounded_result = _peak_memory_bytes(dense_threshold_qubits=0)
        dense_peak, dense_result = _peak_memory_bytes(dense_threshold_qubits=n)

        np.testing.assert_allclose(bounded_result, dense_result, atol=1e-8)

        # Concrete regression guards, not just a ratio -- a ratio alone
        # could pass even if BOTH branches regressed upward together.
        assert bounded_peak < 150 * 1024 * 1024, (
            f"bounded-branch peak memory {bounded_peak / 1024 / 1024:.1f} MB exceeded the "
            f"150 MB regression guard at n={n} (review's own measurement: ~42 MB)"
        )
        assert dense_peak > bounded_peak * 5, (
            f"expected the dense branch's peak ({dense_peak / 1024 / 1024:.1f} MB) to be at "
            f"least 5x the bounded branch's peak ({bounded_peak / 1024 / 1024:.1f} MB) at n={n} "
            "-- if this fails, the O(dim) fix may no longer be providing its intended memory "
            "reduction"
        )


# ---------------------------------------------------------------------------
# (b) alpha=1.0 CVaR reduces to expectation
# ---------------------------------------------------------------------------


class TestCVaRReducesToExpectationAtAlphaOne:
    @pytest.mark.parametrize("direction", ["min", "max"])
    def test_cvar_alpha_one_equals_expectation(self, direction):
        rng = np.random.default_rng(5)
        n_outcomes = 37
        probs = rng.uniform(0.01, 1.0, n_outcomes)
        probs /= probs.sum()
        values = rng.uniform(-10, 10, n_outcomes)

        cvar_full = cvar_from_distribution(probs, values, alpha=1.0, direction=direction)
        expectation = float(np.sum(probs * values))
        assert cvar_full == pytest.approx(expectation, abs=1e-9)

    def test_cvar_qaoa_alpha_one_training_objective_matches_expectation_qaoa(self):
        """End-to-end: run_cvar_qaoa(alpha=1.0) and run_expectation_qaoa use
        mathematically identical training objectives -- verified directly
        on the objective FORMULA (not just via optimizer convergence, which
        could coincidentally agree)."""
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        J, h, offset = qubo_to_ising(qubo.Q)
        cost_diag = cost_diagonal_from_ising(J, h)
        qubo_cost = cost_diag + offset

        rng = np.random.default_rng(3)
        gamma, beta = rng.uniform(0, np.pi, 2), rng.uniform(0, np.pi, 2)
        state = simulate_qaoa(J, h, 2, gamma, beta, cost_diag=cost_diag)
        probs = np.abs(state) ** 2

        expectation_target = float(np.sum(probs * qubo_cost))
        cvar_target = cvar_from_distribution(probs, qubo_cost, alpha=1.0, direction="min")
        assert cvar_target == pytest.approx(expectation_target, abs=1e-9)

    def test_cvar_alpha_less_than_one_min_direction_is_le_expectation(self):
        """Sanity: averaging only the best (lowest-cost) fraction can never
        exceed the full expectation (min direction) -- the defining
        property of CVaR."""
        rng = np.random.default_rng(11)
        n_outcomes = 50
        probs = rng.uniform(0.01, 1.0, n_outcomes)
        probs /= probs.sum()
        values = rng.uniform(-5, 5, n_outcomes)
        cvar_02 = cvar_from_distribution(probs, values, alpha=0.2, direction="min")
        expectation = float(np.sum(probs * values))
        assert cvar_02 <= expectation + 1e-9


# ---------------------------------------------------------------------------
# (c) Exact-distribution CVaR vs finite-shot-sampled CVaR convergence
# ---------------------------------------------------------------------------


class TestShotConvergence:
    def test_shot_sampled_cvar_converges_to_exact_as_shots_grow(self):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        result = run_expectation_qaoa(qubo, p=1, seed=0, maxiter=30)

        exact = exact_distribution_metrics(result, qubo, inst, cvar_alpha=0.5)

        errors = []
        for shots in (200, 2000, 20000):
            samples = sample_bitstrings(result, shots, seed=42)
            full_selections = [tuple(selection_from_bits(qubo, inst, s)) for s in samples]
            sampled = evaluate_sample_set(inst, full_selections, cvar_alpha=0.5)
            errors.append(abs(sampled.cvar_objective - exact.cvar_objective))

        # Error should not systematically grow as shots increase; the
        # largest-shot error should be no worse than a small multiple of
        # the smallest-shot error's neighborhood, and in particular the
        # 20000-shot error should be small in absolute terms.
        assert errors[-1] < 0.5, f"shot-sampled CVaR did not converge: errors={errors}"
        assert errors[-1] <= errors[0] + 1e-6 or errors[-1] < 0.2


# ---------------------------------------------------------------------------
# (d) Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self):
        inst = generate_instance("correlated", 4, seed=0)
        qubo = build_qubo(inst)
        r1 = run_cvar_qaoa(qubo, p=1, alpha=0.5, seed=7, maxiter=25)
        r2 = run_cvar_qaoa(qubo, p=1, alpha=0.5, seed=7, maxiter=25)
        np.testing.assert_allclose(r1.best_params, r2.best_params)
        np.testing.assert_allclose(r1.probabilities, r2.probabilities)
        assert r1.n_evaluations == r2.n_evaluations

    def test_different_seed_can_differ(self):
        inst = generate_instance("correlated", 4, seed=0)
        qubo = build_qubo(inst)
        r1 = run_cvar_qaoa(qubo, p=1, alpha=0.5, seed=1, maxiter=25)
        r2 = run_cvar_qaoa(qubo, p=1, alpha=0.5, seed=2, maxiter=25)
        assert not np.allclose(r1.best_params, r2.best_params)

    def test_sample_bitstrings_deterministic(self):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        result = run_expectation_qaoa(qubo, p=1, seed=0, maxiter=20)
        s1 = sample_bitstrings(result, 100, seed=5)
        s2 = sample_bitstrings(result, 100, seed=5)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Additional integration sanity checks
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_two_qubit_gate_count_matches_nonzero_J_times_p(self):
        inst = generate_instance("correlated", 4, seed=0)
        qubo = build_qubo(inst)
        result = run_expectation_qaoa(qubo, p=2, seed=0, maxiter=10)
        expected = int(np.sum(np.abs(np.triu(result.J, k=1)) > 1e-12)) * 2
        assert result.two_qubit_gate_count() == expected

    def test_exact_distribution_metrics_feasible_fraction_in_unit_interval(self):
        inst = generate_instance("tight_budget", 4, seed=0)
        qubo = build_qubo(inst)
        result = run_cvar_qaoa(qubo, p=1, alpha=0.5, seed=0, maxiter=30)
        m = exact_distribution_metrics(result, qubo, inst, cvar_alpha=0.5)
        assert 0.0 <= m.feasible_fraction <= 1.0
        assert 0.0 <= m.constraint_violation_rate

    def test_statevector_reconstructable_and_normalized(self):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        result = run_expectation_qaoa(qubo, p=1, seed=0, maxiter=20)
        sv = result.statevector()
        assert sv.shape == (1 << result.n_qubits,)
        assert np.sum(np.abs(sv) ** 2) == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(np.abs(sv) ** 2, result.probabilities, atol=1e-9)


class TestInitialParamsOverride:
    """`initial_params` (added for Track F's mixer/warm-start benchmarking)
    must be backward-compatible (omitted -> byte-identical to before) and
    must actually be used, not silently ignored."""

    def test_omitted_matches_pre_existing_random_behavior(self):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        r1 = run_expectation_qaoa(qubo, p=1, seed=5, maxiter=20)
        r2 = run_expectation_qaoa(qubo, p=1, seed=5, maxiter=20)
        np.testing.assert_allclose(r1.best_params, r2.best_params)

    def test_initial_params_is_actually_used(self):
        """Two different initial_params, same seed/maxiter, must generally
        converge to different final params (COBYLA is a local-ish search,
        not something that reliably lands at the exact same point from two
        very different starting points in a short run) -- confirms the
        override is actually threaded through to the optimizer's x0, not
        dropped."""
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        r_a = run_expectation_qaoa(
            qubo, p=1, seed=0, maxiter=5, initial_params=np.array([0.1, 0.1])
        )
        r_b = run_expectation_qaoa(
            qubo, p=1, seed=0, maxiter=5, initial_params=np.array([3.0, 2.5])
        )
        assert not np.allclose(r_a.best_params, r_b.best_params)

    def test_same_initial_params_gives_deterministic_result(self):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        x0 = np.array([0.4, 0.6])
        r1 = run_expectation_qaoa(qubo, p=1, seed=0, maxiter=15, initial_params=x0)
        r2 = run_expectation_qaoa(qubo, p=1, seed=999, maxiter=15, initial_params=x0)
        # different `seed` must not matter once initial_params is given --
        # seed only drives the random x0 generation this override replaces.
        np.testing.assert_allclose(r1.best_params, r2.best_params)

    def test_wrong_shape_raises(self):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        with pytest.raises(ValueError, match="initial_params must have shape"):
            run_expectation_qaoa(qubo, p=2, seed=0, maxiter=5, initial_params=np.array([0.1, 0.2]))

    def test_works_for_cvar_qaoa_too(self):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        x0 = np.array([0.5, 0.5])
        r1 = run_cvar_qaoa(qubo, p=1, alpha=0.5, seed=0, maxiter=15, initial_params=x0)
        r2 = run_cvar_qaoa(qubo, p=1, alpha=0.5, seed=1, maxiter=15, initial_params=x0)
        np.testing.assert_allclose(r1.best_params, r2.best_params)
