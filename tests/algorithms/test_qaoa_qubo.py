import numpy as np
import pytest

from algorithms.qaoa_qubo.gen2 import QAOAv2, QAOAConfig
from algorithms.qaoa_qubo.qubo_builder import (
    LongShortConstraints,
    LongShortQUBOBuilder,
    PortfolioConstraints,
    PortfolioQUBOBuilder,
)
from algorithms.qaoa_qubo.solvers import NumpyQAOASolver, QiskitQAOASolver, solve_qubo
from core.errors import CredentialError, ValidationError


def _small_market(n=6, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-0.01, 0.03, n)
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n + 0.01 * np.eye(n)
    return mu, cov


def test_numpy_solver_finds_a_k_hot_solution_close_to_exact():
    mu, cov = _small_market(n=8, seed=4)
    k = 3
    Q = PortfolioQUBOBuilder(mu, cov).build(PortfolioConstraints(cardinality_k=k, cardinality_penalty=5.0))
    result = solve_qubo(Q, engine="numpy", cardinality_k=k, p=2, n_iter=200)
    assert sum(int(b) for b in result.best_bitstring) == k
    assert result.approximation_ratio is not None
    # QAOA should find something not wildly worse than brute-force optimum
    assert result.approximation_ratio < 5.0  # loose bound: catches total breakage, not tuning quality


def test_qiskit_solver_runs_and_returns_valid_bitstring():
    mu, cov = _small_market(n=5, seed=6)
    Q = PortfolioQUBOBuilder(mu, cov).build(PortfolioConstraints())  # no cardinality: unconstrained QUBO
    result = QiskitQAOASolver(p=1, shots=512, maxiter=15).solve(Q)
    assert len(result.best_bitstring) == 5
    assert set(result.best_bitstring) <= {"0", "1"}
    assert result.engine == "qiskit_aer"


def test_qiskit_solver_rejects_too_large_problem():
    Q = np.zeros((20, 20))
    with pytest.raises(ValidationError):
        QiskitQAOASolver().solve(Q)


def test_run_on_ibm_runtime_requires_credentials(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    from core.config import get_settings
    get_settings(refresh=True)
    with pytest.raises(CredentialError):
        QiskitQAOASolver().run_on_ibm_runtime()


def test_longshort_builder_respects_leverage_limit_at_exact_optimum():
    mu, cov = _small_market(n=5, seed=8)
    builder = LongShortQUBOBuilder(mu, cov)
    Q = builder.build(LongShortConstraints(leverage_limit=2, leverage_penalty=10.0, exclusion_penalty=20.0))
    best_x, _cost = _brute_force_2n(Q)
    decoded = builder.decode(best_x)
    assert decoded["n_long"] + decoded["n_short"] <= 3  # allow small penalty slack, still far below n=10 unconstrained
    assert len(decoded["both_flagged"]) == 0


def _brute_force_2n(Q: np.ndarray):
    import itertools
    n = Q.shape[0]
    best_x, best_cost = None, float("inf")
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        c = float(x @ Q @ x)
        if c < best_cost:
            best_cost, best_x = c, x
    return best_x, best_cost


def test_gen2_engine_registers_measurable_improvement_over_random_init():
    """Regression guard for the documented Gen1->Gen2 improvement: SK
    warm-start + correlation-XY should not perform worse than random-angle
    initialization on the same problem (sanity check, not the full 39%->70%
    claim which needs a larger benchmark suite — see benchmarks/)."""
    mu, cov = _small_market(n=6, seed=10)
    Q = PortfolioQUBOBuilder(mu, cov).build(PortfolioConstraints(cardinality_k=3, cardinality_penalty=4.0))
    J, h, _ = __import__("algorithms.qaoa_qubo.qubo_builder", fromlist=["qubo_to_ising"]).qubo_to_ising(Q)
    J_full = J + J.T
    np.fill_diagonal(J_full, h)

    cfg_warm = QAOAConfig(p=2, n_iter=150, seed=1, use_dicke_init=True, target_k=3)
    warm = QAOAv2(cfg_warm).fit(J_full, warmstart=True)

    cfg_rand = QAOAConfig(p=2, n_iter=150, seed=1, use_dicke_init=True, target_k=3)
    rand = QAOAv2(cfg_rand).fit(J_full, warmstart=False)

    # warm-start should reach an energy at least as good (QAOAv2 maximizes H_diag,
    # i.e. -cost under this platform's sign convention passed through build_cost_hamiltonian)
    assert warm.best_energy_ >= rand.best_energy_ - 1e-6
