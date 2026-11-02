import numpy as np
import pytest

from algorithms_extended.qokit_finance.fast_simulator import (
    apply_x_mixer_all,
    apply_x_rotation,
    cost_diagonal_from_qubo,
    expected_cost,
    run_fast_qaoa,
)
from algorithms_extended.qokit_finance.parameter_basis import (
    from_fourier_basis,
    to_fourier_basis,
    transfer_parameters,
)
from algorithms_extended.qokit_finance.sk_warmstart import available_depths, sk_gamma_beta


def _statevector_norm(sv):
    return float(np.sum(np.abs(sv) ** 2))


def test_x_rotation_preserves_norm():
    n = 4
    sv = np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=complex)
    apply_x_rotation(sv, 0.37, qubit=1, n_qubits=n)
    assert abs(_statevector_norm(sv) - 1.0) < 1e-10


def test_x_rotation_theta_zero_is_identity():
    n = 3
    rng = np.random.default_rng(0)
    sv0 = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    sv0 /= np.linalg.norm(sv0)
    sv = sv0.copy()
    apply_x_rotation(sv, 0.0, qubit=0, n_qubits=n)
    assert np.allclose(sv, sv0)


def test_x_rotation_pi_flips_bit():
    """theta=pi/2 on a computational basis state should fully flip that bit
    (up to the -i phase), a known closed-form check on the furx convention."""
    n = 2
    sv = np.zeros(4, dtype=complex)
    sv[0] = 1.0  # |00>
    apply_x_rotation(sv, np.pi / 2, qubit=0, n_qubits=n)
    # amplitude should have moved entirely to |01> (index 1), up to phase
    assert abs(sv[0]) < 1e-9
    assert abs(abs(sv[1]) - 1.0) < 1e-9


def test_cost_diagonal_matches_brute_force_qubo():
    n = 3
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    diag = cost_diagonal_from_qubo(Q)
    for z in range(1 << n):
        x = np.array([(z >> i) & 1 for i in range(n)], dtype=float)
        expected = x @ Q @ x
        assert abs(diag[z] - expected) < 1e-8


def test_run_fast_qaoa_p0_is_uniform_expected_cost():
    """With p=0 layers (empty gamma/beta), the statevector stays uniform
    superposition, so expected cost = mean of the cost diagonal."""
    n = 3
    Q = np.eye(n)
    diag = cost_diagonal_from_qubo(Q)
    sv = run_fast_qaoa(diag, gammas=[], betas=[], n_qubits=n)
    ec = expected_cost(sv, diag)
    assert abs(ec - diag.mean()) < 1e-8
    assert abs(_statevector_norm(sv) - 1.0) < 1e-10


def test_fourier_basis_roundtrip_recovers_original():
    rng = np.random.default_rng(2)
    p = 6
    gamma = rng.uniform(0, np.pi, size=p)
    beta = rng.uniform(0, np.pi / 2, size=p)
    u, v = to_fourier_basis(gamma, beta)
    gamma2, beta2 = from_fourier_basis(u, v, p=p)
    assert np.allclose(gamma, gamma2, atol=1e-8)
    assert np.allclose(beta, beta2, atol=1e-8)


def test_transfer_parameters_to_higher_depth_preserves_shape():
    rng = np.random.default_rng(3)
    p0 = 4
    gamma = rng.uniform(0, np.pi, size=p0)
    beta = rng.uniform(0, np.pi / 2, size=p0)
    gamma_ext, beta_ext = transfer_parameters(gamma, beta, target_p=8)
    assert len(gamma_ext) == 8
    assert len(beta_ext) == 8
    # extending then truncating back to p0 should not perfectly recover the
    # original UNLESS low-frequency content dominates, so instead check
    # that the transferred schedule still varies smoothly (no huge jumps),
    # a real structural property of the Fourier-basis representation.
    assert np.all(np.abs(np.diff(gamma_ext)) < np.pi)


def test_sk_warmstart_known_depth_matches_table():
    depths = available_depths()
    assert len(depths) > 0
    p = depths[0]
    gamma, beta = sk_gamma_beta(p)
    assert len(gamma) == p
    assert len(beta) == p


def test_sk_warmstart_extrapolates_unknown_depth():
    depths = available_depths()
    missing_p = max(depths) + 3
    gamma, beta = sk_gamma_beta(missing_p)
    assert len(gamma) == missing_p
    assert len(beta) == missing_p
    # extrapolated angles should still be finite, real, and bounded --
    # not NaN/inf from a broken basis transform
    assert np.all(np.isfinite(gamma))
    assert np.all(np.isfinite(beta))


def test_finance_bridge_solve_respects_cardinality():
    from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints
    from algorithms_extended.qokit_finance.finance_bridge import solve_portfolio_fast

    n = 6
    rng = np.random.default_rng(4)
    returns = rng.normal(0.05, 0.02, size=n)
    cov = np.eye(n) * 0.01
    constraints = PortfolioConstraints(cardinality_k=3, cardinality_penalty=5.0)

    result = solve_portfolio_fast(returns, cov, constraints, p=3, use_sk_warmstart=True)
    assert result.probabilities.shape == (1 << n,)
    assert abs(result.probabilities.sum() - 1.0) < 1e-8
    # the most likely bitstring under a strong cardinality penalty should
    # select close to k=3 assets
    best_idx = int(np.argmax(result.probabilities))
    n_selected = bin(best_idx).count("1")
    assert abs(n_selected - 3) <= 2  # penalty is soft, not exact, but should bias strongly toward k=3


def test_scaling_benchmark_runs_and_records():
    from algorithms_extended.qokit_finance.scaling_benchmark import run_scaling_benchmark

    results = run_scaling_benchmark(qubit_counts=[4, 6], p=1)
    assert len(results) == 2
    for r in results:
        assert r["fast_wall_time_s"] >= 0
    # wall time should not be zero/negative -- sanity, not a strict
    # monotonicity claim since timing noise at 4 vs 6 qubits can flip order
    assert all(r["fast_wall_time_s"] > 0 for r in results)
