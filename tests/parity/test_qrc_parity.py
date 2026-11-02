"""Parity/invariant tests for the QRC math core — these validate physical
correctness (Hermiticity, unitarity, trace preservation), not forecasting
accuracy (that's tests/algorithms/test_qrc.py and THIRD_PARTY_NOTICES.md's honest
result history).
"""
import numpy as np

from algorithms.qrc.reservoir_math import (
    build_ising_hamiltonian,
    encode_input_state,
    evolution_operator,
    partial_trace_input,
    precompute_z_diagonals,
    qrc_forward,
    random_couplings,
)
from algorithms.qrc.reservoir import evolution_operator_stable, random_couplings as random_couplings_fixed


def test_hamiltonian_is_hermitian():
    rng = np.random.default_rng(0)
    J = random_couplings(4, rng)
    H = build_ising_hamiltonian(4, J)
    assert np.allclose(H, H.conj().T)


def test_evolution_operator_is_unitary():
    rng = np.random.default_rng(1)
    J = random_couplings(4, rng)
    H = build_ising_hamiltonian(4, J)
    U = evolution_operator(H, tau=1.0)
    identity = np.eye(U.shape[0])
    assert np.allclose(U @ U.conj().T, identity, atol=1e-8)


def test_evolution_operator_stable_matches_expm_version():
    """Gen2's eigendecomposition-based U(tau) must agree with the scipy.expm
    version used in the paper-faithful port, since they claim to compute the
    same operator via a different (faster, repeated-call-friendly) method."""
    rng = np.random.default_rng(2)
    J = random_couplings(4, rng)
    H = build_ising_hamiltonian(4, J)
    U_expm = evolution_operator(H, tau=1.0)
    U_eigh = evolution_operator_stable(H, tau=1.0)
    assert np.allclose(U_expm, U_eigh, atol=1e-8)


def test_encoded_state_is_valid_density_matrix():
    angles = np.array([0.3, -0.7, 1.2])
    rho = encode_input_state(angles)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)  # Hermitian
    eigvals = np.linalg.eigvalsh(rho)
    assert eigvals.min() >= -1e-10  # PSD (it's a pure state, rank 1)


def test_partial_trace_preserves_total_trace():
    angles = np.array([0.1, 0.2])
    rho_input = encode_input_state(angles)
    rho_hidden = np.eye(2, dtype=np.complex128) / 2
    rho_full = np.kron(rho_input, rho_hidden)
    assert np.isclose(np.trace(rho_full), 1.0)
    rho_reduced = partial_trace_input(rho_full, n1=2, n2=1)
    assert np.isclose(np.trace(rho_reduced), 1.0)


def test_z_expectations_bounded():
    rng = np.random.default_rng(3)
    n1, n2 = 3, 2
    n = n1 + n2
    J = random_couplings(n, rng)
    H = build_ising_hamiltonian(n, J)
    U = evolution_operator(H, tau=1.0)
    inputs = rng.uniform(-np.pi, np.pi, size=(3, n1))
    z = qrc_forward(inputs, U, n1, n2)
    assert np.all(z >= -1.0 - 1e-9) and np.all(z <= 1.0 + 1e-9)


def test_fixed_couplings_spectral_radius_matches_alpha():
    """Regression guard for the Gen2 bug fix: spectral radius of the coupling
    matrix must equal the requested alpha, unlike the unfixed Gen1 version
    which has no such guarantee (documented limitation, see THIRD_PARTY_NOTICES.md)."""
    rng = np.random.default_rng(4)
    J = random_couplings_fixed(6, rng, alpha=0.7)
    radius = np.max(np.abs(np.linalg.eigvalsh(J)))
    assert np.isclose(radius, 0.7, atol=1e-9)


def test_unfixed_gen1_couplings_do_not_guarantee_bounded_radius():
    """Documents the known Gen1 limitation directly (not just asserted in
    prose): confirms the un-normalized version can exceed radius 1, which is
    exactly why the fix in algorithms/qrc/reservoir.py exists."""
    rng = np.random.default_rng(5)
    J = random_couplings(10, rng)  # unfixed, U(0,1), no normalization
    radius = np.max(np.abs(np.linalg.eigvalsh(J)))
    assert radius > 1.0  # this is the bug the Gen2 fix addresses
