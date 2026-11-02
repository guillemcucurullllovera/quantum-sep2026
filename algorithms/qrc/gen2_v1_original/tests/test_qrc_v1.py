"""
Tests for QRC v1 — verifies the critical eigenvalue normalization bug fix
and correctness of the quantum reservoir forward pass.

All tests run with numpy only (no qiskit, no lightgbm required).
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from quantum.components.qrc.qrc_v1 import (
    random_couplings,
    build_ising_hamiltonian,
    evolution_operator,
    qrc_forward,
    precompute_z_diagonals,
    QRCConfig,
)


RNG = np.random.default_rng(42)


class TestEigenvalueNormalization:
    """
    CRITICAL: this test verifies the bug fix from phase0 _qrc.py.

    Phase0 bug: random_couplings() returned unnormalized U(0,1) values.
    For n=10, spectral radius ≈ 5, which violates the echo state property.
    Fix: normalize so max|λ(J)| = alpha.
    """

    @pytest.mark.parametrize("n, alpha", [
        (4, 0.9),
        (8, 0.9),
        (10, 0.9),
        (10, 0.95),
        (6, 0.7),
    ])
    def test_spectral_radius_equals_alpha(self, n, alpha):
        """max|eigenvalue(J)| must equal alpha within floating-point tolerance."""
        rng = np.random.default_rng(seed=n * 100 + int(alpha * 100))
        J = random_couplings(n, rng, alpha=alpha)
        eigvals = np.linalg.eigvalsh(J)
        spectral_radius = np.max(np.abs(eigvals))
        assert abs(spectral_radius - alpha) < 1e-10, (
            f"Spectral radius {spectral_radius:.6f} ≠ alpha {alpha}. "
            "Eigenvalue normalization fix not applied."
        )

    def test_coupling_matrix_is_symmetric(self):
        J = random_couplings(6, RNG, alpha=0.9)
        np.testing.assert_allclose(J, J.T, atol=1e-15,
                                   err_msg="Coupling matrix must be symmetric.")

    def test_coupling_matrix_zero_diagonal(self):
        J = random_couplings(6, RNG, alpha=0.9)
        np.testing.assert_allclose(np.diag(J), 0.0, atol=1e-15,
                                   err_msg="Diagonal must be zero (no self-coupling).")

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            random_couplings(4, RNG, alpha=1.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            random_couplings(4, RNG, alpha=0.0)


class TestHamiltonian:
    def test_hamiltonian_is_hermitian(self):
        n = 4
        J = random_couplings(n, RNG, alpha=0.9)
        H = build_ising_hamiltonian(n, J, field_nu=1.0)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12,
                                   err_msg="Ising Hamiltonian must be Hermitian.")

    def test_hamiltonian_shape(self):
        for n in [2, 4, 6]:
            J = random_couplings(n, RNG, alpha=0.9)
            H = build_ising_hamiltonian(n, J)
            assert H.shape == (2**n, 2**n), f"Wrong shape for n={n}: {H.shape}"


class TestEvolutionOperator:
    def test_evolution_is_unitary(self):
        """U U† must be identity (within numerical precision)."""
        n = 4
        J = random_couplings(n, RNG, alpha=0.9)
        H = build_ising_hamiltonian(n, J)
        U = evolution_operator(H, tau=1.0)
        eye_approx = U @ U.conj().T
        np.testing.assert_allclose(
            eye_approx, np.eye(2**n), atol=1e-10,
            err_msg="Evolution operator must be unitary: U U† = I"
        )

    def test_evolution_deterministic(self):
        """Same Hamiltonian + tau → same U."""
        n = 4
        J = random_couplings(n, np.random.default_rng(7), alpha=0.9)
        H = build_ising_hamiltonian(n, J)
        U1 = evolution_operator(H, tau=0.5)
        U2 = evolution_operator(H, tau=0.5)
        np.testing.assert_array_equal(U1, U2)


class TestQRCForward:
    def test_z_expectations_in_range(self):
        """Z-expectations must be in [-1, 1] (they are eigenvalues of Z ⊗ ... ⊗ Z)."""
        n1, n2 = 3, 3
        n = n1 + n2
        J = random_couplings(n, RNG, alpha=0.9)
        H = build_ising_hamiltonian(n, J)
        U = evolution_operator(H, tau=1.0)
        z_diags = precompute_z_diagonals(n)
        inputs = RNG.uniform(-np.pi, np.pi, (3, n1))
        z_exp = qrc_forward(inputs, U, n1, n2, z_diags=z_diags)
        assert z_exp.shape == (n,), f"Wrong shape: {z_exp.shape}"
        assert np.all(z_exp >= -1.0 - 1e-10) and np.all(z_exp <= 1.0 + 1e-10), (
            f"Z-expectations out of [-1,1]: min={z_exp.min():.4f} max={z_exp.max():.4f}"
        )

    def test_z_expectations_finite(self):
        """After eigenvalue fix, reservoir must produce finite values (no NaN/inf)."""
        n1, n2 = 3, 3
        n = n1 + n2
        J = random_couplings(n, RNG, alpha=0.9)
        H = build_ising_hamiltonian(n, J)
        U = evolution_operator(H, tau=1.0)
        inputs = RNG.uniform(-np.pi, np.pi, (3, n1))
        z_exp = qrc_forward(inputs, U, n1, n2)
        assert np.all(np.isfinite(z_exp)), (
            "Reservoir produced NaN or Inf. Check eigenvalue normalization."
        )

    def test_different_inputs_different_outputs(self):
        """Reservoir must respond differently to different inputs (non-trivial memory)."""
        n1, n2 = 3, 3
        n = n1 + n2
        J = random_couplings(n, np.random.default_rng(99), alpha=0.9)
        H = build_ising_hamiltonian(n, J)
        U = evolution_operator(H)
        inputs1 = np.zeros((3, n1))
        inputs2 = np.full((3, n1), np.pi / 2)
        z1 = qrc_forward(inputs1, U, n1, n2)
        z2 = qrc_forward(inputs2, U, n1, n2)
        assert not np.allclose(z1, z2), (
            "Reservoir gives identical output for different inputs — "
            "reservoir is not responding to input."
        )


class TestQRCConfig:
    def test_default_config(self):
        cfg = QRCConfig()
        assert cfg.n1 + cfg.n2 == 10
        assert 0 < cfg.alpha < 1
        assert cfg.tau > 0

    def test_backend_from_env(self, monkeypatch):
        monkeypatch.setenv("QUANTUM_BACKEND", "fake_heron")
        cfg = QRCConfig()
        assert cfg.backend == "fake_heron"
