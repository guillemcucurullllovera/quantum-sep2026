"""Reusable Quantum Reservoir Computing architecture.

Consolidates Asymmetrica's two development generations:
- Gen1 (`algorithms_extended/qrc_arxiv_replication`): the paper-faithful port,
    kept unmodified there for replication validity.
    - Gen2: found and
    fixed a real bug in coupling-matrix generation, and generalized the forward
    pass to a persistent-state (true recurrent) architecture.

This module is the canonical, reusable implementation going forward. It is
registered under core.registry.ALGORITHM_REGISTRY as "qrc".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms.qrc.reservoir_math import (
    build_ising_hamiltonian,
    encode_input_state,
    partial_trace_input,
    precompute_z_diagonals,
    qrc_forward,
)
from core.errors import ValidationError
from core.registry import ALGORITHM_REGISTRY


def random_couplings(n: int, rng: np.random.Generator, alpha: float = 0.9) -> np.ndarray:
    """Spectral-radius-normalized random coupling matrix.

BUG FIX vs. the Gen1/paper port (`algorithms_extended...reservoir_core.random_couplings`):
    drawing J_ij ~ U(0,1) with no normalization gives a spectral radius of
    O(n/2), which for n=10 is ~5 — far outside the edge-of-chaos regime that
    the echo-state property (bounded memory, non-divergent dynamics) requires.
    Normalizing to a target spectral radius alpha < 1 (default 0.9, just below
    the transverse field strength nu=1) places the system near the phase
    transition where reservoir memory capacity is maximized. This fix was
    found during Gen2 development; whether it also closes the gap documented
    in `THIRD_PARTY_NOTICES.md` for the paper
    replication specifically has NOT been re-tested — that is a legitimate
    follow-up experiment, not a completed one.
    """
    if not 0 < alpha < 1:
        raise ValidationError(f"alpha must be in (0, 1); got {alpha}")
    U = rng.random((n, n))
    U = np.triu(U, k=1)
    J = U + U.T
    eigvals = np.linalg.eigvalsh(J)
    spectral_radius = np.max(np.abs(eigvals))
    if spectral_radius < 1e-12:
        raise ValidationError("Coupling matrix has near-zero spectral radius")
    return J / spectral_radius * alpha


def evolution_operator_stable(H: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """U(tau) = exp(-i H tau) via eigendecomposition (avoids scipy.linalg.expm's
    Pade-approximant cost for repeated calls with the same H across an
    experiment sweep)."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    exp_eigs = np.exp(-1j * eigenvalues * tau)
    return eigenvectors @ (exp_eigs[:, None] * eigenvectors.conj().T)


def qrc_sequence_features(
    X: np.ndarray,
    U: np.ndarray,
    n1: int,
    n2: int,
    washout: int = 0,
    U_dag: np.ndarray | None = None,
    z_diags: np.ndarray | None = None,
) -> np.ndarray:
    """Persistent-state (true recurrent) QRC forward pass over a full sequence.

    Unlike the windowed `qrc_forward` (which resets the hidden state every
    call and cannot see beyond its window), this maintains rho_hidden across
    all T timesteps — the correct architecture for tasks needing memory
    longer than the window size. `X` is (T, n1) of already-encoded angles.
    Returns (T - washout, n1+n2) reservoir features.
    """
    T, d = X.shape
    if d != n1:
        raise ValidationError(f"X has {d} features but n1={n1}")
    n = n1 + n2
    dim_hidden = 1 << n2
    if U_dag is None:
        U_dag = U.conj().T
    if z_diags is None:
        z_diags = precompute_z_diagonals(n)

    rho_hidden = np.zeros((dim_hidden, dim_hidden), dtype=np.complex128)
    rho_hidden[0, 0] = 1.0

    features = []
    for t in range(T):
        rho_input = encode_input_state(X[t])
        rho = np.kron(rho_input, rho_hidden)
        rho = U @ rho @ U_dag
        rho_hidden = partial_trace_input(rho, n1, n2)
        if t >= washout:
            diag = np.real(np.diag(rho))
            features.append(z_diags @ diag)
    return np.array(features)


@dataclass
class QuantumReservoirConfig:
    n1: int = 5             # input qubits
    n2: int = 5             # hidden qubits
    alpha: float = 0.9      # coupling spectral radius (edge-of-chaos)
    tau: float = 1.0        # evolution time
    field_nu: float = 1.0   # transverse field strength
    seed: int = 42

    @property
    def n(self) -> int:
        return self.n1 + self.n2


class QuantumReservoir:
    """Reusable quantum reservoir: build once, extract features from many
    sequences/windows. Both the windowed and persistent-state forward passes
    are exposed since different applications need different memory depth.
    """

    def __init__(self, config: QuantumReservoirConfig | None = None) -> None:
        self.config = config or QuantumReservoirConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.J: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.U_dag: np.ndarray | None = None
        self.z_diags: np.ndarray | None = None
        self._built = False

    def build(self) -> "QuantumReservoir":
        cfg = self.config
        self.J = random_couplings(cfg.n, self._rng, cfg.alpha)
        self.H = build_ising_hamiltonian(cfg.n, self.J, cfg.field_nu)
        self.U = evolution_operator_stable(self.H, cfg.tau)
        self.U_dag = self.U.conj().T
        self.z_diags = precompute_z_diagonals(cfg.n)
        self._built = True
        return self

    def _require_built(self) -> None:
        if not self._built:
            raise ValidationError("Call .build() before extracting features")

    def windowed_features(self, windows: np.ndarray) -> np.ndarray:
        """windows: (n_windows_dim, window_len, n1) -> (n_windows_dim, n) features."""
        self._require_built()
        cfg = self.config
        out = np.empty((len(windows), cfg.n), dtype=np.float64)
        for i, w in enumerate(windows):
            out[i] = qrc_forward(w, self.U, cfg.n1, cfg.n2, self.U_dag, self.z_diags)
        return out

    def sequence_features(self, X: np.ndarray, washout: int = 0) -> np.ndarray:
        """X: (T, n1) encoded-angle sequence -> (T-washout, n) persistent-state features."""
        self._require_built()
        cfg = self.config
        return qrc_sequence_features(X, self.U, cfg.n1, cfg.n2, washout, self.U_dag, self.z_diags)


@ALGORITHM_REGISTRY.register("qrc")
class QRCAlgorithm:
    """Thin registry-facing wrapper so `core.cli` / `experiments/runners` can
    discover and instantiate QRC without importing this module directly."""

    config_cls = QuantumReservoirConfig
    reservoir_cls = QuantumReservoir


__all__ = [
    "QuantumReservoirConfig",
    "QuantumReservoir",
    "QRCAlgorithm",
    "random_couplings",
    "evolution_operator_stable",
    "qrc_sequence_features",
]
