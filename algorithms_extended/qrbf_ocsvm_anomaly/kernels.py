"""
Quantum and classical kernels for the qrbf-OCSVM replication.

We implement the three hardware-efficient feature maps from Pranjic et al.
Appendix A — qrbf-CX, qrbf-ECR, qrbf-RXX — at n_qubits=10, with two
single-qubit rotations per qubit (R_X then R_Z) and nearest-neighbour
two-qubit entanglers on a linear topology.

The projected quantum kernel uses 1-qubit reduced density matrices (1Q-RDMs):

    K_γ(x_i, x_j) = exp(-γ Σ_k ||ρ_k(x_i) - ρ_k(x_j)||²_F)

For the simulator-limit replication (paper Section III.A), we compute the
1Q-RDMs analytically from the exact statevector via partial trace, rather
than simulating the 200-snapshot tomography protocol. This isolates the
kernel/OCSVM behaviour from finite-sampling noise and matches the "simulator"
column of the paper's Table 1.
"""

from __future__ import annotations

import math
from typing import Callable, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

N_QUBITS = 10


# ── feature scaling ─────────────────────────────────────────────────

def scale_to_pi(X: np.ndarray) -> np.ndarray:
    """Min-max-scale each column to [-π, π] (paper-style rotation encoding)."""
    X = np.asarray(X, dtype=float).copy()
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (2.0 * (X - lo) / rng - 1.0) * math.pi


# ── feature maps ────────────────────────────────────────────────────

def feature_map_cx(x: np.ndarray) -> QuantumCircuit:
    """qrbf-CX: R_X(x_k) then R_Z(x_{10+k}) per qubit, nearest-neighbour CX ring."""
    qc = QuantumCircuit(N_QUBITS)
    for k in range(N_QUBITS):
        qc.rx(float(x[k]), k)
        qc.rz(float(x[N_QUBITS + k]), k)
    for k in range(N_QUBITS - 1):
        qc.cx(k, k + 1)
    return qc


def feature_map_ecr(x: np.ndarray) -> QuantumCircuit:
    """qrbf-ECR: R_X then R_Z per qubit, nearest-neighbour ECR entanglers."""
    qc = QuantumCircuit(N_QUBITS)
    for k in range(N_QUBITS):
        qc.rx(float(x[k]), k)
        qc.rz(float(x[N_QUBITS + k]), k)
    for k in range(N_QUBITS - 1):
        qc.ecr(k, k + 1)
    return qc


def feature_map_rxx(x: np.ndarray) -> QuantumCircuit:
    """qrbf-RXX: R_X then R_Z per qubit, nearest-neighbour R_XX(π/2) (maximally entangling)."""
    qc = QuantumCircuit(N_QUBITS)
    for k in range(N_QUBITS):
        qc.rx(float(x[k]), k)
        qc.rz(float(x[N_QUBITS + k]), k)
    for k in range(N_QUBITS - 1):
        qc.rxx(math.pi / 2.0, k, k + 1)
    return qc


FEATURE_MAPS: dict[str, Callable[[np.ndarray], QuantumCircuit]] = {
    "CX":  feature_map_cx,
    "ECR": feature_map_ecr,
    "RXX": feature_map_rxx,
}


# ── 1Q-RDM extraction ───────────────────────────────────────────────

def state_to_1q_rdms(statevec: np.ndarray) -> np.ndarray:
    """
    Compute all 10 single-qubit reduced density matrices from a 1024-dim
    state vector via partial trace.

    Returns array of shape (n_qubits, 2, 2) complex.
    """
    n = N_QUBITS
    dim = 1 << n
    assert statevec.shape == (dim,), f"expected {dim}-d statevector, got {statevec.shape}"
    psi = statevec.astype(np.complex128)
    # reshape into a tensor with one axis per qubit (little-endian: axis 0 is q0)
    tensor = psi.reshape([2] * n)

    rdms = np.empty((n, 2, 2), dtype=np.complex128)
    for k in range(n):
        # Move qubit k to the LAST axis; flatten the other (n-1) axes into a single
        # column dimension; ρ_k = T T† over the remaining axes.
        t = np.moveaxis(tensor, k, -1)            # shape (2,2,...,2, 2)
        mat = t.reshape(-1, 2)                    # (2^(n-1), 2)
        rdm = mat.conj().T @ mat                  # (2, 2)
        rdms[k] = rdm
    return rdms


def compute_rdms_for_dataset(
    X: np.ndarray,
    feature_map: Callable[[np.ndarray], QuantumCircuit],
) -> np.ndarray:
    """Compute (N, n_qubits, 2, 2) tensor of 1Q-RDMs for an entire dataset."""
    n_samples = X.shape[0]
    rdms = np.empty((n_samples, N_QUBITS, 2, 2), dtype=np.complex128)
    for i, x in enumerate(X):
        sv = Statevector(feature_map(x)).data
        rdms[i] = state_to_1q_rdms(sv)
    return rdms


# ── projected quantum kernel ────────────────────────────────────────

def qrbf_kernel_matrix(
    rdms_A: np.ndarray,
    rdms_B: np.ndarray,
    gamma: float = 0.1,
) -> np.ndarray:
    """
    Build the qrbf kernel matrix between two sets of 1Q-RDMs.

    K[i, j] = exp(-γ · Σ_k ||ρ_k(x_i^A) - ρ_k(x_j^B)||²_F)
    """
    na = rdms_A.shape[0]
    nb = rdms_B.shape[0]
    # Pairwise sum of squared Frobenius norms of RDM differences.
    # Δ has shape (na, nb, n_qubits, 2, 2)
    diff = rdms_A[:, None, :, :, :] - rdms_B[None, :, :, :, :]
    # Frobenius norm squared per qubit, summed across qubits.
    sq = (diff.conj() * diff).real.sum(axis=(2, 3, 4))   # (na, nb)
    return np.exp(-gamma * sq)


def rbf_kernel_matrix(
    X_A: np.ndarray, X_B: np.ndarray, gamma: float = 0.1
) -> np.ndarray:
    """Classical RBF kernel: K[i,j] = exp(-γ ||x_i - x_j||²)."""
    diff = X_A[:, None, :] - X_B[None, :, :]
    sq = (diff * diff).sum(axis=2)
    return np.exp(-gamma * sq)
