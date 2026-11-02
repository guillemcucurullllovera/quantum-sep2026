"""Fast diagonal-cost QAOA statevector simulation.

Adapted from QOKit's "FUR" (Fast Unitary Representation) technique
(qokit/fur/python/{fur.py,qaoa_fur.py}, Apache-2.0, JP Morgan Chase & Co,
arXiv:2309.04841). This is a from-scratch reimplementation of the same
algorithmic idea, not a copy of QOKit's source, integrated with this
platform's own `core.schemas`/`core.experiment_registry` rather than
QOKit's own utilities.

The idea QOKit's simulator exploits, and that this module reproduces: for
QAOA specifically, you never need a general-purpose circuit simulator.
The cost-Hamiltonian layer is diagonal (a plain elementwise phase
multiply on the 2**n-dimensional statevector), and the X-mixer layer
acts on one qubit at a time as a 2x2 rotation applied to index pairs
that differ only in bit q — both are directly vectorizable in numpy
without ever constructing gate matrices or a circuit object. For QAOA
depth p and n qubits this is asymptotically the same cost as a real
simulator but with a much smaller constant factor than routing through
Qiskit's general `transpile()` + `Statevector` machinery, because there
is no gate-decomposition or circuit-object overhead at all.
"""
from __future__ import annotations

import math

import numpy as np


def apply_x_rotation(sv: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply exp(-i*theta*X) on `qubit`, in place, to statevector `sv`.

    Index-pair bit-masking approach (QOKit's `furx`): for each pair of
    basis-state indices that differ only in bit `qubit`, apply the 2x2
    rotation directly to that pair of amplitudes. No gate matrix, no
    Kronecker product, no circuit.
    """
    n_states = len(sv)
    n_groups = n_states // 2
    mask_lo = (1 << qubit) - 1
    mask_hi = mask_lo ^ ((n_states - 1) >> 1)

    cos_t = math.cos(theta)
    sin_t = -1j * math.sin(theta)

    idx = np.arange(n_groups)
    ia = (idx & mask_lo) | ((idx & mask_hi) << 1)
    ib = ia | (1 << qubit)

    a, b = sv[ia], sv[ib]
    sv[ia] = cos_t * a + sin_t * b
    sv[ib] = sin_t * a + cos_t * b
    return sv


def apply_x_mixer_all(sv: np.ndarray, beta: float, n_qubits: int) -> np.ndarray:
    """Apply the standard QAOA X-mixer exp(-i*beta*sum_j X_j) to every qubit."""
    for q in range(n_qubits):
        apply_x_rotation(sv, beta, q, n_qubits)
    return sv


def cost_diagonal_from_qubo(Q: np.ndarray) -> np.ndarray:
    """Enumerate the diagonal cost Hamiltonian h_z = x^T Q x for every
    z in {0,1}^n, given a QUBO matrix Q (as produced by
    `algorithms.qaoa_qubo.qubo_builder.PortfolioQUBOBuilder.build`).

    Practical only for n small enough that 2**n fits in memory (n <~ 22
    on a modest machine) -- the same limitation QOKit's own CPU simulator
    has; this is not a scalability claim, it's a per-call speed claim at
    a given n relative to circuit-based simulation.
    """
    n = Q.shape[0]
    n_states = 1 << n
    diag = np.zeros(n_states, dtype=float)
    # vectorized bit-unpacking: bits[:, j] = bit j of each basis-state index
    idx = np.arange(n_states)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    diag = np.einsum("si,ij,sj->s", bits, Q, bits)
    return diag


def run_fast_qaoa(
    cost_diag: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Run a full p-layer QAOA circuit (X-mixer) via the diagonal/FUR
    technique and return the final statevector. Initial state is the
    uniform superposition, matching QOKit's `default_sv0` convention.
    """
    n_states = 1 << n_qubits
    sv = np.full(n_states, 1.0 / math.sqrt(n_states), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        sv *= np.exp(-0.5j * gamma * cost_diag)
        apply_x_mixer_all(sv, beta, n_qubits)
    return sv


def expected_cost(sv: np.ndarray, cost_diag: np.ndarray) -> float:
    probs = np.abs(sv) ** 2
    return float(np.dot(probs, cost_diag))
