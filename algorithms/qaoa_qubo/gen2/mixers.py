"""
QAOA Mixer Implementations: Ring-XY and Correlation-Based Complete-XY

Two mixer Hamiltonians for QAOA applied to financial portfolio QUBO:

1. ring_xy_pairs:  H_B = Σ_{i} (X_i X_{i+1} + Y_i Y_{i+1})   ← Phase0 (wrong)
2. correlation_xy_pairs: H_B = Σ_{(i,j): |corr|>thresh} (X_i X_j + Y_i Y_j)  ← v2 fix

Why ring-XY fails for portfolio QUBO:
  The ring topology (i→i+1 mod n) imposes a 1-D spatial structure that has
  nothing to do with the asset correlation structure. The mixer's ground state
  (equal superposition over states with equal Hamming weight) is correct for
  the QAOA partition function, but the energy landscape is not aligned with
  the QUBO correlations — the optimizer has to "fight" the mixer structure.

Why correlation-XY works:
  The XY mixer moves probability amplitude between states that differ by
  swapping two bits. If assets i and j are correlated, swapping their
  inclusion changes the QUBO cost in a predictable way. Building the mixer
  graph from asset correlations aligns the mixing with the problem structure.
  This is the key insight from Hadfield et al. (2019) mixer design principles.

Reference: Hadfield et al. (2019) "From the Quantum Approximate Optimization
  Algorithm to a Quantum Alternating Operator Ansatz", Algorithms.
"""

from __future__ import annotations

import numpy as np


def ring_xy_pairs(n: int) -> list[tuple[int, int]]:
    """
    Ring-XY mixer pairs: each qubit coupled to its nearest neighbor.

    Topology: 0—1—2—...—(n-1)—0 (ring).

    Args:
        n: Number of qubits.

    Returns:
        pairs: List of (i, j) qubit index pairs, length n.
    """
    return [(i, (i + 1) % n) for i in range(n)]


def correlation_xy_pairs(
    corr_matrix: np.ndarray,
    threshold: float = 0.3,
) -> list[tuple[int, int]]:
    """
    Correlation-based complete-XY mixer pairs.

    Includes edge (i,j) if |corr(i,j)| > threshold. Edges are undirected:
    (i,j) appears once with i < j.

    If no pairs exceed the threshold, falls back to ring topology with a
    warning — this prevents an empty mixer (no QAOA evolution at all).

    Args:
        corr_matrix: Asset correlation matrix, shape (n, n). Diagonal ignored.
        threshold: Minimum |correlation| to include an edge (default 0.3).

    Returns:
        pairs: List of (i, j) pairs with |corr(i,j)| > threshold.
    """
    n = corr_matrix.shape[0]
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if abs(corr_matrix[i, j]) > threshold
    ]
    if not pairs:
        # Fallback: ring topology ensures mixer is non-trivial
        pairs = ring_xy_pairs(n)
    return pairs


def complete_xy_pairs(n: int) -> list[tuple[int, int]]:
    """
    Complete graph XY mixer: all pairs coupled.

    Used when no correlation structure is available.

    Args:
        n: Number of qubits.

    Returns:
        pairs: All (i, j) pairs with i < j.
    """
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def apply_xy_pair(
    state: np.ndarray,
    qi: int,
    qj: int,
    beta: float,
) -> np.ndarray:
    """
    Apply exp(-i β (X_qi X_qj + Y_qi Y_qj)) to the statevector.

    The XY interaction couples the {|01⟩, |10⟩} subspace of qubits (qi, qj):
      exp(-i β H_XY) on |01⟩ = cos(2β)|01⟩ - i sin(2β)|10⟩
      exp(-i β H_XY) on |10⟩ = -i sin(2β)|01⟩ + cos(2β)|10⟩

    |00⟩ and |11⟩ are unaffected (eigenvalue 0 of H_XY = X_i X_j + Y_i Y_j).

    Args:
        state: Complex statevector, shape (2**n,).
        qi, qj: Qubit indices (0-indexed). Must be distinct.
        beta: Mixer angle.

    Returns:
        new_state: Updated statevector after XY rotation.
    """
    c = np.cos(2 * beta)
    s = np.sin(2 * beta)
    new_state = state.copy()
    mask_qi_0 = 1 << qi
    mask_qj_1 = 1 << qj

    # Iterate over all basis states where qi=0, qj=1
    for k in range(len(state)):
        bit_i = (k >> qi) & 1
        bit_j = (k >> qj) & 1
        if bit_i == 0 and bit_j == 1:
            k_pair = k ^ mask_qi_0 ^ mask_qj_1   # flip both bits → qi=1, qj=0
            a = state[k]
            b = state[k_pair]
            new_state[k] = c * a - 1j * s * b
            new_state[k_pair] = -1j * s * a + c * b

    return new_state


def apply_mixer(
    state: np.ndarray,
    pairs: list[tuple[int, int]],
    beta: float,
) -> np.ndarray:
    """
    Apply the full XY mixer exp(-i β H_B) by sequentially applying each pair.

    Note: individual pair operators don't commute in general, so the sequential
    product is a first-order Trotter approximation. For small β this is accurate.

    Args:
        state: Complex statevector, shape (2**n,).
        pairs: List of (qi, qj) pairs defining the mixer graph.
        beta: Mixer angle.

    Returns:
        state: Updated statevector.
    """
    for qi, qj in pairs:
        state = apply_xy_pair(state, qi, qj, beta)
    return state


def initial_dicke_state(n: int, k: int) -> np.ndarray:
    """
    Prepare the Dicke state |D_k^n⟩ = equal superposition over all n-qubit
    states with Hamming weight k.

    This is the correct initial state for QAOA with XY mixers — the XY mixer
    preserves Hamming weight, so the state stays within the k-excitation sector.

    For portfolio optimization with target cardinality k (= number of assets
    to hold), use the Dicke state with that k.

    Args:
        n: Number of qubits.
        k: Target Hamming weight (number of |1⟩s).

    Returns:
        state: Normalized Dicke state, shape (2**n,).
    """
    from math import comb
    N = 2**n
    state = np.zeros(N, dtype=complex)
    count = comb(n, k)
    norm = 1.0 / np.sqrt(count) if count > 0 else 0.0
    for idx in range(N):
        if bin(idx).count('1') == k:
            state[idx] = norm
    return state


def plus_state(n: int) -> np.ndarray:
    """
    Prepare |+⟩^n = H^⊗n |0⟩^n — uniform superposition over all basis states.

    Used as initial state when no cardinality constraint is imposed.

    Args:
        n: Number of qubits.

    Returns:
        state: Normalized |+⟩^n, shape (2**n,).
    """
    N = 2**n
    return np.ones(N, dtype=complex) / np.sqrt(N)
