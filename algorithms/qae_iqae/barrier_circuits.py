"""Factored barrier circuit: distribution loader + pure comparator (Woerner-Egger VaR
building block). A|0> loads sum sqrt(p_j)|j> on the index register, then flips the
ancilla for every in-barrier index. a = P(anc=1) = sum_{j in mask} p_j = grid-discretised
P(S_T <= barrier) (or, for `build_return_distribution_barrier`, P(PnL <= threshold)).

No payoff rescaling; the comparator is exact (theta in {0, pi}). Asymmetrica-original
(2026-07-23 addition to the upstream-derived circuit core) — see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import math
from typing import Dict

from algorithms_extended.qae_spinqit.quadrature import index_to_bits


def _flip_ancilla_on_index(qc, j: int, n: int, anc: int):
    """Flip the ancilla iff the index register equals |j> (X-mask + multi-controlled X)."""
    bits = [(j >> b) & 1 for b in range(n)]
    flipped = [q for q in range(n) if bits[q] == 0]
    for q in flipped:
        qc.x(q)
    if n == 1:
        qc.cx(0, anc)
    else:
        qc.mcx(list(range(n)), anc)
    for q in flipped:
        qc.x(q)


def build_A_barrier(probs, mask):
    """Factored A for the crash-barrier / loss-probability estimand (loader + pure comparator)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    m = len(probs)
    n = int(round(math.log2(m)))
    if 2 ** n != m:
        raise ValueError("len(probs) must be a power of 2")
    amps = [math.sqrt(p) if p > 0 else 0.0 for p in probs]
    nrm = math.sqrt(sum(a * a for a in amps)) or 1.0
    amps = [a / nrm for a in amps]

    qc = QuantumCircuit(n + 1, name="A_barrier")
    qc.append(StatePreparation(amps), list(range(n)))
    anc = n
    for j, mk in enumerate(mask):
        if mk:
            _flip_ancilla_on_index(qc, j, n, anc)
    return qc


def barrier_comparator_angle_table(mask) -> Dict:
    """Angle table of the barrier INDICATOR (theta=pi in-barrier, else 0) - for the
    affinity diagnostic only."""
    m = len(mask)
    n = int(round(math.log2(m)))
    return {index_to_bits(j, n): (math.pi if mk else 0.0) for j, mk in enumerate(mask)}


def comparator_affinity(mask, tol: float = 1e-9):
    """Least-squares affine fit of the barrier indicator theta(b) ~ c0 + sum c_i*b_i over
    the n index bits; return (max_residual, is_affine)."""
    m = len(mask)
    n = int(round(math.log2(m)))
    import numpy as np
    X = np.array([[1.0] + list(index_to_bits(j, n)) for j in range(m)])
    y = np.array([math.pi if mk else 0.0 for mk in mask])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = float(np.max(np.abs(X @ coeffs - y)))
    return resid, resid <= tol
