"""QAOA mixer-Hamiltonian constructions, adapted from PennyLane's
``pennylane.qaoa.mixers`` (Apache-2.0, Xanadu Quantum Technologies Inc.) —
see ``upstream/qaoa/mixers.py`` for the original and ``THIRD_PARTY_NOTICES.md``
for what changed.

Reimplemented to build Qiskit ``SparsePauliOp`` objects directly (no
PennyLane operator/tape machinery, no rustworkx dependency -- uses plain
edge lists) so these mixers are directly usable by
``algorithms.qaoa_qubo.solvers.QiskitQAOASolver`` as alternative mixer
choices alongside the platform's existing standard/correlation-XY/SK
mixers (``algorithms.qaoa_qubo.gen2.mixers``).

``xy_mixer`` is a principled, general graph-based cardinality-preserving
mixer -- it only moves amplitude between computational basis states with
equal Hamming weight, which is exactly the property QAOA solver
approximated ad hoc via SA-polish repair. Using this mixer directly can
avoid needing that repair step for cardinality-constrained problems.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def _pauli_string(n_qubits: int, indices_paulis: dict[int, str]) -> str:
    chars = ["I"] * n_qubits
    for idx, p in indices_paulis.items():
        chars[idx] = p
    return "".join(reversed(chars))  # Qiskit little-endian convention


def x_mixer(n_qubits: int) -> SparsePauliOp:
    """H_M = sum_i X_i -- the standard QAOA mixer (Farhi et al. 2014, arXiv:1411.4028)."""
    terms = [(_pauli_string(n_qubits, {i: "X"}), 1.0) for i in range(n_qubits)]
    return SparsePauliOp.from_list(terms)


def xy_mixer(n_qubits: int, edges: list[tuple[int, int]]) -> SparsePauliOp:
    """H_M = 1/2 sum_{(i,j) in E} X_i X_j + Y_i Y_j -- cardinality-preserving
    mixer (Hadfield et al. 2019, doi:10.3390/a12020034). Only transfers
    amplitude between equal-Hamming-weight basis states, so a QAOA circuit
    initialized in a feasible-cardinality state stays feasible throughout
    (given a cardinality-respecting initial state and cost Hamiltonian).
    """
    terms = []
    for i, j in edges:
        terms.append((_pauli_string(n_qubits, {i: "X", j: "X"}), 0.5))
        terms.append((_pauli_string(n_qubits, {i: "Y", j: "Y"}), 0.5))
    if not terms:
        return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    return SparsePauliOp.from_list(terms).simplify()


def complete_graph_edges(n_qubits: int) -> list[tuple[int, int]]:
    """All-to-all connectivity for xy_mixer -- the standard choice when there's
    no natural sparse graph structure (e.g. an unconstrained asset universe)."""
    return list(combinations(range(n_qubits), 2))


def ring_graph_edges(n_qubits: int) -> list[tuple[int, int]]:
    """Ring connectivity -- O(n) terms instead of xy_mixer's O(n^2) for the
    complete graph, trading mixer expressivity for shallower circuits."""
    return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]


def bit_flip_mixer(n_qubits: int, edges: list[tuple[int, int]], b: int = 0) -> SparsePauliOp:
    """H_M = sum_v (1/2^d(v)) X_v * prod_{w in N(v)} (I + (-1)^b Z_w)
    (Hadfield et al. 2019). Only flips qubit v when all its graph-neighbors
    are in a specified state -- useful for structural constraints beyond
    simple cardinality (e.g. "don't add this asset unless its sector anchor
    is already selected").
    """
    if b not in (0, 1):
        raise ValueError("b must be 0 or 1")
    neighbors: dict[int, list[int]] = {v: [] for v in range(n_qubits)}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    total = SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    sign = 1.0 if b == 0 else -1.0
    for v in range(n_qubits):
        nv = neighbors[v]
        deg = len(nv)
        prefactor = 1.0 / (2**deg) if deg else 1.0
        # expand prod_{w in N(v)} (I + sign*Z_w) into its 2^deg Pauli terms
        term_ops = SparsePauliOp.from_list([("I" * n_qubits, 1.0)])
        for w in nv:
            identity_part = SparsePauliOp.from_list([("I" * n_qubits, 1.0)])
            z_part = SparsePauliOp.from_list([(_pauli_string(n_qubits, {w: "Z"}), sign)])
            term_ops = term_ops.compose(identity_part + z_part)
        x_v = SparsePauliOp.from_list([(_pauli_string(n_qubits, {v: "X"}), prefactor)])
        total = total + x_v.compose(term_ops)
    return total.simplify()
