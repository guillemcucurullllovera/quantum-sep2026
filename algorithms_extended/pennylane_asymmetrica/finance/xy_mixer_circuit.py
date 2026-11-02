"""Closes a documented gap in ``algorithms.qaoa_qubo.solvers``: its own
docstring states the Qiskit QAOA path implements only the standard RX
mixer, not the cardinality-preserving XY-mixer the NumPy engines use,
because building the hardware circuit (Dicke-state init + XY-mixer
evolution) wasn't done. This module does it, using
``mixer_hamiltonians.xy_mixer`` (adapted from PennyLane) for the
Hamiltonian and Qiskit's own state-vector ``initialize`` for the Dicke
state (exact but not scalable past ~15-18 qubits -- documented below,
not hidden).
"""
from __future__ import annotations

from itertools import combinations
from math import comb, sqrt

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, PauliEvolutionGate

from .mixer_hamiltonians import xy_mixer


def dicke_state_circuit(n_qubits: int, hamming_weight: int) -> QuantumCircuit:
    """Equal superposition over every n-qubit computational basis state with
    exactly ``hamming_weight`` ones -- the natural cardinality-K initial
    state to pair with an XY mixer (mixing never leaves this subspace).

    Uses exact statevector initialization: correct for any n, but memory
    cost is O(2^n) like any Aer statevector simulation -- fine for the
    problem sizes this platform's tests run at (n <= ~16), not intended
    for larger n without a dedicated Dicke-state-preparation circuit
    (a known, more scalable construction exists in the literature but was
    not ported here -- documented gap, not a silent limitation).
    """
    if not (0 <= hamming_weight <= n_qubits):
        raise ValueError("hamming_weight must be between 0 and n_qubits")
    dim = 2**n_qubits
    amp = 1.0 / sqrt(comb(n_qubits, hamming_weight))
    state = np.zeros(dim, dtype=complex)
    for ones in combinations(range(n_qubits), hamming_weight):
        idx = sum(1 << q for q in ones)
        state[idx] = amp
    qc = QuantumCircuit(n_qubits)
    qc.initialize(state, range(n_qubits))
    return qc


def xy_mixer_layer(n_qubits: int, edges: list[tuple[int, int]], beta: float) -> QuantumCircuit:
    """One XY-mixer evolution layer, exp(-i*beta*H_mixer), as an actual gate
    sequence (via Qiskit's PauliEvolutionGate + Trotterization), not just
    the abstract Hamiltonian."""
    H = xy_mixer(n_qubits, edges)
    qc = QuantumCircuit(n_qubits)
    qc.append(PauliEvolutionGate(H, time=beta), range(n_qubits))
    return qc.decompose()


def build_cardinality_preserving_qaoa_circuit(
    n_qubits: int,
    hamming_weight: int,
    edges: list[tuple[int, int]],
    cost_diagonal: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> QuantumCircuit:
    """Full cardinality-preserving QAOA circuit: Dicke-state init, then
    alternating {diagonal cost phase, XY-mixer evolution} for p = len(gamma)
    rounds. ``cost_diagonal`` is a length-2^n array of the cost Hamiltonian's
    diagonal (same convention as ``algorithms.qaoa_qubo.solvers._cost_diagonal_from_ising``).

    This is the circuit ``algorithms/qaoa_qubo/solvers.py`` documented
    as not yet built for the Qiskit execution path.
    """
    p = len(gamma)
    if len(beta) != p:
        raise ValueError("gamma and beta must have equal length")

    qc = dicke_state_circuit(n_qubits, hamming_weight)
    for layer in range(p):
        # diagonal cost phase: exp(-i*gamma*diag(H_cost)) applied via a
        # diagonal gate (phase per computational basis state)
        phases = np.exp(-1j * gamma[layer] * cost_diagonal)
        qc.append(DiagonalGate(list(phases)), range(n_qubits))
        qc.compose(xy_mixer_layer(n_qubits, edges, beta[layer]), inplace=True)
    return qc
