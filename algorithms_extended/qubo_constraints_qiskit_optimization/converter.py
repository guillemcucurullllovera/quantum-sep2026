"""QUBO -> Qiskit SparsePauliOp converter.

Materially adapted from qiskit-optimization's `to_ising` translation
convention (variable ordering, x_i = (1 - Z_i)/2 substitution, weight/4
quadratic scaling) — see THIRD_PARTY_NOTICES.md in this directory for exactly what
was adapted vs. written fresh. This is NOT a copy of
`qiskit_optimization/translators/ising.py`: that module converts a
`QuadraticProgram` object (with its own constraint/variable-type machinery)
to a `SparsePauliOp`; this module instead takes the raw QUBO matrices this
platform's `algorithms.qaoa_qubo.qubo_builder` already produces and converts
them directly, so our constraint-suite QUBOs (cardinality, turnover,
transaction cost, sector limits, liquidity, long/short) can be handed
straight to any Qiskit primitive (Estimator, QAOA, VQE) without going
through `QuadraticProgram` at all.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp


def qubo_to_sparse_pauli_op(Q: np.ndarray) -> tuple[SparsePauliOp, float]:
    """Convert a QUBO matrix Q (cost = x^T Q x, x in {0,1}^n) into a Qiskit
    `SparsePauliOp` Ising Hamiltonian plus a constant offset, following the
    same x_i = (1 - Z_i)/2 substitution qiskit-optimization uses, but
    operating directly on a numpy QUBO matrix rather than a QuadraticProgram.

    Returns:
        (operator, offset) such that <psi|operator|psi> + offset equals the
        QUBO cost for a computational-basis state |psi> encoding a bitstring
        via the same little-endian convention Qiskit uses elsewhere in this
        platform's solvers.py.
    """
    n = Q.shape[0]
    Qs = (Q + Q.T) / 2  # symmetrize, matching qiskit-optimization's convention
    zero = np.zeros(n, dtype=bool)

    paulis: list[Pauli] = []
    coeffs: list[float] = []
    offset = 0.0

    # x_i = (1 - Z_i)/2. Diagonal term Q_ii*x_i contributes a constant
    # Q_ii/2 to offset and -Q_ii/2 to the Z_i coefficient. Each off-diagonal
    # pair contributes via x_i*x_j = (1 - Z_i - Z_j + Z_iZ_j)/4: a constant
    # Qs_ij/2, a Z_iZ_j term of Qs_ij/2, AND a -Qs_ij/2 correction to BOTH
    # the Z_i and Z_j linear coefficients — that per-pair linear correction
    # is the term a naive port easily drops (it was dropped in an earlier
    # draft of this function; caught by test_qubo_sparse_pauli_parity.py).
    linear = -Qs.diagonal().copy() / 2  # start from the diagonal's own contribution
    offset += float(Qs.diagonal().sum()) / 2

    for i in range(n):
        for j in range(i + 1, n):
            coeff = Qs[i, j]
            if abs(coeff) < 1e-14:
                continue
            weight = coeff  # x_i*x_j uses 2*Qs_ij (Q_ij+Q_ji already folded into Qs), quarter-scaled below
            offset += weight / 2
            linear[i] -= weight / 2
            linear[j] -= weight / 2
            z = zero.copy()
            z[i] = True
            z[j] = True
            paulis.append(Pauli((z, zero)))
            coeffs.append(weight / 2)

    for i in range(n):
        if abs(linear[i]) > 1e-14:
            z = zero.copy()
            z[i] = True
            paulis.append(Pauli((z, zero)))
            coeffs.append(linear[i])

    if not paulis:
        # degenerate all-zero QUBO: SparsePauliOp needs at least one term
        paulis.append(Pauli((zero, zero)))
        coeffs.append(0.0)

    op = SparsePauliOp(paulis, coeffs).simplify()
    return op, offset


def sparse_pauli_op_expectation_matches_qubo(Q: np.ndarray, x: np.ndarray, atol: float = 1e-8) -> bool:
    """Validation helper: check <z|H|z> + offset == x^T Q x for a specific
    computational basis bitstring x (used in tests as the parity check
    against algorithms.qaoa_qubo.qubo_builder's direct QUBO evaluation)."""
    from qiskit.quantum_info import Statevector

    n = Q.shape[0]
    op, offset = qubo_to_sparse_pauli_op(Q)
    # build the computational basis statevector for bitstring x (little-endian)
    idx = sum(int(x[i]) << i for i in range(n))
    sv = Statevector.from_int(idx, dims=2**n)
    expectation = sv.expectation_value(op).real + offset
    direct = float(np.asarray(x) @ Q @ np.asarray(x))
    return abs(expectation - direct) < atol
