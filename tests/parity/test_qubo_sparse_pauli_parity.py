import numpy as np

from algorithms_extended.qubo_constraints_qiskit_optimization import qubo_to_sparse_pauli_op
from algorithms_extended.qubo_constraints_qiskit_optimization.converter import (
    sparse_pauli_op_expectation_matches_qubo,
)


def _random_symmetric_qubo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return (A + A.T) / 2


def test_sparse_pauli_op_matches_direct_qubo_evaluation_on_all_bitstrings():
    n = 4
    Q = _random_symmetric_qubo(n, seed=7)
    for bits in range(2 ** n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        assert sparse_pauli_op_expectation_matches_qubo(Q, x), f"mismatch at bits={bits:0{n}b}"


def test_operator_is_hermitian_and_nontrivial():
    Q = _random_symmetric_qubo(5, seed=1)
    op, offset = qubo_to_sparse_pauli_op(Q)
    assert len(op) > 0
    assert np.allclose(op.to_matrix(), op.to_matrix().conj().T)
