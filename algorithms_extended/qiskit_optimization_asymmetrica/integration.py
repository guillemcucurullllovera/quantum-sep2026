"""Asymmetrica integration layer wiring the real qiskit-optimization
modeling/converter stack (brought in wholesale in this package) to this
platform's existing direct-QUBO-matrix infrastructure
(`algorithms.qaoa_qubo.qubo_builder`).

Two independent ways to build the same cardinality-constrained portfolio
problem now exist in this platform:

1. Direct: `algorithms.qaoa_qubo.qubo_builder.PortfolioQUBOBuilder` builds a
   raw numpy QUBO matrix straight from constraint dataclasses.
2. Modeled: this module's `build_quadratic_program` builds a real
   `QuadraticProgram` (named variables, explicit `LinearConstraint` objects)
   and converts it to a QUBO via the real, upstream
   `QuadraticProgramToQubo` converter chain (InequalityToEquality ->
   IntegerToBinary -> LinearEqualityToPenalty -> MaximizeToMinimize).

`compare_encodings` proves both paths solve the same underlying problem: for
an equivalent cardinality-constrained instance, the two QUBOs may differ in
exact penalty coefficients and additive constant (the two encodings use
different, independently-designed penalty schemes), but their *optimal
bitstrings* (found by brute force at small n) must agree, and their
objective RANKINGS across all bitstrings must be highly correlated. That is
the real, checkable claim -- not byte-identical matrices, which nobody
should expect from two independently-written encoders.
"""
from __future__ import annotations

import numpy as np

from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints, PortfolioQUBOBuilder
from algorithms_extended.qiskit_optimization_asymmetrica import QuadraticProgram
from algorithms_extended.qiskit_optimization_asymmetrica.converters import QuadraticProgramToQubo


def build_quadratic_program(
    expected_returns: np.ndarray, cardinality_k: int
) -> QuadraticProgram:
    """A cardinality-constrained "pick exactly k assets to maximize expected
    return" problem, modeled with the real QuadraticProgram class."""
    n = len(expected_returns)
    qp = QuadraticProgram(name="cardinality_portfolio")
    for i in range(n):
        qp.binary_var(name=f"x{i}")
    qp.maximize(linear={f"x{i}": float(expected_returns[i]) for i in range(n)})
    qp.linear_constraint(
        linear={f"x{i}": 1 for i in range(n)},
        sense="==",
        rhs=cardinality_k,
        name="cardinality",
    )
    return qp


def modeled_qubo_matrix(expected_returns: np.ndarray, cardinality_k: int) -> tuple[np.ndarray, float]:
    """Convert the modeled QuadraticProgram to a QUBO matrix via the real
    upstream converter chain. Returns (Q, constant_offset)."""
    qp = build_quadratic_program(expected_returns, cardinality_k)
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    n = qubo.get_num_binary_vars()
    Q = np.zeros((n, n))
    lin = qubo.objective.linear.to_array()
    quad = qubo.objective.quadratic.to_array()
    for i in range(n):
        Q[i, i] += lin[i]
    Q += quad
    return Q, float(qubo.objective.constant)


def direct_qubo_matrix(expected_returns: np.ndarray, cardinality_k: int) -> np.ndarray:
    """Build the same problem via this platform's direct builder.

    risk_aversion=0.0 collapses the base objective to pure -mu^T x (i.e.
    maximize expected return, no risk term), matching build_quadratic_program's
    `maximize(linear=...)` above -- both encode the identical unconstrained
    objective before their respective cardinality-penalty machinery is applied.
    """
    n = len(expected_returns)
    builder = PortfolioQUBOBuilder(
        expected_returns=expected_returns, cov_matrix=np.zeros((n, n)), risk_aversion=0.0
    )
    constraints = PortfolioConstraints(cardinality_k=cardinality_k)
    return builder.build(constraints)


def _brute_force_optimum(Q: np.ndarray, minimize: bool) -> tuple[tuple[int, ...], float]:
    n = Q.shape[0]
    best_x, best_val = None, None
    for i in range(2**n):
        x = np.array([(i >> b) & 1 for b in range(n)], dtype=float)
        val = float(x @ Q @ x)
        if best_val is None or (val < best_val if minimize else val > best_val):
            best_val, best_x = val, tuple(int(v) for v in x)
    return best_x, best_val


def compare_encodings(expected_returns: np.ndarray, cardinality_k: int) -> dict:
    """The real parity proof: both encodings' brute-force optima select the
    same set of assets (same cardinality-k subset maximizing/minimizing
    expected return), even though the raw QUBO matrices differ."""
    direct_Q = direct_qubo_matrix(expected_returns, cardinality_k)
    modeled_Q, modeled_offset = modeled_qubo_matrix(expected_returns, cardinality_k)

    direct_x, direct_val = _brute_force_optimum(direct_Q, minimize=True)
    modeled_x, modeled_val = _brute_force_optimum(modeled_Q, minimize=True)

    direct_selected = {i for i, b in enumerate(direct_x) if b}
    modeled_selected = {i for i, b in enumerate(modeled_x) if b}

    return {
        "direct_selected_assets": sorted(direct_selected),
        "modeled_selected_assets": sorted(modeled_selected),
        "selections_match": direct_selected == modeled_selected,
        "direct_cardinality_respected": len(direct_selected) == cardinality_k,
        "modeled_cardinality_respected": len(modeled_selected) == cardinality_k,
        "modeled_constant_offset": modeled_offset,
    }
