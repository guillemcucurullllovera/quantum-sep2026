"""SolverChoice.XY_MIXER_QAOA: a genuinely different QAOA circuit
architecture from every other generation in this platform -- a real Qiskit
circuit (Dicke-state cardinality-k initial state + cardinality-preserving
XY-mixer evolution layers, from
`algorithms_extended.pennylane_asymmetrica.finance.xy_mixer_circuit`)
executed on Aer's statevector simulator, not a NumPy/diagonal-only
approximation. Every candidate bitstring this circuit's mixer can ever reach
already satisfies the cardinality constraint exactly (the mixer only moves
amplitude between equal-Hamming-weight basis states), so this is the one
solver generation where cardinality is enforced by construction rather than
by penalty or repair.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from algorithms.qaoa_qubo.qubo_builder import PortfolioQUBOBuilder
from algorithms_extended.pennylane_asymmetrica.finance.mixer_hamiltonians import complete_graph_edges
from algorithms_extended.pennylane_asymmetrica.finance.xy_mixer_circuit import (
    build_cardinality_preserving_qaoa_circuit,
)
from algorithms_extended.qokit_finance.fast_simulator import cost_diagonal_from_qubo
from algorithms_extended.qokit_finance.sk_warmstart import sk_gamma_beta
from applications.portfolio.domain import PortfolioSpec
from applications.portfolio.exceptions import SolverFailureError
from core.errors import ValidationError


def _expectation_via_aer(qc, cost_diagonal: np.ndarray) -> tuple[float, np.ndarray]:
    backend = AerSimulator(method="statevector")
    qc2 = qc.copy()
    qc2.save_statevector()
    result = backend.run(qc2).result()
    sv = np.asarray(result.get_statevector(qc2))
    probs = np.abs(sv) ** 2
    return float(probs @ cost_diagonal), probs


def solve_xy_mixer_qaoa(
    spec: PortfolioSpec,
    seed: int = 42,
    *,
    p: int = 2,
    maxiter: int = 60,
    edges: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, float, float, dict]:
    """Returns (bitstring, qubo_cost, wall_time_s, extra_diagnostics)."""
    spec.validate_shape()
    k = spec.cardinality_k()
    if k is None:
        raise ValidationError("XY_MIXER_QAOA requires a cardinality constraint")

    n = spec.n_assets()
    if n > 14:
        raise ValidationError(
            f"XY_MIXER_QAOA: n={n} assets -- the Dicke-state initial-state circuit is an "
            "exact 2**n statevector `initialize`, impractical beyond ~14 qubits."
        )

    from applications.portfolio.multi_solver import _to_portfolio_constraints

    constraints = _to_portfolio_constraints(spec)
    Q = PortfolioQUBOBuilder(spec.expected_returns(), spec.cov(), risk_aversion=spec.risk_aversion).build(constraints)
    cost_diagonal = cost_diagonal_from_qubo(Q)

    mixer_edges = edges if edges is not None else complete_graph_edges(n)
    gamma0, beta0 = sk_gamma_beta(p)  # same cross-generation SK warm start QOKIT_FAST uses

    t0 = time.perf_counter()
    n_evals = [0]

    def objective(params: np.ndarray) -> float:
        n_evals[0] += 1
        gamma, beta = params[:p], params[p:]
        qc = build_cardinality_preserving_qaoa_circuit(n, k, mixer_edges, cost_diagonal, gamma, beta)
        exp, _probs = _expectation_via_aer(qc, cost_diagonal)
        return exp

    try:
        x0 = np.concatenate([gamma0[:p], beta0[:p]])
        result = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
    except Exception as exc:
        raise SolverFailureError(f"XY_MIXER_QAOA optimization failed: {exc}") from exc

    gamma, beta = result.x[:p], result.x[p:]
    qc = build_cardinality_preserving_qaoa_circuit(n, k, mixer_edges, cost_diagonal, gamma, beta)
    _exp, probs = _expectation_via_aer(qc, cost_diagonal)
    dt = time.perf_counter() - t0

    best_idx = int(np.argmax(probs))
    bits = [(best_idx >> b) & 1 for b in range(n)]
    x = np.array(bits, dtype=float)
    cost = float(x @ Q @ x)

    extra = {
        "n_evaluations": n_evals[0],
        "best_state_probability": float(probs[best_idx]),
        "gamma": gamma.tolist(),
        "beta": beta.tolist(),
        "mixer_edges": len(mixer_edges),
    }
    return x, cost, dt, extra
