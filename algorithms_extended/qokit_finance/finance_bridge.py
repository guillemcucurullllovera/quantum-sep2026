"""Wires the QOKit-adapted fast simulator + parameter transfer to this
platform's own portfolio-QUBO construction (`algorithms.qaoa_qubo.qubo_builder`). Does not rebuild constraint logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints, PortfolioQUBOBuilder
from algorithms_extended.qokit_finance.fast_simulator import cost_diagonal_from_qubo, expected_cost, run_fast_qaoa
from algorithms_extended.qokit_finance.sk_warmstart import sk_gamma_beta
from core.registry import ALGORITHM_REGISTRY
from core.schemas import RunMetadata


@dataclass
class FastQAOAResult:
    expected_cost: float
    probabilities: np.ndarray
    cost_diag: np.ndarray
    gamma: np.ndarray
    beta: np.ndarray
    metadata: RunMetadata


def solve_portfolio_fast(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    constraints: PortfolioConstraints,
    p: int = 4,
    risk_aversion: float = 0.5,
    use_sk_warmstart: bool = True,
    optimize: bool = True,
    maxiter: int = 100,
) -> FastQAOAResult:
    """Build a portfolio QUBO with the platform's existing constraint
    suite, then run it through the QOKit-adapted fast diagonal simulator
    instead of a Qiskit-circuit-based one. `p` must keep n_qubits small
    enough for `cost_diagonal_from_qubo`'s 2**n enumeration to be
    practical (n <~ 18-20).

    `use_sk_warmstart` controls the STARTING point for classical
    optimization (SK-model-transferred angles vs. random), not the final
    answer -- SK angles are a warm start, not a fixed schedule, since they
    were optimized for a different (SK, not this specific QUBO) cost
    landscape. When `optimize=False`, the warm-start angles are returned
    as-is without refinement (useful for benchmarking raw simulation
    speed in isolation, see scaling_benchmark.py).
    """
    builder = PortfolioQUBOBuilder(expected_returns, cov_matrix, risk_aversion=risk_aversion)
    Q = builder.build(constraints)
    n = Q.shape[0]
    cost_diag = cost_diagonal_from_qubo(Q)

    if use_sk_warmstart:
        gamma0, beta0 = sk_gamma_beta(p)
    else:
        rng = np.random.default_rng(42)
        gamma0 = rng.uniform(0, np.pi, size=p)
        beta0 = rng.uniform(0, np.pi / 2, size=p)

    if optimize:
        def objective(x: np.ndarray) -> float:
            g, b = x[:p], x[p:]
            sv = run_fast_qaoa(cost_diag, g, b, n)
            return expected_cost(sv, cost_diag)

        x0 = np.concatenate([gamma0, beta0])
        res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
        gamma, beta = res.x[:p], res.x[p:]
    else:
        gamma, beta = gamma0, beta0

    sv = run_fast_qaoa(cost_diag, gamma, beta, n)
    probs = np.abs(sv) ** 2
    ec = expected_cost(sv, cost_diag)

    return FastQAOAResult(
        expected_cost=ec,
        probabilities=probs,
        cost_diag=cost_diag,
        gamma=np.asarray(gamma),
        beta=np.asarray(beta),
        metadata=RunMetadata(algorithm="qokit_finance.fast_qaoa", generation=None, adapted_from="QOKit (arXiv:2309.04841)"),
    )


@ALGORITHM_REGISTRY.register("qokit_finance_fast_qaoa")
class QOKitFastQAOA:
    """Registry entry point wrapping `solve_portfolio_fast`."""

    def solve(self, expected_returns, cov_matrix, constraints, **kwargs) -> FastQAOAResult:
        return solve_portfolio_fast(expected_returns, cov_matrix, constraints, **kwargs)
