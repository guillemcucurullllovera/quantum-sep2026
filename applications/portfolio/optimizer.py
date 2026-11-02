"""PortfolioOptimizer application: composes qubo_builder + solvers +
classical_baseline into one entry point, with a benchmark-comparison report
(QAOA vs exact vs simulated annealing on the same problem instance)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.errors import ValidationError
from core.registry import APPLICATION_REGISTRY
from algorithms.qaoa_qubo.classical_baseline import brute_force_k_hot, simulated_annealing_k_hot
from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints, PortfolioQUBOBuilder
from algorithms.qaoa_qubo.solvers import NumpyQAOASolver


@dataclass
class PortfolioOptimizationReport:
    qaoa_bitstring: str
    qaoa_cost: float
    sa_cost: float
    exact_cost: float | None
    qaoa_approx_ratio: float | None
    sa_approx_ratio: float | None
    selected_assets: list[str]


class PortfolioOptimizer:
    """Cardinality-constrained portfolio construction, benchmarked against
    classical baselines on every call — this platform's policy is that no
    quantum result is reported without one."""

    def __init__(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, asset_names: list[str] | None = None):
        self.mu = np.asarray(expected_returns, dtype=float)
        self.sigma = np.asarray(cov_matrix, dtype=float)
        self.n = len(self.mu)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n)]
        self.builder = PortfolioQUBOBuilder(self.mu, self.sigma)

    def optimize(
        self,
        constraints: PortfolioConstraints,
        *,
        qaoa_p: int = 2,
        qaoa_n_iter: int = 150,
        seed: int = 42,
        exact_max_n: int = 18,
        sa_sweeps: int = 3000,
    ) -> PortfolioOptimizationReport:
        if constraints.cardinality_k is None:
            raise ValidationError(
                "PortfolioOptimizer.optimize requires constraints.cardinality_k to be set "
                "(the exact-k-hot baselines and Dicke-state QAOA init both need it)"
            )
        k = constraints.cardinality_k
        Q = self.builder.build(constraints)

        qaoa = NumpyQAOASolver(p=qaoa_p, n_iter=qaoa_n_iter, seed=seed)
        qaoa_result = qaoa.solve(Q, cardinality_k=k)
        qaoa_x = np.array([int(b) for b in qaoa_result.best_bitstring], dtype=float)
        qaoa_cost = float(qaoa_x @ Q @ qaoa_x)

        sa_x, sa_cost, _ = simulated_annealing_k_hot(Q, k, n_sweeps=sa_sweeps, seed=seed)

        exact_cost = None
        exact_ratio = None
        if self.n <= exact_max_n:
            _exact_x, exact_cost = brute_force_k_hot(Q, k)

        qaoa_ratio = qaoa_cost / exact_cost if exact_cost not in (None, 0) else None
        sa_ratio = sa_cost / exact_cost if exact_cost not in (None, 0) else None

        selected = [self.asset_names[i] for i in range(self.n) if qaoa_x[i] > 0.5]

        return PortfolioOptimizationReport(
            qaoa_bitstring=qaoa_result.best_bitstring,
            qaoa_cost=qaoa_cost,
            sa_cost=sa_cost,
            exact_cost=exact_cost,
            qaoa_approx_ratio=qaoa_ratio,
            sa_approx_ratio=sa_ratio,
            selected_assets=selected,
        )


@APPLICATION_REGISTRY.register("portfolio")
def run_portfolio_optimization(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    constraints: PortfolioConstraints,
    **kwargs,
) -> PortfolioOptimizationReport:
    return PortfolioOptimizer(expected_returns, cov_matrix).optimize(constraints, **kwargs)
