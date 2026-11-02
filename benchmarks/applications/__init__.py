"""End-to-end application-layer benchmarks -- not just an algorithm in
isolation (that's benchmarks.quantum_vs_classical), but the full
applications.* entry point a real user would call, which already bundles
its own classical comparison by this platform's design convention.
"""
from __future__ import annotations

from applications.portfolio.optimizer import PortfolioOptimizer
from applications.portfolio.synthetic_market import generate_synthetic_market
from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints
from core.schemas import BenchmarkResult


def portfolio_application_benchmark(n_assets: int = 10, cardinality: int = 5, seed: int = 0) -> BenchmarkResult:
    mu, sigma, names = generate_synthetic_market(n_assets, seed=seed)
    optimizer = PortfolioOptimizer(mu, sigma, names)
    report = optimizer.optimize(PortfolioConstraints(cardinality_k=cardinality), seed=seed)

    return BenchmarkResult(
        method_a="qaoa_portfolio_application",
        method_b="simulated_annealing_baseline",
        metric_name="qubo_cost",
        value_a=report.qaoa_cost,
        value_b=report.sa_cost,
        seed=seed,
        context={
            "n_assets": n_assets,
            "cardinality": cardinality,
            "exact_cost": report.exact_cost,
            "qaoa_approx_ratio": report.qaoa_approx_ratio,
            "sa_approx_ratio": report.sa_approx_ratio,
            "selected_assets": report.selected_assets,
        },
    )
