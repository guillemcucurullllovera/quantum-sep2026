"""Real QAOA-vs-classical comparison, wired to actual algorithm modules.

Note on integration: algorithms.qaoa_qubo/qae_iqae/qrc/pqk register
themselves in core.registry.ALGORITHM_REGISTRY, but their native call
signatures (e.g. solve_qubo(Q, engine=...) -> QAOASolverResult) do not match
what experiments.runners.generic_runner.run_experiment expects (a callable
taking (config: ExperimentConfig, seed) -> ExperimentResult). This is a real,
previously-undetected gap between the registry contract and how algorithms
actually registered themselves — flagged in the report, not fixed
here (algorithms/ is out of this module's scope). Every benchmark in this
package therefore calls the algorithms' real native APIs directly rather
than going through the (currently non-functional for this purpose)
ALGORITHM_REGISTRY/generic_runner path.
"""
from __future__ import annotations

from applications.portfolio.synthetic_market import generate_synthetic_market
from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints, PortfolioQUBOBuilder
from algorithms.qaoa_qubo.solvers import NumpyQAOASolver
from classical.optimization.brute_force import brute_force_qubo
from classical.optimization.simulated_annealing import simulated_annealing_qubo
from benchmarks.runner import MethodOutput, run_benchmark


def qaoa_vs_classical_small_portfolio(n_assets: int = 6, seed: int = 0):
    """n_assets small enough for brute force to be exact ground truth."""
    mu, sigma, _ = generate_synthetic_market(n_assets, seed=seed)
    Q = PortfolioQUBOBuilder(mu, sigma).build(PortfolioConstraints(cardinality_k=n_assets // 2))

    def run_qaoa() -> MethodOutput:
        result = NumpyQAOASolver(p=2, seed=seed).solve(Q)
        return MethodOutput(value=result.best_cost)

    def run_classical() -> MethodOutput:
        _, obj = brute_force_qubo(Q)
        return MethodOutput(value=obj)

    return run_benchmark(
        method_a=run_qaoa,
        method_b=run_classical,
        method_a_name="qaoa_numpy",
        method_b_name="brute_force_exact",
        metric_name="qubo_objective",
        seed=seed,
        context={"n_assets": n_assets},
    )


def qaoa_vs_simulated_annealing(n_assets: int = 12, seed: int = 0):
    """n_assets too large for brute force; SA is the ground-truth proxy here."""
    mu, sigma, _ = generate_synthetic_market(n_assets, seed=seed)
    Q = PortfolioQUBOBuilder(mu, sigma).build(PortfolioConstraints(cardinality_k=n_assets // 2))

    def run_qaoa() -> MethodOutput:
        result = NumpyQAOASolver(p=3, seed=seed).solve(Q)
        return MethodOutput(value=result.best_cost)

    def run_sa() -> MethodOutput:
        _, obj = simulated_annealing_qubo(Q, seed=seed)
        return MethodOutput(value=obj)

    return run_benchmark(
        method_a=run_qaoa,
        method_b=run_sa,
        method_a_name="qaoa_numpy",
        method_b_name="simulated_annealing",
        metric_name="qubo_objective",
        seed=seed,
        context={"n_assets": n_assets},
    )
