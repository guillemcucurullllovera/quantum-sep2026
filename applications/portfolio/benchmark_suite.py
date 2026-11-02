"""Cross-generation benchmarking integrated with this platform's shared
`benchmarks.runner` harness and `core.experiment_registry` -- every solve
this module runs is logged as a real `ExperimentResult`, not just printed,
matching the platform-wide policy every other algorithm here follows.
"""
from __future__ import annotations

from applications.portfolio.domain import PortfolioSpec, SolverChoice
from applications.portfolio.exceptions import SolverFailureError
from applications.portfolio.multi_solver import ALL_SOLVER_CHOICES, solve_portfolio
from benchmarks.runner import MethodOutput, run_benchmark
from core.errors import CredentialError, ValidationError
from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, BenchmarkResult, ExperimentConfig, ExperimentResult, RunMetadata


def benchmark_pair(spec: PortfolioSpec, method_a: SolverChoice, method_b: SolverChoice, seed: int = 42) -> BenchmarkResult:
    """A single head-to-head comparison via `benchmarks.runner.run_benchmark`
    (this platform's shared quantum-vs-classical harness) rather than
    hand-timing the two solves directly."""

    def _make(choice: SolverChoice):
        def _run() -> MethodOutput:
            r = solve_portfolio(spec, choice, seed=seed)
            return MethodOutput(value=r.cost)
        return _run

    return run_benchmark(
        method_a=_make(method_a), method_b=_make(method_b),
        method_a_name=method_a.value, method_b_name=method_b.value,
        metric_name="portfolio_qubo_cost", seed=seed,
        context={"n_assets": spec.n_assets()},
    )


def run_full_generation_benchmark(
    spec: PortfolioSpec,
    *,
    choices: list[SolverChoice] | None = None,
    seed: int = 42,
    registry: ExperimentRegistry | None = None,
) -> tuple[list[ExperimentResult], dict[str, str]]:
    """Solve `spec` with every requested solver generation and log each as a
    real `ExperimentResult` through `core.experiment_registry`. A solver
    that is genuinely inapplicable to this spec (too large, an unsupported
    constraint combination, a missing hardware credential) is recorded in
    the returned `failures` dict with its real error message rather than
    silently skipped or allowed to abort the whole benchmark -- but only
    for this platform's own typed exceptions; anything else still
    propagates, so a real bug is never mistaken for "solver doesn't apply
    here".
    """
    resolved = list(choices) if choices is not None else list(ALL_SOLVER_CHOICES)
    registry = registry or ExperimentRegistry()
    results: list[ExperimentResult] = []
    failures: dict[str, str] = {}

    for choice in resolved:
        try:
            r = solve_portfolio(spec, choice, seed=seed)
        except (ValidationError, SolverFailureError, CredentialError) as exc:
            failures[choice.value] = str(exc)
            continue

        backend = BackendDescriptor(name=choice.value, kind=BackendKind.SIMULATOR, num_qubits=spec.n_assets())
        config = ExperimentConfig(
            name=f"portfolio_{choice.value}", algorithm=choice.value, application="portfolio",
            backend=backend, seed=seed, params={"n_assets": spec.n_assets()},
        )
        meta = RunMetadata(algorithm=choice.value, generation=choice.value, seed=seed)
        exp_result = ExperimentResult(
            metadata=meta, config=config,
            metrics={"cost": r.cost, "wall_time_s": r.wall_time_s, "feasible": float(r.feasible)},
            notes=f"solver={choice.value}; violations={r.violations}" if r.violations else f"solver={choice.value}",
        )
        registry.record(exp_result)
        results.append(exp_result)

    return results, failures
