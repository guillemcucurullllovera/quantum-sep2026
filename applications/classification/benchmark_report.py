"""Comprehensive multi-method comparison report: run every applicable
algorithm on one shared benchmark task through the SAME dispatcher
interface, log the result through `core.experiment_registry`, and report
honestly -- no assumed winner, consistent with this platform's documented
no-overclaiming policy (see `algorithms/vqc_qml/__init__.py`,
`applications/classification/three_way_comparison.py`).

Where `three_way_comparison.py` hand-wires exactly 4 methods for one
specific dataset, `run_comprehensive_comparison` below asks the dispatcher
which algorithms actually support the requested `task_type`
(`dispatcher.supported_task_types`) and runs all of them -- so adding a new
algorithm to the dispatcher's registry makes it show up here automatically,
with no edit needed in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from applications.classification.dispatcher import default_metric_for, dispatch, supported_task_types
from applications.classification.result import DispatchResult
from applications.classification.task_spec import Algorithm, ClassificationTaskSpec, TaskType, QUANTUM_ALGORITHMS
from benchmarks.runner import MethodOutput, run_benchmark
from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, BenchmarkResult, ExperimentConfig, ExperimentResult, RunMetadata


@dataclass
class MethodOutcome:
    algorithm: str
    metric_value: float
    result: DispatchResult | None
    error: str | None = None


@dataclass
class ComprehensiveComparisonReport:
    task_type: str
    metric_name: str
    outcomes: list[MethodOutcome] = field(default_factory=list)
    higher_is_better: bool = True

    @property
    def successful(self) -> list[MethodOutcome]:
        return [o for o in self.outcomes if o.error is None]

    @property
    def best_method(self) -> str | None:
        ok = self.successful
        if not ok:
            return None
        key = (lambda o: o.metric_value) if self.higher_is_better else (lambda o: -o.metric_value)
        return max(ok, key=key).algorithm

    def to_summary_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "best_method": self.best_method,
            "outcomes": [
                {"algorithm": o.algorithm, "metric_value": o.metric_value, "error": o.error} for o in self.outcomes
            ],
        }


_LOWER_IS_BETTER = {"mse", "rmse", "mae", "calibration_error"}


def run_comprehensive_comparison(
    task_type: TaskType,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int = 4,
    seed: int = 42,
    metric_name: str | None = None,
    algorithms: list[Algorithm] | None = None,
    algorithm_params: dict[Algorithm, dict] | None = None,
    record: bool = False,
) -> ComprehensiveComparisonReport:
    """Run every algorithm that supports `task_type` (or the explicit
    `algorithms` subset) on the identical (X_train, y_train, X_test, y_test)
    split, through `dispatcher.dispatch` -- one interface, real distinct
    numbers per method, no method special-cased to win.

    A method whose environment dependency is missing (e.g. scikit-learn for
    `CLASSICAL_RANDOM_FOREST`) or that raises for a data-shape reason is
    recorded as a failed outcome with its error message, not silently
    dropped -- so the report is honest about what did and didn't run.
    """
    metric_name = metric_name or default_metric_for(task_type)
    higher_is_better = metric_name not in _LOWER_IS_BETTER
    algorithm_params = algorithm_params or {}

    candidates = algorithms or [a for a in Algorithm if task_type in supported_task_types(a)]

    outcomes: list[MethodOutcome] = []
    for algo in candidates:
        spec = ClassificationTaskSpec(
            task_type=task_type,
            algorithm=algo,
            n_qubits=n_qubits,
            seed=seed,
            algorithm_params=algorithm_params.get(algo, {}),
        )
        try:
            result = dispatch(spec, X_train, y_train, X_test, y_test)
            outcomes.append(MethodOutcome(algorithm=algo.value, metric_value=result.metric(metric_name), result=result))
        except Exception as exc:  # noqa: BLE001 - deliberately broad: one method's failure must not sink the report
            outcomes.append(MethodOutcome(algorithm=algo.value, metric_value=float("nan"), result=None, error=str(exc)))

    report = ComprehensiveComparisonReport(
        task_type=task_type.value, metric_name=metric_name, outcomes=outcomes, higher_is_better=higher_is_better
    )

    if record:
        _record_report(report, n_qubits=n_qubits, seed=seed)

    return report


def _record_report(report: ComprehensiveComparisonReport, n_qubits: int, seed: int) -> str:
    """Log the comparison through `core.experiment_registry`, per the
    platform's benchmarking policy of every comparison run being queryable
    history rather than an ad hoc printed table."""
    backend = BackendDescriptor(name="statevector_exact", kind=BackendKind.SIMULATOR, num_qubits=n_qubits)
    config = ExperimentConfig(
        name="comprehensive_classification_comparison",
        algorithm="classification_dispatcher",
        application="applications.classification.benchmark_report",
        backend=backend,
        seed=seed,
        params={"task_type": report.task_type, "metric_name": report.metric_name},
    )
    meta = RunMetadata(algorithm="classification_dispatcher", generation="wave5", seed=seed)
    metrics = {
        f"{o.algorithm}_{report.metric_name}": o.metric_value for o in report.outcomes if o.error is None
    }
    result = ExperimentResult(
        metadata=meta,
        config=config,
        metrics=metrics,
        notes=(
            f"Comprehensive {report.task_type} comparison across {len(report.outcomes)} methods, "
            f"metric={report.metric_name}. best_method={report.best_method}. "
            f"{sum(1 for o in report.outcomes if o.error)} method(s) failed to run."
        ),
    )
    return ExperimentRegistry().record(result)


def run_best_quantum_vs_classical_benchmark(
    report: ComprehensiveComparisonReport,
) -> BenchmarkResult:
    """Reduce an already-run `ComprehensiveComparisonReport` to a single
    `benchmarks.runner.run_benchmark`-style head-to-head: the best-scoring
    quantum method in the report vs. the best-scoring classical method,
    through `core.schemas.BenchmarkResult` -- the platform-wide benchmark
    record type every `benchmarks/quantum_vs_classical/*` module produces,
    so a dispatcher-driven comparison plugs into the same downstream
    tooling (`benchmarks.report`, `core.experiment_registry`) as a
    hand-written one.

    Raises `ValueError` if the report has no successful outcome on one side
    -- a benchmark with a missing side is not silently half-reported.
    """
    quantum = [o for o in report.successful if Algorithm(o.algorithm) in QUANTUM_ALGORITHMS]
    classical = [o for o in report.successful if Algorithm(o.algorithm) not in QUANTUM_ALGORITHMS]
    if not quantum or not classical:
        raise ValueError(
            f"need at least one successful quantum and one successful classical outcome; "
            f"got {len(quantum)} quantum, {len(classical)} classical"
        )
    pick = (lambda outs: max(outs, key=lambda o: o.metric_value)) if report.higher_is_better else (
        lambda outs: min(outs, key=lambda o: o.metric_value)
    )
    best_quantum, best_classical = pick(quantum), pick(classical)

    def _quantum() -> MethodOutput:
        return MethodOutput(value=best_quantum.metric_value)

    def _classical() -> MethodOutput:
        return MethodOutput(value=best_classical.metric_value)

    return run_benchmark(
        method_a=_quantum, method_b=_classical,
        method_a_name=best_quantum.algorithm, method_b_name=best_classical.algorithm,
        metric_name=report.metric_name,
        context={"task_type": report.task_type, "n_methods_compared": len(report.outcomes)},
    )


__all__ = [
    "MethodOutcome",
    "ComprehensiveComparisonReport",
    "run_comprehensive_comparison",
    "run_best_quantum_vs_classical_benchmark",
]
