"""Statistical summaries over ExperimentResults pulled from the registry.

Real, small statistics functions — not a dashboard, just the arithmetic a
report or notebook would otherwise reimplement per-notebook.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from core.experiment_registry import ExperimentRegistry
from core.schemas import ExperimentResult


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    n: int
    mean: float
    stdev: float
    minimum: float
    maximum: float


def summarize_metric(results: list[ExperimentResult], metric: str) -> MetricSummary:
    values = [r.metrics[metric] for r in results if metric in r.metrics]
    if not values:
        raise ValueError(f"No results contain metric '{metric}'")
    return MetricSummary(
        metric=metric,
        n=len(values),
        mean=statistics.fmean(values),
        stdev=statistics.pstdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
    )


def summarize_algorithm(algorithm: str, *, registry: ExperimentRegistry | None = None) -> list[ExperimentResult]:
    """Fetch every logged run for an algorithm, most recent first."""
    registry = registry or ExperimentRegistry()
    return [registry.get(run_id) for run_id in registry.list_by_algorithm(algorithm)]


def compare_quantum_vs_classical(result: ExperimentResult) -> dict[str, float] | None:
    """Return {metric: quantum - classical} for every metric present in both
    the quantum result and its logged classical_baseline_metrics, or None if
    no baseline was recorded for this run.
    """
    if not result.classical_baseline_metrics:
        return None
    return {
        m: result.metrics[m] - result.classical_baseline_metrics[m]
        for m in result.metrics
        if m in result.classical_baseline_metrics
    }
