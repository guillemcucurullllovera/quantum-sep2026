"""Post-hoc aggregation of registry results into a report-ready table."""
from __future__ import annotations

from core.experiment_registry import ExperimentRegistry
from core.schemas import ExperimentResult


def to_rows(results: list[ExperimentResult]) -> list[dict]:
    """Flatten a list of ExperimentResults into plain dicts suitable for a
    CSV/DataFrame/markdown table: one row per run, metrics as columns.
    """
    rows = []
    for r in results:
        row = {
            "run_id": r.metadata.run_id,
            "algorithm": r.metadata.algorithm,
            "generation": r.metadata.generation,
            "adapted_from": r.metadata.adapted_from,
            "created_at": r.metadata.created_at,
        }
        row.update({f"metric.{k}": v for k, v in r.metrics.items()})
        if r.classical_baseline_metrics:
            row.update({f"classical.{k}": v for k, v in r.classical_baseline_metrics.items()})
        rows.append(row)
    return rows


def aggregate_by_algorithm(*, registry: ExperimentRegistry | None = None) -> dict[str, list[dict]]:
    """One row-set per algorithm currently present in the registry."""
    registry = registry or ExperimentRegistry()
    out: dict[str, list[dict]] = {}
    with registry._connect() as conn:  # read-only aggregate query, registry has no public "list all algorithms" yet
        algos = [r[0] for r in conn.execute("SELECT DISTINCT algorithm FROM experiments").fetchall()]
    for algo in algos:
        results = [registry.get(rid) for rid in registry.list_by_algorithm(algo)]
        out[algo] = to_rows(results)
    return out
