"""python -m tools.result_aggregation --algorithm qaoa_qubo

Pulls every recorded run for an algorithm out of the experiment registry
and prints a summary table -- the CLI front door to browsing
core.experiment_registry without writing a script each time.
"""
from __future__ import annotations

import argparse
import sys

from core.experiment_registry import ExperimentRegistry


def aggregate(algorithm: str) -> list[dict]:
    registry = ExperimentRegistry()
    rows = []
    for run_id in registry.list_by_algorithm(algorithm):
        result = registry.get(run_id)
        rows.append(
            {
                "run_id": run_id,
                "created_at": result.metadata.created_at,
                "generation": result.metadata.generation,
                "adapted_from": result.metadata.adapted_from,
                "metrics": result.metrics,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.result_aggregation")
    parser.add_argument("--algorithm", required=True)
    args = parser.parse_args(argv)

    rows = aggregate(args.algorithm)
    if not rows:
        print(f"No recorded runs for algorithm={args.algorithm!r}")
        return 0

    print(f"{len(rows)} run(s) for '{args.algorithm}':")
    for r in rows:
        metric_str = ", ".join(f"{k}={v:.4g}" for k, v in r["metrics"].items())
        print(f"  {r['run_id']}  gen={r['generation']}  {metric_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
