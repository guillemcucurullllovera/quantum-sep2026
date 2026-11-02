"""python -m tools.reporting --run-ids id1 id2 id3 --output report.png

CLI front door to benchmarks.report: pull recorded BenchmarkResults are not
directly queryable by id in the current registry (it stores ExperimentResult,
not BenchmarkResult), so this tool works at the ExperimentResult level --
render a comparison chart across N recorded experiment runs' metrics.
"""
from __future__ import annotations

import argparse
import sys

from core.experiment_registry import ExperimentRegistry
from core.schemas import BenchmarkResult
from benchmarks.report import render_comparison_chart


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.reporting")
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--metric", required=True, help="metric name to compare across runs")
    parser.add_argument("--output-name", default="cli_report.png")
    args = parser.parse_args(argv)

    registry = ExperimentRegistry()
    results = []
    for rid in args.run_ids:
        r = registry.get(rid)
        if args.metric not in r.metrics:
            print(f"run {rid} has no metric {args.metric!r} (has: {list(r.metrics)})", file=sys.stderr)
            continue
        results.append(r)

    if len(results) < 2:
        print("Need at least 2 runs with the requested metric to compare.", file=sys.stderr)
        return 1

    benchmark_results = []
    for a, b in zip(results, results[1:]):
        benchmark_results.append(
            BenchmarkResult(
                method_a=a.metadata.algorithm,
                method_b=b.metadata.algorithm,
                metric_name=args.metric,
                value_a=a.metrics[args.metric],
                value_b=b.metrics[args.metric],
            )
        )

    path = render_comparison_chart(benchmark_results, run_id=args.run_ids[0], name=args.output_name)
    print(f"Wrote report to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
