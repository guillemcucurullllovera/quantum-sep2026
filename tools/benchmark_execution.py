"""python -m tools.benchmark_execution --method-a name_a --method-b name_b

Thin CLI over benchmarks.runner.run_benchmark for ad hoc two-method
comparisons without writing a script.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable

from benchmarks.runner import MethodOutput, run_benchmark

# Registry of nullary callables usable directly from the CLI -- kept small
# and explicit rather than importing every algorithm eagerly. Each callable
# must return a benchmarks.runner.MethodOutput.
_QUICK_BENCHMARKS: dict[str, Callable[[], MethodOutput]] = {}


def register_quick_benchmark(name: str, fn: Callable[[], MethodOutput]) -> None:
    """Applications/algorithms can register a zero-arg callable (returning
    a MethodOutput) here at import time for quick CLI comparisons."""
    _QUICK_BENCHMARKS[name] = fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.benchmark_execution")
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--metric-name", default="value")
    parser.add_argument("--list", action="store_true", help="list registered quick benchmarks")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(_QUICK_BENCHMARKS):
            print(name)
        return 0

    for name in (args.method_a, args.method_b):
        if name not in _QUICK_BENCHMARKS:
            print(f"Unknown quick-benchmark method {name!r}. Registered: {sorted(_QUICK_BENCHMARKS)}", file=sys.stderr)
            return 1

    result = run_benchmark(
        method_a=_QUICK_BENCHMARKS[args.method_a],
        method_b=_QUICK_BENCHMARKS[args.method_b],
        method_a_name=args.method_a,
        method_b_name=args.method_b,
        metric_name=args.metric_name,
    )
    print(f"{args.method_a}: {result.value_a}  (wall={result.wall_time_a_s:.4f}s)")
    print(f"{args.method_b}: {result.value_b}  (wall={result.wall_time_b_s:.4f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
