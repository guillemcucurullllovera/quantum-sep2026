"""Common benchmark framework: compare two callables ("method A" and "method
B" — typically a quantum algorithm and its classical baseline) on the same
problem instance and capture a standardized BenchmarkResult.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from core.schemas import BenchmarkResult


class MethodOutput:
    """What a benchmarked callable must return: a scalar metric value, plus
    optional circuit-level stats (only meaningful for quantum methods)."""

    def __init__(
        self,
        value: float,
        *,
        circuit_depth: int | None = None,
        two_qubit_gate_count: int | None = None,
        shots: int | None = None,
    ) -> None:
        self.value = value
        self.circuit_depth = circuit_depth
        self.two_qubit_gate_count = two_qubit_gate_count
        self.shots = shots


def run_benchmark(
    *,
    method_a: Callable[[], MethodOutput],
    method_b: Callable[[], MethodOutput],
    method_a_name: str,
    method_b_name: str,
    metric_name: str,
    seed: int | None = None,
    context: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run both methods once (each callable is responsible for its own
    timing-relevant setup) and return a standardized comparison record.
    """
    t0 = time.perf_counter()
    out_a = method_a()
    wall_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_b = method_b()
    wall_b = time.perf_counter() - t0

    return BenchmarkResult(
        method_a=method_a_name,
        method_b=method_b_name,
        metric_name=metric_name,
        value_a=out_a.value,
        value_b=out_b.value,
        wall_time_a_s=wall_a,
        wall_time_b_s=wall_b,
        circuit_depth=out_a.circuit_depth if out_a.circuit_depth is not None else out_b.circuit_depth,
        two_qubit_gate_count=(
            out_a.two_qubit_gate_count if out_a.two_qubit_gate_count is not None else out_b.two_qubit_gate_count
        ),
        shots=out_a.shots if out_a.shots is not None else out_b.shots,
        seed=seed,
        context=context or {},
    )
