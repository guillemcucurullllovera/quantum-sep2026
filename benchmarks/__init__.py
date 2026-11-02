"""Asymmetrica Quantum Platform — benchmarks package."""
from benchmarks.runner import MethodOutput, run_benchmark
from benchmarks.report import render_comparison_chart
from benchmarks.reproducibility import check_reproducibility, ReproducibilityCheckResult

__all__ = [
    "MethodOutput",
    "run_benchmark",
    "render_comparison_chart",
    "check_reproducibility",
    "ReproducibilityCheckResult",
]
