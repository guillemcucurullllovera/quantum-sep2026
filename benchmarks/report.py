"""Render a BenchmarkResult (or a list of them) as a bar-chart comparison
figure, saved through core.result_store so it lands next to the run's other
artifacts under a consistent run_id-addressed path.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — this platform never assumes a display
import matplotlib.pyplot as plt

from core.result_store import ResultStore
from core.schemas import BenchmarkResult


def render_comparison_chart(results: list[BenchmarkResult], *, run_id: str, name: str = "benchmark_chart.png") -> str:
    if not results:
        raise ValueError("render_comparison_chart requires at least one BenchmarkResult")

    labels = [r.metric_name for r in results]
    values_a = [r.value_a for r in results]
    values_b = [r.value_b for r in results]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.2), 4))
    x = range(len(results))
    width = 0.35
    ax.bar([i - width / 2 for i in x], values_a, width, label=results[0].method_a)
    ax.bar([i + width / 2 for i in x], values_b, width, label=results[0].method_b)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_title("Quantum vs classical benchmark comparison")
    fig.tight_layout()

    store = ResultStore()
    path = store.path_for(run_id, name)
    fig.savefig(path)
    plt.close(fig)
    return str(path)
