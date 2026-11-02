"""Comprehensive, honest comparison report: quantum (QRC Gen1-informed /
Gen2 / Gen2-original / arXiv-faithful) vs. classical (ESN, Mamba SSM, naive
baselines) on identical rolling-evaluation data. No assumed winner --
whichever method has the lowest mean walk-forward MSE is reported as best,
and the arXiv replication's own documented accuracy-gate failure is
surfaced explicitly whenever that method is included, not hidden.
"""
from __future__ import annotations

from applications.time_series.domain import MultiMethodForecastResult


def format_report(result: MultiMethodForecastResult) -> str:
    lines = [
        f"Multi-method forecast comparison: dataset={result.dataset_name!r} "
        f"n={result.n_samples} features={result.n_features} target={result.spec.target_kind.value} "
        f"validation={result.spec.validation.value}",
        "",
    ]

    def _sort_key(item):
        name, r = item
        if r.failed:
            return (2, 0.0)
        if r.mean_mse != r.mean_mse:  # NaN
            return (1, 0.0)
        return (0, r.mean_mse)

    for name, r in sorted(result.results.items(), key=_sort_key):
        tag = "quantum" if r.is_quantum else "classical"
        if r.failed:
            lines.append(f"  {name:22s} [{tag:9s}] FAILED: {r.error}")
            continue
        marker = "  <- best" if name == result.best_method else ""
        lines.append(
            f"  {name:22s} [{tag:9s}] MSE={r.mean_mse:.6f} (+/-{r.std_mse:.6f})  MAE={r.mean_mae:.6f}  "
            f"{len(r.fold_metrics)} folds  {r.wall_time_s * 1000:.1f}ms{marker}"
        )
        if r.notes:
            lines.append(f"      {r.notes}")

    if result.naive_floor_mse is not None:
        beaters = [
            n for n, r in result.results.items()
            if not r.failed and r.mean_mse == r.mean_mse and r.mean_mse < result.naive_floor_mse
            and n != "naive_constant"
        ]
        lines.append("")
        lines.append(f"  naive-constant floor MSE={result.naive_floor_mse:.6f}; beaten by: {beaters or 'none'}")

    if result.arxiv_gate_note:
        lines.append("")
        lines.append(f"  NOTE: {result.arxiv_gate_note}")

    if result.best_quantum_vs_best_classical is not None:
        b = result.best_quantum_vs_best_classical
        lines.append("")
        lines.append(
            f"  benchmarks.runner comparison: {b.method_a} (MSE={b.value_a:.6f}) vs. "
            f"{b.method_b} (MSE={b.value_b:.6f})"
        )

    if result.noise_comparison is not None:
        nc = result.noise_comparison
        lines.append("")
        lines.append(
            f"  Noise-aware check ({nc.method.value}): noiseless MSE={nc.noiseless_mse:.6f}, "
            f"noisy MSE={nc.noisy_mse:.6f} ({nc.mse_degradation_ratio:.2f}x degradation)"
        )
        if nc.notes:
            lines.append(f"      {nc.notes}")

    return "\n".join(lines)


__all__ = ["format_report"]
