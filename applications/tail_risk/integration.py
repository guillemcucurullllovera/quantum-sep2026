"""Experiment-registry logging and benchmarks.runner integration for the
tail-risk application layer, so a multi-method comparison run is a real,
queryable experiment (`core.experiment_registry`) and the QAE-vs-classical
head-to-head goes through the platform's shared `benchmarks.runner`
framework rather than an ad hoc print statement.
"""
from __future__ import annotations

from benchmarks.runner import MethodOutput, run_benchmark
from core.experiment_registry import ExperimentRegistry
from core.metrics import value_at_risk
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    BenchmarkResult,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)

from applications.tail_risk.comparison import _qae_located_var_cvar
from applications.tail_risk.domain import TailRiskComparisonReport
from applications.tail_risk.errors import validate_pnl_sample

__all__ = ["log_comparison", "benchmark_qae_vs_historical"]


def log_comparison(
    report: TailRiskComparisonReport, *, application: str = "tail_risk", seed: int = 42,
) -> str:
    """Flatten a `TailRiskComparisonReport` into one `ExperimentResult`
    (per-method VaR/CVaR/wall-time as metrics) and record it via
    `core.experiment_registry.ExperimentRegistry`. Returns the run_id.
    """
    metrics: dict[str, float] = {
        "alpha": report.alpha, "n_samples": float(report.n_samples), "prob_of_loss": report.prob_of_loss,
        "var_spread": report.var_spread(), "cvar_spread": report.cvar_spread(),
    }
    for name, m in report.results.items():
        metrics[f"{name}_var"] = m.var
        metrics[f"{name}_cvar"] = m.cvar
        metrics[f"{name}_wall_time_s"] = m.wall_time_s

    backend = BackendDescriptor(name="local_simulator", kind=BackendKind.SIMULATOR, num_qubits=0)
    config = ExperimentConfig(
        name="tail_risk_multi_method_comparison", algorithm="tail_risk_comparison", application=application,
        backend=backend, seed=seed, params={"alpha": report.alpha, "methods": sorted(report.results)},
    )
    metadata = RunMetadata(algorithm="tail_risk_comparison", generation=None, seed=seed)
    result = ExperimentResult(
        metadata=metadata, config=config, metrics=metrics, notes="; ".join(report.notes) or None,
    )
    return ExperimentRegistry().record(result)


def benchmark_qae_vs_historical(
    pnl_samples, alpha: float = 0.95, *, n_grid_qubits: int = 3, epsilon: float = 0.03, shots: int = 1024,
    seed: int = 42,
) -> BenchmarkResult:
    """Head-to-head QAE-located VaR vs. historical VaR through the shared
    `benchmarks.runner` framework -- wall-clock cost is what actually
    differs (QAE runs `n_grid_qubits`-ish simulator circuits; historical is
    one `np.percentile` call), which is exactly what `run_benchmark` times.
    """
    pnl = validate_pnl_sample(pnl_samples)

    def method_a() -> MethodOutput:
        var, _cvar, extra = _qae_located_var_cvar(
            pnl, alpha, n_grid_qubits=n_grid_qubits, epsilon=epsilon, shots=shots, seed=seed,
        )
        return MethodOutput(value=var, shots=shots)

    def method_b() -> MethodOutput:
        return MethodOutput(value=value_at_risk(pnl, alpha))

    return run_benchmark(
        method_a=method_a, method_b=method_b,
        method_a_name="qae_amplitude_estimation", method_b_name="historical",
        metric_name="var_at_alpha", seed=seed, context={"alpha": alpha, "n_samples": int(pnl.size)},
    )
