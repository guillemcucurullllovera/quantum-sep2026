"""Unified multi-method forecasting dispatcher -- the one API this platform
builds so quantum (QRC Gen1-informed / Gen2 / Gen2-original / arXiv-faithful)
and classical (ESN, Mamba SSM, naive baselines) methods are run through
identical rolling walk-forward evaluation on the identical preprocessed
series, and reported together honestly -- no assumed winner, and the arXiv
replication's documented accuracy-gate failure is surfaced explicitly
whenever that method is included, not hidden.
"""
from __future__ import annotations

import numpy as np

from applications.time_series.domain import (
    QUANTUM_METHODS,
    MethodChoice,
    MethodResult,
    MultiMethodForecastResult,
    TimeSeriesForecastSpec,
)
from applications.time_series.evaluation import rolling_evaluate_method
from applications.time_series.noise import compare_noisy_vs_noiseless
from applications.time_series.preprocessing import build_pipeline
from benchmarks.runner import MethodOutput, run_benchmark
from core.dataset_interfaces import FinancialDataset
from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, BenchmarkResult, ExperimentConfig, ExperimentResult, RunMetadata

ARXIV_GATE_NOTE = (
    "MethodChoice.QRC_GEN1_ARXIV replays algorithms_extended.qrc_arxiv_replication, "
    "which documents (THIRD_PARTY_NOTICES.md) that its from-scratch reimplementation of "
    "arXiv:2505.13933 did NOT pass its own accuracy gate on the authors' data "
    "(best MSE 0.2066 vs paper target 0.105, 1.97x over the 1.1x threshold; "
    "GATE_PASSED=False by a hard assertion in that module's __init__.py). Its "
    "result in this comparison is reported as-is, not adjusted for that history."
)


def _log_experiment(registry: ExperimentRegistry, method: MethodChoice, result: MethodResult, spec: TimeSeriesForecastSpec) -> str:
    backend = BackendDescriptor(
        name="numpy_statevector" if result.is_quantum else "classical",
        kind=BackendKind.SIMULATOR,
        num_qubits=(spec.qrc.n1 + spec.qrc.n2) if result.is_quantum else 0,
    )
    exp_config = ExperimentConfig(
        name="time_series_multi_method_forecast",
        algorithm=method.value,
        application="time_series",
        backend=backend,
        seed=spec.seed,
        params={
            "target_kind": spec.target_kind.value,
            "validation": spec.validation.value,
            "n_splits": spec.n_splits,
            "horizon": spec.horizon,
            "use_fracdiff": spec.features.use_fracdiff,
        },
    )
    metadata = RunMetadata(
        algorithm=method.value,
        generation="gen2" if method == MethodChoice.QRC_GEN2 else ("gen1" if method == MethodChoice.QRC_GEN1_ARXIV else None),
        seed=spec.seed,
    )
    exp_result = ExperimentResult(
        metadata=metadata,
        config=exp_config,
        metrics={
            "mse_mean": result.mean_mse,
            "mse_std": result.std_mse,
            "mae_mean": result.mean_mae,
            "n_folds_evaluated": float(len(result.fold_metrics)),
            "wall_time_s": result.wall_time_s,
        },
        notes=result.notes,
    )
    return registry.record(exp_result)


def _best_quantum_vs_best_classical(results: dict[str, MethodResult], seed: int) -> BenchmarkResult | None:
    """Wire benchmarks.runner's standard two-method comparison framework
    into the dispatcher: pick whichever quantum method and whichever
    non-naive classical method scored lowest mean MSE, and record a
    standardized BenchmarkResult for them. The methods are already
    evaluated (rolling_evaluate_method already ran both), so the callables
    passed to run_benchmark just replay the already-measured value/wall
    time rather than recomputing -- run_benchmark's own per-call timing
    would otherwise measure only the (near-zero) lambda overhead, which
    would misrepresent the real cost; using its MethodOutput/BenchmarkResult
    schema is what matters here, not re-timing.
    """
    quantum = {k: r for k, r in results.items() if r.method in QUANTUM_METHODS and not r.failed and np.isfinite(r.mean_mse)}
    classical = {
        k: r for k, r in results.items()
        if r.method not in QUANTUM_METHODS and "naive" not in k and not r.failed and np.isfinite(r.mean_mse)
    }
    if not quantum or not classical:
        return None
    best_q_name = min(quantum, key=lambda k: quantum[k].mean_mse)
    best_c_name = min(classical, key=lambda k: classical[k].mean_mse)
    best_q, best_c = quantum[best_q_name], classical[best_c_name]

    return run_benchmark(
        method_a=lambda: MethodOutput(best_q.mean_mse),
        method_b=lambda: MethodOutput(best_c.mean_mse),
        method_a_name=best_q_name,
        method_b_name=best_c_name,
        metric_name="walk_forward_mean_mse",
        seed=seed,
        context={"best_quantum_wall_time_s": best_q.wall_time_s, "best_classical_wall_time_s": best_c.wall_time_s},
    )


def run_dispatcher(
    dataset: FinancialDataset, spec: TimeSeriesForecastSpec, log_experiments: bool = True,
) -> MultiMethodForecastResult:
    """Run every method in `spec.methods` through identical preprocessing
    and rolling evaluation, and return one normalized comparison result.
    """
    X, y = dataset.features(), dataset.targets()
    X_proc, y_proc = build_pipeline(X, y, spec.features, spec.horizon)

    registry = ExperimentRegistry() if log_experiments else None
    results: dict[str, MethodResult] = {}
    run_ids: dict[str, str] = {}

    for method in spec.methods:
        result = rolling_evaluate_method(method, X_proc, y_proc, spec)
        results[method.value] = result
        if registry is not None and not result.failed:
            run_ids[method.value] = _log_experiment(registry, method, result, spec)

    naive = results.get(MethodChoice.NAIVE_CONSTANT.value)
    naive_floor_mse = naive.mean_mse if naive is not None and not naive.failed else None

    candidates = {k: r for k, r in results.items() if not r.failed and not np.isnan(r.mean_mse)}
    best_method = min(candidates, key=lambda k: candidates[k].mean_mse) if candidates else None

    arxiv_note = ARXIV_GATE_NOTE if MethodChoice.QRC_GEN1_ARXIV in spec.methods else None

    noise_comparison = None
    if spec.noise.enabled:
        noise_comparison = compare_noisy_vs_noiseless(X_proc, y_proc, spec, log_experiments=log_experiments)

    benchmark = _best_quantum_vs_best_classical(results, spec.seed)

    return MultiMethodForecastResult(
        spec=spec,
        dataset_name=dataset.descriptor.name,
        n_samples=len(y_proc),
        n_features=X_proc.shape[1],
        results=results,
        best_method=best_method,
        naive_floor_mse=naive_floor_mse,
        arxiv_gate_note=arxiv_note,
        noise_comparison=noise_comparison,
        best_quantum_vs_best_classical=benchmark,
        run_ids=run_ids,
    )


__all__ = ["run_dispatcher", "ARXIV_GATE_NOTE"]
