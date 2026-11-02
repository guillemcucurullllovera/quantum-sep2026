"""Tests for `applications.classification.benchmark_report`: the
comprehensive multi-method comparison. Real assertions that it reports
distinct numbers per method with no assumed winner, and that a failing
method is recorded honestly rather than silently dropped."""
from __future__ import annotations

import numpy as np
import pytest

from applications.classification.benchmark_report import (
    run_best_quantum_vs_classical_benchmark,
    run_comprehensive_comparison,
)
from applications.classification.task_spec import Algorithm, TaskType


def _binary_data(n=40, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    n_test = n // 4
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


def test_comprehensive_comparison_runs_every_requested_method_and_reports_honestly():
    X_train, y_train, X_test, y_test = _binary_data()
    algos = [
        Algorithm.VQC_V1_LINEAR,
        Algorithm.CLASSICAL_LOGISTIC_REGRESSION,
        Algorithm.CLASSICAL_RANDOM_FOREST,
        Algorithm.CLASSICAL_SVM_RBF,
    ]
    report = run_comprehensive_comparison(
        TaskType.BINARY, X_train, y_train, X_test, y_test, n_qubits=4, seed=0, algorithms=algos
    )
    assert len(report.outcomes) == 4
    assert {o.algorithm for o in report.outcomes} == {a.value for a in algos}
    for o in report.outcomes:
        assert o.error is None
        assert 0.0 <= o.metric_value <= 1.0

    # best_method must actually be the max, not a hardcoded preference
    best = report.best_method
    assert best is not None
    best_value = next(o.metric_value for o in report.outcomes if o.algorithm == best)
    assert best_value == max(o.metric_value for o in report.outcomes)

    # honest comparison: methods must produce real, not-all-identical numbers
    values = [o.metric_value for o in report.outcomes]
    assert len(set(values)) > 1, "all methods reported identical metrics -- comparison is not real"


def test_comprehensive_comparison_records_failures_without_sinking_the_report():
    X_train, y_train, X_test, y_test = _binary_data()
    # REGIME_HMM does not support BINARY -- forcing it in via the explicit
    # `algorithms` list should record a failed outcome, not raise.
    algos = [Algorithm.CLASSICAL_LOGISTIC_REGRESSION, Algorithm.REGIME_HMM]
    report = run_comprehensive_comparison(
        TaskType.BINARY, X_train, y_train, X_test, y_test, n_qubits=4, seed=0, algorithms=algos
    )
    assert len(report.outcomes) == 2
    ok = [o for o in report.outcomes if o.error is None]
    failed = [o for o in report.outcomes if o.error is not None]
    assert len(ok) == 1 and ok[0].algorithm == "classical_logistic_regression"
    assert len(failed) == 1 and failed[0].algorithm == "regime_hmm"
    assert failed[0].error  # non-empty message
    assert report.best_method == "classical_logistic_regression"


def test_comprehensive_comparison_regression_uses_lower_is_better():
    rng = np.random.default_rng(1)
    n = 40
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.05, size=n)
    n_test = n // 4
    X_train, y_train, X_test, y_test = X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]

    report = run_comprehensive_comparison(
        TaskType.REGRESSION, X_train, y_train, X_test, y_test, n_qubits=4, seed=0,
        algorithms=[Algorithm.CLASSICAL_RIDGE_REGRESSION],
    )
    assert report.metric_name == "rmse"
    assert report.higher_is_better is False
    assert report.best_method == "classical_ridge_regression"


def test_best_quantum_vs_classical_benchmark_uses_benchmarks_runner():
    """run_best_quantum_vs_classical_benchmark reduces a comparison report to
    a core.schemas.BenchmarkResult via benchmarks.runner.run_benchmark --
    the platform-wide benchmark record type, not a bespoke one."""
    from core.schemas import BenchmarkResult

    X_train, y_train, X_test, y_test = _binary_data()
    report = run_comprehensive_comparison(
        TaskType.BINARY, X_train, y_train, X_test, y_test, n_qubits=4, seed=0,
        algorithms=[Algorithm.VQC_V1_LINEAR, Algorithm.CLASSICAL_LOGISTIC_REGRESSION, Algorithm.CLASSICAL_RANDOM_FOREST],
    )
    result = run_best_quantum_vs_classical_benchmark(report)
    assert isinstance(result, BenchmarkResult)
    assert result.method_a == "vqc_v1_linear"
    assert result.method_b in {"classical_logistic_regression", "classical_random_forest"}
    assert result.metric_name == "accuracy"


def test_best_quantum_vs_classical_benchmark_requires_both_sides():
    X_train, y_train, X_test, y_test = _binary_data()
    report = run_comprehensive_comparison(
        TaskType.BINARY, X_train, y_train, X_test, y_test, n_qubits=4, seed=0,
        algorithms=[Algorithm.CLASSICAL_LOGISTIC_REGRESSION],
    )
    with pytest.raises(ValueError):
        run_best_quantum_vs_classical_benchmark(report)


def test_comprehensive_comparison_can_record_to_experiment_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_SEP2026_REGISTRY_DB", str(tmp_path / "registry.sqlite3"))
    from core.config import get_settings
    from core.experiment_registry import ExperimentRegistry

    get_settings(refresh=True)
    X_train, y_train, X_test, y_test = _binary_data()
    report = run_comprehensive_comparison(
        TaskType.BINARY, X_train, y_train, X_test, y_test, n_qubits=4, seed=0,
        algorithms=[Algorithm.CLASSICAL_LOGISTIC_REGRESSION], record=True,
    )
    assert report.best_method == "classical_logistic_regression"
    reg = ExperimentRegistry(db_path=tmp_path / "registry.sqlite3")
    assert reg.count() >= 1
