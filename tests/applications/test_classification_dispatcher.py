"""Real smoke tests for the unified dispatcher
(`applications.classification.dispatcher.dispatch`): every one of the ~17
concrete algorithms it can route to is exercised through the SAME
`dispatch(spec, X_train, y_train, X_test, y_test)` call, on tiny (fast)
data, and checked for a well-formed `DispatchResult` -- proving the
dispatcher's single entry point actually reaches each underlying
implementation (VQC v1/v4/v5, PQK Gen1/Gen2, QuantumSVC, FinancialQNN,
PegasosQSVC, EstimatorQNN, covariant-kernel SVM, RegimeHMM, OCSVM (classical
+ qrbf), and 4 classical baselines), not just that the module imports.
"""
from __future__ import annotations

import numpy as np
import pytest

from applications.classification.dispatcher import dispatch, supported_task_types
from applications.classification.task_spec import Algorithm, ClassificationTaskSpec, TaskType


def _binary_data(n=16, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    n_test = n // 4
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


def _multiclass_data(n_per_class=8, n_features=4, seed=1):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in range(3):
        Xs.append(rng.normal(loc=c * 3.0, scale=0.5, size=(n_per_class, n_features)))
        ys.append(np.full(n_per_class, c))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_test = len(y) // 4
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


def _anomaly_data(n_normal=14, n_anomaly=5, n_features=6, seed=2):
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(n_normal, n_features))
    anomaly = rng.normal(size=(n_anomaly, n_features)) + 4.0
    X = np.vstack([normal, anomaly])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)])
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_test = len(y) // 4
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


def _tiny_binary_data(n_train=4, n_test=2, n_features=2, seed=5):
    """Balanced-by-construction tiny binary dataset (2 of each class in
    train, 1 of each in test) for algorithms too expensive to run on the
    shared 16-sample fixture (e.g. covariant-kernel training/inference is
    O(n^2) real circuit evaluations)."""
    rng = np.random.default_rng(seed)
    X_train = np.concatenate(
        [rng.normal(loc=-1.5, size=(n_train // 2, n_features)), rng.normal(loc=1.5, size=(n_train // 2, n_features))]
    )
    y_train = np.concatenate([np.zeros(n_train // 2, dtype=int), np.ones(n_train // 2, dtype=int)])
    X_test = np.concatenate(
        [rng.normal(loc=-1.5, size=(n_test // 2, n_features)), rng.normal(loc=1.5, size=(n_test // 2, n_features))]
    )
    y_test = np.concatenate([np.zeros(n_test // 2, dtype=int), np.ones(n_test // 2, dtype=int)])
    return X_train, y_train, X_test, y_test


def _regression_data(n=16, n_features=4, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.2 + rng.normal(scale=0.05, size=n)
    n_test = n // 4
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


_DATA_BY_TASK = {
    TaskType.BINARY: _binary_data(),
    TaskType.MULTICLASS: _multiclass_data(),
    TaskType.REGIME: _multiclass_data(),
    TaskType.ANOMALY: _anomaly_data(),
    TaskType.REGRESSION: _regression_data(),
}

# Every Algorithm, exercised at least once, each through dispatch() directly.
# n_qubits and algorithm_params are kept small purely for test speed (see
# each module's own docstring for what a real-scale run looks like).
_CASES = [
    (Algorithm.VQC_V1_LINEAR, TaskType.BINARY, 4, {}),
    (Algorithm.VQC_V4_RING, TaskType.BINARY, 4, {}),
    (Algorithm.VQC_V5_REUPLOAD, TaskType.MULTICLASS, 4, {}),
    (Algorithm.PQK_GEN1_FIDELITY, TaskType.BINARY, 4, {}),
    (Algorithm.PQK_GEN2, TaskType.BINARY, 3, {"kta_steps": 5, "nystrom_m": 8}),
    (Algorithm.QUANTUM_SVC, TaskType.BINARY, 4, {}),
    (Algorithm.FINANCIAL_QNN, TaskType.BINARY, 3, {"epochs": 8}),
    (Algorithm.PEGASOS_QSVC, TaskType.BINARY, 3, {"num_steps": 15}),
    (Algorithm.ESTIMATOR_QNN, TaskType.BINARY, 3, {"maxiter": 8}),
    (Algorithm.COVARIANT_KERNEL_SVM, TaskType.BINARY, 2, {"maxiter": 2, "shots": 16}),
    (Algorithm.REGIME_HMM, TaskType.REGIME, 4, {}),
    (Algorithm.OCSVM_ANOMALY, TaskType.ANOMALY, 4, {}),
    (Algorithm.QRBF_OCSVM, TaskType.ANOMALY, 4, {}),
    (Algorithm.CLASSICAL_LOGISTIC_REGRESSION, TaskType.BINARY, 4, {}),
    (Algorithm.CLASSICAL_RANDOM_FOREST, TaskType.MULTICLASS, 4, {}),
    (Algorithm.CLASSICAL_SVM_RBF, TaskType.BINARY, 4, {}),
    (Algorithm.CLASSICAL_RIDGE_REGRESSION, TaskType.REGRESSION, 4, {}),
]

assert {c[0] for c in _CASES} == set(Algorithm), "every Algorithm must have a dispatcher smoke test"


@pytest.mark.parametrize("algorithm,task_type,n_qubits,params", _CASES, ids=[c[0].value for c in _CASES])
def test_dispatch_routes_and_runs(algorithm, task_type, n_qubits, params):
    X_train, y_train, X_test, y_test = _DATA_BY_TASK[task_type]
    if algorithm == Algorithm.COVARIANT_KERNEL_SVM:
        # covariant-kernel training/inference is O(n_train^2 + n_train*n_test)
        # real AerSimulator circuit evaluations -- use a dedicated tiny
        # (class-balanced) fixture instead of the shared 16-sample one.
        X_train, y_train, X_test, y_test = _tiny_binary_data()
    spec = ClassificationTaskSpec(
        task_type=task_type, algorithm=algorithm, n_qubits=n_qubits, seed=0, algorithm_params=params
    )
    result = dispatch(spec, X_train, y_train, X_test, y_test)

    assert result.algorithm == algorithm
    assert result.task_type == task_type
    assert result.n_train == len(y_train)
    assert result.n_test == len(y_test)
    assert len(result.predictions) == len(y_test)
    assert result.fit_time_s >= 0.0
    assert task_type in supported_task_types(algorithm)

    if task_type in (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGIME, TaskType.ANOMALY):
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
    if task_type == TaskType.REGRESSION:
        assert result.metrics["mse"] >= 0.0
        assert result.metrics["rmse"] >= 0.0


def test_unsupported_task_type_raises_clear_error():
    from applications.classification.errors import UnsupportedTaskError

    X_train, y_train, X_test, y_test = _regression_data()
    spec = ClassificationTaskSpec(task_type=TaskType.REGRESSION, algorithm=Algorithm.REGIME_HMM, n_qubits=4)
    with pytest.raises(UnsupportedTaskError):
        dispatch(spec, X_train, y_train, X_test, y_test)


def test_all_algorithms_enumerated_in_supported_task_types():
    """Every Algorithm resolves to a non-empty supported-task-type set --
    catches a registry/spec drift (an Algorithm added to the enum but never
    wired into _SUPPORTED_TASKS / _ADAPTERS)."""
    for algo in Algorithm:
        assert supported_task_types(algo), f"{algo} has no supported task types"
