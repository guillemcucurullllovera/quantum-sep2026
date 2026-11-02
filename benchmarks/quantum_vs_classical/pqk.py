"""PQK vs classical RBF kernel, on the same synthetic classification set."""
from __future__ import annotations

import numpy as np

from algorithms.pqk.feature_maps import N_QUBITS
from algorithms.pqk.kernels import projected_gram
from algorithms.pqk.kernels_gen1 import SimpleSVM
from classical.kernels.baselines import train_classical_svm
from benchmarks.runner import MethodOutput, run_benchmark

_FEATURE_MAP_NAME = "AngleCNOT"  # algorithms.pqk.feature_maps.FEATURE_MAPS key


def _synthetic_classification(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, N_QUBITS))  # AngleCNOTMap.encode indexes features[0..N_QUBITS-1]
    y01 = (np.sin(3 * X[:, 0]) + X[:, 1] ** 2 > 0.3).astype(float)  # {0, 1}, matches train_classical_svm's contract
    return X, y01


def pqk_vs_rbf_classification(n_samples: int = 40, seed: int = 0):
    X, y01 = _synthetic_classification(n_samples, seed)
    split = int(n_samples * 0.7)
    X_train, y_train, X_test, y_test = X[:split], y01[:split], X[split:], y01[split:]

    def run_pqk() -> MethodOutput:
        y_pm = 2 * y_train - 1  # SimpleSVM expects {-1, +1}, same convention train_classical_svm uses internally
        y_test_pm = 2 * y_test - 1
        K_train = projected_gram(X_train, _FEATURE_MAP_NAME, X_train)
        K_test = projected_gram(X_test, _FEATURE_MAP_NAME, X_train)
        svm = SimpleSVM()
        svm.fit(K_train, y_pm)
        return MethodOutput(value=svm.accuracy(K_test, y_test_pm))

    def run_rbf() -> MethodOutput:
        result = train_classical_svm(X_train, y_train, X_test, y_test, kernel="rbf")
        return MethodOutput(value=result["val_accuracy"])

    return run_benchmark(
        method_a=run_pqk,
        method_b=run_rbf,
        method_a_name="pqk_angle_cnot",
        method_b_name="rbf_svm",
        metric_name="test_accuracy",
        seed=seed,
        context={"n_samples": n_samples},
    )
