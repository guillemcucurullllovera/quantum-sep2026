"""Asymmetrica-original integration layer wiring the real upstream
qiskit-machine-learning classes (PegasosQSVC, VQR, FidelityQuantumKernel)
into this platform's existing comparators and datasets.

This is a "material extension": not just vendoring upstream code, but
making it interoperate with the platform's existing work and producing
real, honest comparative results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import Statevector

from algorithms_extended.qiskit_ml_asymmetrica.classifiers.pegasos_qsvc import PegasosQSVC
from algorithms_extended.qiskit_ml_asymmetrica.kernels.fidelity_quantum_kernel import FidelityQuantumKernel
from algorithms_extended.qiskit_ml_asymmetrica.regressors.vqr import VQR
from algorithms_extended.qiskit_ml_asymmetrica.optimizers.spsa import SPSA
from algorithms_extended.qml_financial_classifiers.qsvc_pipeline import QuantumSVC
from algorithms.pqk.kernels_gen1 import fidelity_kernel_element
from applications.classification.synthetic_financial_dataset import SyntheticDirectionalDataset
from applications.classification.synthetic_volatility_dataset import SyntheticVolatilityDataset
from core.metrics import accuracy, mse, rmse
from core.seeds import seed_everything


@dataclass
class ClassifierComparisonResult:
    pegasos_qsvc_accuracy: float
    platform_quantumsvc_accuracy: float
    pqk_fidelity_accuracy: float


@dataclass
class RegressorComparisonResult:
    upstream_vqr_mse: float
    platform_qnn_regressor_mse: float


def kernel_parity_check(n_samples: int = 6, n_features: int = 2, seed: int = 0) -> dict:
    """Real parity test: the upstream FidelityQuantumKernel's Gram matrix
    (computed via a real Qiskit circuit + state_fidelities.ComputeUncompute
    primitive) vs. a direct |<psi(x)|psi(y)>|^2 statevector computation on
    the identical ZZFeatureMap circuit, using this platform's own
    `fidelity_kernel_element` (algorithms.pqk.kernels_gen1) for the second
    half of the calculation. Two independent computational paths to the
    same quantity -- if they don't agree to numerical precision, one of
    them has a real bug.
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, (n_samples, n_features))
    fm = ZZFeatureMap(n_features, reps=1)

    upstream_kernel = FidelityQuantumKernel(feature_map=fm)
    K_upstream = upstream_kernel.evaluate(X)

    states = [Statevector(fm.assign_parameters(x)).data for x in X]
    K_platform = np.array([[fidelity_kernel_element(si, sj) for sj in states] for si in states])

    max_abs_diff = float(np.max(np.abs(K_upstream - K_platform)))
    return {
        "max_abs_diff": max_abs_diff,
        "agrees_within_1e-6": max_abs_diff < 1e-6,
        "K_upstream_shape": K_upstream.shape,
        "K_platform_shape": K_platform.shape,
    }


def classifier_comparison(seed: int = 0) -> ClassifierComparisonResult:
    """PegasosQSVC (upstream) vs. QuantumSVC (algorithms_extended/
    qml_financial_classifiers) vs. this platform's PQKv2 fidelity classifier,
    all on the identical synthetic directional dataset and identical
    train/test split. Honest three-way comparison, no assumed winner.
    """
    seed_everything(seed)
    ds = SyntheticDirectionalDataset(n_samples=240, n_features=4, seed=seed)
    X, y = ds.features(), ds.targets()
    y_pm1 = np.where(y > 0, 1, -1)
    n_train = int(0.7 * len(y))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_pm1[:n_train], y_pm1[n_train:]

    fm = ZZFeatureMap(X.shape[1], reps=1)

    pegasos = PegasosQSVC(quantum_kernel=FidelityQuantumKernel(feature_map=fm), C=1.0, num_steps=100, seed=seed)
    pegasos.fit(X_train, y_train)
    pegasos_acc = accuracy(y_test, pegasos.predict(X_test))

    # QuantumSVC computes its own Gram matrix internally from raw
    # features via its own feature_map string attribute -- pass raw X, not
    # a precomputed kernel matrix.
    qsvc = QuantumSVC(C=1.0)
    qsvc.fit(X_train, y_train)
    qsvc_acc = qsvc.score(X_test, y_test)

    from applications.classification.kernel_classifier import _fit_pqk

    pqk_model, _ = _fit_pqk(X_train, y_train, seed=seed)
    pqk_acc = pqk_model.accuracy(X_test, y_test)

    return ClassifierComparisonResult(
        pegasos_qsvc_accuracy=pegasos_acc,
        platform_quantumsvc_accuracy=qsvc_acc,
        pqk_fidelity_accuracy=pqk_acc,
    )


def regressor_comparison(seed: int = 0) -> RegressorComparisonResult:
    """Upstream VQR vs. QNNRegressor on the identical synthetic
    volatility-regression dataset.
    """
    seed_everything(seed)
    ds = SyntheticVolatilityDataset(n_samples=200, n_features=4, seed=seed)
    X, y = ds.features(), ds.targets()
    n_train = int(0.7 * len(y))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    fm = ZZFeatureMap(X.shape[1], reps=1)
    ansatz = RealAmplitudes(X.shape[1], reps=1)
    vqr = VQR(feature_map=fm, ansatz=ansatz, optimizer=SPSA(maxiter=60))
    vqr.fit(X_train, y_train)
    vqr_preds = np.asarray(vqr.predict(X_test)).reshape(-1)
    vqr_mse = mse(y_test, vqr_preds)

    from applications.classification.qnn_regression import QNNRegressor, RegressorConfig

    qnn = QNNRegressor(RegressorConfig(n_qubits=X.shape[1], seed=seed)).fit(X_train, y_train)
    qnn_preds = qnn.predict(X_test)
    qnn_mse = mse(y_test, qnn_preds)

    return RegressorComparisonResult(
        upstream_vqr_mse=vqr_mse,
        platform_qnn_regressor_mse=qnn_mse,
    )
