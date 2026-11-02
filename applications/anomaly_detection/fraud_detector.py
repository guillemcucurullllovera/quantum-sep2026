"""Financial transaction/trade anomaly-detection application, built on
`algorithms_extended.qrbf_ocsvm_anomaly`.

Honest by construction: the underlying algorithm's own THIRD_PARTY_NOTICES.md and
REPLICATION_REPORT.md document that qrbf-OCSVM loses decisively to a
classical RBF-OCSVM baseline on the paper's own benchmark. This application
does not hide that -- `FraudDetectionReport` always carries both the quantum
and classical results side by side, and `recommended_model` picks whichever
actually scored better on the data at hand rather than defaulting to either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from algorithms_extended.qrbf_ocsvm_anomaly.kernels import FEATURE_MAPS, compute_rdms_for_dataset, qrbf_kernel_matrix, scale_to_pi
from algorithms_extended.qrbf_ocsvm_anomaly.replication import f1_score_anomaly, ocsvm_predict_anomaly
from core.dataset_interfaces import FinancialDataset
from core.registry import APPLICATION_REGISTRY


@dataclass
class FraudDetectionReport:
    quantum_feature_map: str
    quantum_test_f1: float
    classical_test_f1: float
    recommended_model: str
    n_train: int
    n_test: int


def _train_test_split(X: np.ndarray, y: np.ndarray, seed: int, test_fraction: float = 0.3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def detect_anomalies(
    dataset: FinancialDataset, *, feature_map_name: str = "ECR",
    gamma: float = 0.1, nu: float = 0.1, seed: int = 0,
) -> FraudDetectionReport:
    """Run both the quantum qrbf-OCSVM and a classical RBF-OCSVM on the same
    train/test split of `dataset`, report both honestly."""
    X, y = dataset.features(), dataset.targets()
    Xtr, ytr, Xte, yte = _train_test_split(X, y, seed)

    # -- classical baseline (real sklearn OneClassSVM, matches the paper's own
    #    methodology exactly, per algorithms_extended.qrbf_ocsvm_anomaly.replication)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    classical_model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu).fit(Xtr_s)
    classical_f1 = f1_score_anomaly(yte, ocsvm_predict_anomaly(classical_model, Xte_s))

    # -- quantum path
    fm = FEATURE_MAPS[feature_map_name]
    Xtr_scaled, Xte_scaled = scale_to_pi(Xtr), scale_to_pi(Xte)
    rdms_tr = compute_rdms_for_dataset(Xtr_scaled, fm)
    rdms_te = compute_rdms_for_dataset(Xte_scaled, fm)
    K_train = qrbf_kernel_matrix(rdms_tr, rdms_tr, gamma=gamma)
    K_test = qrbf_kernel_matrix(rdms_te, rdms_tr, gamma=gamma)
    quantum_model = OneClassSVM(kernel="precomputed", nu=nu).fit(K_train)
    quantum_f1 = f1_score_anomaly(yte, ocsvm_predict_anomaly(quantum_model, K_test))

    return FraudDetectionReport(
        quantum_feature_map=feature_map_name,
        quantum_test_f1=quantum_f1,
        classical_test_f1=classical_f1,
        recommended_model="quantum" if quantum_f1 > classical_f1 else "classical",
        n_train=len(ytr), n_test=len(yte),
    )


@APPLICATION_REGISTRY.register("anomaly_detection_fraud")
class FraudDetectionApplication:
    """Registered application entry point."""

    def __init__(self, dataset: FinancialDataset, **kwargs) -> None:
        self.dataset = dataset
        self.kwargs = kwargs

    def run(self) -> FraudDetectionReport:
        return detect_anomalies(self.dataset, **self.kwargs)
