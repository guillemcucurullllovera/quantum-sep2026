"""General-purpose anomaly-scoring application: the fraud-detection
benchmark (`fraud_detector.py`) generalized to ANY finance anomaly context
-- trade/position anomalies (`synthetic_trade_anomalies
.SyntheticTradePositionDataset`), or any other `FinancialDataset` whose
`targets()` are a binary {0=normal, 1=anomaly} array -- routed through the
SAME unified dispatcher used by classification and regime detection
(`applications.classification.dispatcher`), rather than a bespoke
quantum/classical wiring per anomaly context.

`fraud_detector.detect_anomalies` is untouched (kept for exact backward
compatibility and because it documents the specific arXiv:2411.16970
replication lineage); this module is the product-layer generalization that
requirement #7 this platform's design asks for: reuse the underlying
`qrbf_ocsvm_anomaly` algorithm and `OCSVMAnomalyDetector` classical
baseline through the dispatcher, for contexts beyond fraud.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applications.classification.dispatcher import dispatch
from applications.classification.result import DispatchResult
from applications.classification.task_spec import Algorithm, ClassificationTaskSpec, TaskType
from core.dataset_interfaces import FinancialDataset


@dataclass
class GeneralAnomalyReport:
    dataset_name: str
    quantum_result: DispatchResult
    classical_result: DispatchResult
    recommended_model: str
    n_train: int
    n_test: int

    @property
    def quantum_f1(self) -> float:
        return self.quantum_result.metric("f1")

    @property
    def classical_f1(self) -> float:
        return self.classical_result.metric("f1")


def _train_test_split(X: np.ndarray, y: np.ndarray, seed: int, test_fraction: float = 0.3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def detect_general_anomalies(
    dataset: FinancialDataset,
    *,
    feature_map_name: str = "ECR",
    gamma: float = 0.1,
    nu: float = 0.1,
    seed: int = 0,
    test_fraction: float = 0.3,
) -> GeneralAnomalyReport:
    """Run both `Algorithm.QRBF_OCSVM` (quantum, projected-kernel OneClassSVM)
    and `Algorithm.OCSVM_ANOMALY` (classical, RBF-kernel OneClassSVM) on the
    same split of ANY `FinancialDataset` with binary anomaly targets,
    through `dispatcher.dispatch` -- one interface for fraud transactions,
    trade/position anomalies, or any future anomaly context, instead of a
    bespoke pipeline per dataset shape.

    Honest by construction, same as `fraud_detector.detect_anomalies`:
    `recommended_model` picks whichever actually scored higher F1 on this
    run's data, and both results are always reported side by side. The
    underlying quantum algorithm's own THIRD_PARTY_NOTICES.md documents a replication
    failure vs. the classical baseline on the paper's original benchmark
    (see `algorithms_extended.qrbf_ocsvm_anomaly`) -- nothing here assumes
    that changes for a different dataset.
    """
    X, y = dataset.features(), dataset.targets()
    X_train, y_train, X_test, y_test = _train_test_split(X, y, seed, test_fraction)

    quantum_spec = ClassificationTaskSpec(
        task_type=TaskType.ANOMALY, algorithm=Algorithm.QRBF_OCSVM, n_qubits=10, seed=seed,
        algorithm_params={"feature_map": feature_map_name, "gamma": gamma, "nu": nu},
    )
    classical_spec = ClassificationTaskSpec(
        task_type=TaskType.ANOMALY, algorithm=Algorithm.OCSVM_ANOMALY, n_qubits=10, seed=seed,
        algorithm_params={"gamma": gamma, "nu": nu},
    )

    quantum_result = dispatch(quantum_spec, X_train, y_train, X_test, y_test)
    classical_result = dispatch(classical_spec, X_train, y_train, X_test, y_test)

    recommended = "quantum" if quantum_result.metric("f1") > classical_result.metric("f1") else "classical"
    return GeneralAnomalyReport(
        dataset_name=dataset.descriptor.name,
        quantum_result=quantum_result,
        classical_result=classical_result,
        recommended_model=recommended,
        n_train=len(y_train),
        n_test=len(y_test),
    )


__all__ = ["GeneralAnomalyReport", "detect_general_anomalies"]
