"""Three-way quantum classifier comparison + classical baseline, on the
identical walk-forward folds of `SyntheticDirectionalDataset`.

This platform now has three architecturally distinct quantum classification
approaches:
  - VQC (algorithms.vqc_qml): hand-rolled ansatz, closed-form ridge readout,
    gradient-free outer loop.
  - PQK (algorithms.pqk.pqk_v2): kernel method, KTA-aligned feature map,
    Nystrom-approximated kernel ridge.
  - QNN (algorithms_extended.qiskit_ml_finance): true end-to-end gradient
    descent through the circuit via parameter-shift.

Running all three plus a classical baseline on the same data/folds is a
genuine comparison, not padding -- and per this platform's documented
no-overclaiming policy (see algorithms/vqc_qml/__init__.py), results are
reported honestly regardless of which method wins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.registry import APPLICATION_REGISTRY

from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from algorithms.vqc_qml.training import TrainingConfig, VQCTrainer
from algorithms.pqk.pqk_v2 import PQKConfig, PQKv2
from algorithms_extended.qiskit_ml_finance.classifier import ClassifierConfig, QNNClassifier
from classical.machine_learning.baselines import logistic_regression_baseline


@dataclass
class ThreeWayFoldResult:
    fold: int
    vqc_accuracy: float
    pqk_accuracy: float
    qnn_accuracy: float
    classical_accuracy: float
    best_method: str


def run_three_way_comparison(
    dataset: FinancialDataset,
    n_qubits: int = 4,
    n_splits: int = 3,
    seed: int = 0,
) -> list[ThreeWayFoldResult]:
    X, y = dataset.features(), dataset.targets()
    splits = dataset.split_walk_forward(n_splits=n_splits)

    results: list[ThreeWayFoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        vqc = VQCTrainer(VQCAnsatz(n_qubits=n_qubits, version=VQCVersion.V5_REUPLOAD), TrainingConfig()).fit(
            X_train, y_train
        )
        vqc_acc = vqc.accuracy(X_test, y_test)

        pqk = PQKv2(PQKConfig(n_qubits=n_qubits, seed=seed, kta_steps=20, nystrom_m=min(30, len(X_train)))).fit(
            X_train, y_train
        )
        pqk_acc = pqk.accuracy(X_test, y_test)

        qnn = QNNClassifier(ClassifierConfig(n_qubits=n_qubits, seed=seed)).fit(X_train, y_train)
        qnn_acc = qnn.score(X_test, y_test)

        classical = logistic_regression_baseline(X_train, y_train, X_test, y_test)

        accs = {
            "vqc": vqc_acc,
            "pqk": pqk_acc,
            "qnn": qnn_acc,
            "classical": classical.accuracy,
        }
        best_method = max(accs, key=accs.get)

        results.append(
            ThreeWayFoldResult(
                fold=fold,
                vqc_accuracy=vqc_acc,
                pqk_accuracy=pqk_acc,
                qnn_accuracy=qnn_acc,
                classical_accuracy=classical.accuracy,
                best_method=best_method,
            )
        )
    return results


@APPLICATION_REGISTRY.register("three_way_classification_comparison")
class ThreeWayClassificationComparisonApp:
    """Registered application entry point (see core.registry / core.cli)."""

    @staticmethod
    def run(**kwargs) -> list[ThreeWayFoldResult]:
        from applications.classification.synthetic_financial_dataset import SyntheticDirectionalDataset

        dataset = kwargs.pop("dataset", None) or SyntheticDirectionalDataset()
        return run_three_way_comparison(dataset, **kwargs)
