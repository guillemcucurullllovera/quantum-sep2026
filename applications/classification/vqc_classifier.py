"""VQC-based directional classification application: composes
algorithms.vqc_qml with a classical baseline and walk-forward validation
into one reusable, registered application.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.metrics import accuracy
from core.registry import APPLICATION_REGISTRY

from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from algorithms.vqc_qml.calibration import PlattCalibrator
from algorithms.vqc_qml.training import TrainingConfig, VQCTrainer
from classical.machine_learning.baselines import logistic_regression_baseline


@dataclass
class DirectionalClassificationResult:
    fold: int
    vqc_accuracy: float
    baseline_accuracy: float
    vqc_calibration_error: float
    quantum_beats_classical: bool


def run_directional_classification(
    dataset: FinancialDataset,
    n_qubits: int = 6,
    version: VQCVersion = VQCVersion.V5_REUPLOAD,
    n_splits: int = 3,
    training_config: TrainingConfig | None = None,
) -> list[DirectionalClassificationResult]:
    """Walk-forward evaluation of the VQC classifier against a classical
    logistic-regression baseline on the same folds. Positive results are
    presented as benchmarked research findings, not an advantage claim —
    per the platform's own classical-simulability audit for this circuit
    family (see algorithms/vqc_qml/__init__.py docstring).
    """
    X, y = dataset.features(), dataset.targets()
    splits = dataset.split_walk_forward(n_splits=n_splits)
    ansatz = VQCAnsatz(n_qubits=n_qubits, version=version)
    cfg = training_config or TrainingConfig()

    results: list[DirectionalClassificationResult] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        trainer = VQCTrainer(ansatz, cfg).fit(X_train, y_train)
        vqc_acc = trainer.accuracy(X_test, y_test)

        # calibrate on the positive-class decision score
        train_scores = trainer.decision_scores(X_train)[:, -1]
        y_train_binary = (y_train == np.unique(y_train)[-1]).astype(float)
        calibrator = PlattCalibrator().fit(train_scores, y_train_binary)

        test_scores = trainer.decision_scores(X_test)[:, -1]
        y_test_binary = (y_test == np.unique(y_test)[-1]).astype(float)
        cal_error = calibrator.evaluate_calibration(test_scores, y_test_binary)

        baseline = logistic_regression_baseline(X_train, y_train, X_test, y_test)

        results.append(
            DirectionalClassificationResult(
                fold=fold,
                vqc_accuracy=vqc_acc,
                baseline_accuracy=baseline.accuracy,
                vqc_calibration_error=cal_error,
                quantum_beats_classical=vqc_acc > baseline.accuracy,
            )
        )
    return results


@APPLICATION_REGISTRY.register("vqc_directional_classification")
class VQCDirectionalClassificationApp:
    """Registered application entry point (see core.registry / core.cli)."""

    @staticmethod
    def run(**kwargs) -> list[DirectionalClassificationResult]:
        from applications.classification.synthetic_financial_dataset import SyntheticDirectionalDataset

        dataset = kwargs.pop("dataset", None) or SyntheticDirectionalDataset()
        return run_directional_classification(dataset, **kwargs)
