"""Multi-class regime classification application, built on the same VQC
trainer as directional classification (RidgeReadout is already one-vs-rest
multiclass) plus a random-forest classical baseline for comparison."""
from __future__ import annotations

from dataclasses import dataclass

from core.dataset_interfaces import FinancialDataset
from core.registry import APPLICATION_REGISTRY

from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from algorithms.vqc_qml.training import TrainingConfig, VQCTrainer
from classical.machine_learning.baselines import _HAS_SKLEARN, random_forest_baseline
from classical.machine_learning.baselines import logistic_regression_baseline


@dataclass
class RegimeDetectionResult:
    fold: int
    vqc_accuracy: float
    baseline_accuracy: float
    baseline_name: str


def run_regime_detection(
    dataset: FinancialDataset,
    n_qubits: int = 6,
    version: VQCVersion = VQCVersion.V5_REUPLOAD,
    n_splits: int = 3,
) -> list[RegimeDetectionResult]:
    X, y = dataset.features(), dataset.targets()
    splits = dataset.split_walk_forward(n_splits=n_splits)
    ansatz = VQCAnsatz(n_qubits=n_qubits, version=version)

    results: list[RegimeDetectionResult] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        trainer = VQCTrainer(ansatz, TrainingConfig()).fit(X_train, y_train)
        vqc_acc = trainer.accuracy(X_test, y_test)

        if _HAS_SKLEARN:
            baseline = random_forest_baseline(X_train, y_train, X_test, y_test)
        else:
            # multiclass fallback baseline: majority-class accuracy is not a fair
            # comparator, so restrict to sklearn-available environments for this
            # application (documented gap when sklearn is absent).
            baseline_name = "unavailable_without_sklearn"
            results.append(RegimeDetectionResult(fold, vqc_acc, float("nan"), baseline_name))
            continue

        results.append(RegimeDetectionResult(fold, vqc_acc, baseline.accuracy, baseline.name))
    return results


@APPLICATION_REGISTRY.register("vqc_regime_detection")
class VQCRegimeDetectionApp:
    @staticmethod
    def run(**kwargs) -> list[RegimeDetectionResult]:
        from applications.regime_detection.synthetic_regime_dataset import SyntheticRegimeDataset

        dataset = kwargs.pop("dataset", None) or SyntheticRegimeDataset()
        return run_regime_detection(dataset, **kwargs)
