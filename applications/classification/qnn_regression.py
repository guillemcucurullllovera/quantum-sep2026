"""QNN-based financial regression application: predicts a continuous target
(forward realized volatility) via `algorithms_extended.qiskit_ml_finance
.QNNRegressor`, walk-forward validated against a classical ridge-regression
baseline on the identical folds.

Closes a real gap: `algorithms/vqc_qml` (v1-v5) is classification-only, and
so is every other quantum algorithm in this platform prior to this platform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.metrics import mse, rmse
from core.registry import APPLICATION_REGISTRY

from algorithms_extended.qiskit_ml_finance.regressor import QNNRegressor, RegressorConfig
from algorithms_extended.qiskit_ml_finance.classical_baseline import ridge_regression_baseline


@dataclass
class VolatilityRegressionFoldResult:
    fold: int
    qnn_mse: float
    qnn_rmse: float
    classical_mse: float
    classical_rmse: float
    qnn_beats_classical: bool
    final_training_loss: float


def run_volatility_regression(
    dataset: FinancialDataset,
    n_qubits: int = 4,
    n_splits: int = 3,
    seed: int = 0,
) -> list[VolatilityRegressionFoldResult]:
    X, y = dataset.features(), dataset.targets()
    splits = dataset.split_walk_forward(n_splits=n_splits)

    results: list[VolatilityRegressionFoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        qnn = QNNRegressor(RegressorConfig(n_qubits=n_qubits, seed=seed)).fit(X_train, y_train)
        preds = qnn.predict(X_test)
        qnn_mse, qnn_rmse = mse(y_test, preds), rmse(y_test, preds)

        classical = ridge_regression_baseline(X_train, y_train, X_test, y_test)

        results.append(
            VolatilityRegressionFoldResult(
                fold=fold,
                qnn_mse=qnn_mse,
                qnn_rmse=qnn_rmse,
                classical_mse=classical.mse,
                classical_rmse=classical.rmse,
                qnn_beats_classical=qnn_mse < classical.mse,
                final_training_loss=qnn.loss_history_[-1],
            )
        )
    return results


@APPLICATION_REGISTRY.register("qnn_volatility_regression")
class QNNVolatilityRegressionApp:
    """Registered application entry point (see core.registry / core.cli)."""

    @staticmethod
    def run(**kwargs) -> list[VolatilityRegressionFoldResult]:
        from applications.classification.synthetic_volatility_dataset import SyntheticVolatilityDataset

        dataset = kwargs.pop("dataset", None) or SyntheticVolatilityDataset()
        return run_volatility_regression(dataset, **kwargs)
