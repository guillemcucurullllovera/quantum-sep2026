"""Composed use-cases for FinancialQNN. This module lives here rather than
under applications/ because this  workstream's scope explicitly
excludes touching applications/ (owned by other agents this platform) --
functionally these are application-layer demos, just physically located
alongside the algorithm they showcase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.schemas import DatasetDescriptor
from core.metrics import mse

from applications.regime_detection.synthetic_regime_dataset import SyntheticRegimeDataset
from classical.machine_learning.baselines import logistic_regression_baseline

from algorithms_extended.qiskit_ml_finance.regressor import QNNRegressor, RegressorConfig
from algorithms_extended.qiskit_ml_finance.classifier import QNNClassifier, ClassifierConfig
from algorithms_extended.qiskit_ml_finance.classical_baseline import ridge_regression_baseline


class SyntheticVolatilityLevelDataset(FinancialDataset):
    """A continuous realized-volatility-level target (not the classification
    targets used elsewhere in the platform) -- built here because no
    regression-shaped financial dataset existed before this workstream.
    """

    def __init__(self, n_samples: int = 120, n_features: int = 4, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        # underlying vol regime drives both features and the continuous target
        regime = rng.uniform(0.05, 0.45, n_samples)  # true realized vol level
        returns_window = rng.normal(0, 1, (n_samples, 20)) * regime[:, None]
        momentum = returns_window.mean(axis=1)
        realized = returns_window.std(axis=1)
        skew_proxy = ((returns_window - returns_window.mean(axis=1, keepdims=True)) ** 3).mean(axis=1)
        extra = rng.normal(0, 1, (n_samples, max(0, n_features - 3)))
        X = np.concatenate([momentum[:, None], realized[:, None], skew_proxy[:, None], extra], axis=1)
        self._X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        self._y = regime  # continuous target: the true vol level that generated the window
        self._n_features = n_features

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_volatility_level", n_samples=len(self._y), n_features=self._n_features, domain="volatility"
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y


@dataclass
class VolatilityRegressionReport:
    qnn_mse: float
    classical_mse: float
    n_train: int
    n_test: int


def run_volatility_regression_demo(seed: int = 42) -> VolatilityRegressionReport:
    ds = SyntheticVolatilityLevelDataset(n_samples=120, n_features=4, seed=seed)
    splits = ds.split_walk_forward(n_splits=1, min_train_fraction=0.7)
    train_idx, test_idx = splits[0]
    X, y = ds.features(), ds.targets()

    qnn = QNNRegressor(RegressorConfig(n_qubits=4, epochs=50, seed=seed)).fit(X[train_idx], y[train_idx])
    qnn_preds = qnn.predict(X[test_idx])

    classical = ridge_regression_baseline(X[train_idx], y[train_idx], X[test_idx], y[test_idx])

    return VolatilityRegressionReport(
        qnn_mse=mse(y[test_idx], qnn_preds),
        classical_mse=classical.mse,
        n_train=len(train_idx),
        n_test=len(test_idx),
    )


@dataclass
class RegimeClassificationReport:
    qnn_accuracy: float
    classical_accuracy: float
    n_train: int
    n_test: int


def run_regime_classification_demo(seed: int = 42) -> RegimeClassificationReport:
    """Binary subset (trend_up vs ranging) of the 3-class regime dataset --
    QNNClassifier is binary-only currently (see classifier.py); documented
    here rather than silently generalized to multiclass."""
    ds = SyntheticRegimeDataset(n_per_regime=60, n_features=4, seed=seed)
    X, y_full = ds.features(), ds.targets()
    mask = y_full != 1  # drop trend_down (label 1), keep trend_up(0)/ranging(2)
    X, y = X[mask], y_full[mask]
    n = len(y)
    n_train = int(n * 0.7)
    idx = np.arange(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    qnn = QNNClassifier(ClassifierConfig(n_qubits=4, epochs=60, seed=seed)).fit(X[train_idx], y[train_idx])
    qnn_acc = qnn.score(X[test_idx], y[test_idx])

    classical = logistic_regression_baseline(X[train_idx], y[train_idx], X[test_idx], y[test_idx])

    return RegimeClassificationReport(
        qnn_accuracy=qnn_acc, classical_accuracy=classical.accuracy, n_train=n_train, n_test=n - n_train
    )
