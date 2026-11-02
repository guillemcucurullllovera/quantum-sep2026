"""Synthetic 3-regime dataset: trending-up, trending-down, ranging, each with
distinguishable drift/volatility statistics, for regime-classification
development and testing.
"""
from __future__ import annotations

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.schemas import DatasetDescriptor

REGIMES = ("trend_up", "trend_down", "ranging")


class SyntheticRegimeDataset(FinancialDataset):
    def __init__(self, n_per_regime: int = 200, n_features: int = 6, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        drift_vol = {"trend_up": (0.004, 0.008), "trend_down": (-0.004, 0.008), "ranging": (0.0, 0.003)}

        X_parts, y_parts = [], []
        for label_idx, regime in enumerate(REGIMES):
            drift, vol = drift_vol[regime]
            returns = rng.normal(drift, vol, (n_per_regime, 20))
            momentum = returns.mean(axis=1)
            realized_vol = returns.std(axis=1)
            skew_proxy = ((returns - returns.mean(axis=1, keepdims=True)) ** 3).mean(axis=1)
            extra = rng.normal(0, 1, (n_per_regime, max(0, n_features - 3)))
            feats = np.concatenate([momentum[:, None], realized_vol[:, None], skew_proxy[:, None], extra], axis=1)
            X_parts.append(feats[:, :n_features])
            y_parts.append(np.full(n_per_regime, label_idx))

        self._X = np.concatenate(X_parts, axis=0)
        self._X = (self._X - self._X.mean(axis=0)) / (self._X.std(axis=0) + 1e-9)
        self._y = np.concatenate(y_parts, axis=0)
        self._n_features = n_features

        # interleave so walk-forward folds see all regimes, not one block each
        order = np.argsort(rng.permutation(len(self._y)) % 37)  # deterministic pseudo-interleave
        self._X, self._y = self._X[order], self._y[order]

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_regime", n_samples=len(self._y), n_features=self._n_features, domain="regime_detection"
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y
