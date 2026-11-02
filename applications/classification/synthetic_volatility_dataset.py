"""Synthetic continuous-target dataset for financial regression: predicts a
forward realized-volatility level from the same style of rolling features
used by `SyntheticDirectionalDataset`, but with a continuous (not binarized)
target -- closes the gap that nothing in the platform trained a VQC/QNN
circuit against a continuous financial target end-to-end (see
algorithms_extended.qiskit_ml_finance.regressor module docstring).
"""
from __future__ import annotations

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.schemas import DatasetDescriptor


class SyntheticVolatilityDataset(FinancialDataset):
    """Features: rolling momentum, current realized vol, RSI-like oscillator,
    volume-proxy. Target: forward realized volatility (continuous, >= 0),
    a smooth nonlinear function of the features plus noise -- so a linear
    classical regressor is a meaningful, non-trivial comparator.
    """

    def __init__(self, n_samples: int = 600, n_features: int = 8, noise: float = 0.1, seed: int = 13) -> None:
        rng = np.random.default_rng(seed)
        raw_returns = rng.normal(0, 0.01, n_samples + 30)
        price = np.cumprod(1 + raw_returns)

        momentum = (price[30:] - price[20:-10]) / price[20:-10]
        vol = np.array([raw_returns[i : i + 20].std() for i in range(n_samples)])
        rsi_like = np.tanh(momentum * 10)
        volume_proxy = rng.normal(0, 1, n_samples)

        feats = [momentum, vol, rsi_like, volume_proxy]
        while len(feats) < n_features:
            feats.append(rng.normal(0, 1, n_samples))
        X = np.stack(feats[:n_features], axis=1)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

        # forward realized vol: nonlinear (quadratic-in-momentum + linear-in-vol)
        # function of the features, scaled to a plausible annualized-vol-ish range
        raw_target = 0.15 + 0.05 * X[:, 0] ** 2 + 0.08 * np.abs(X[:, 1]) + noise * rng.normal(0, 1, n_samples)
        y = np.clip(raw_target, 0.01, None)

        self._X, self._y = X, y
        self._n_features = n_features

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_volatility",
            n_samples=len(self._y),
            n_features=self._n_features,
            domain="volatility_regression",
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y
