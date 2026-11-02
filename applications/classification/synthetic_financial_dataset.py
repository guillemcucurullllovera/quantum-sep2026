"""Synthetic financial time-series dataset for classifier development and
testing, until real market-data wiring lands (see hardware/ + data
integration work elsewhere in the platform). Sequential, not i.i.d — labels
depend on a rolling nonlinear function of recent synthetic "returns", so
walk-forward splitting is meaningful (unlike a plain iid toy dataset).
"""
from __future__ import annotations

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.schemas import DatasetDescriptor


class SyntheticDirectionalDataset(FinancialDataset):
    """Features: rolling momentum, rolling volatility, RSI-like bounded
    oscillator, volume-proxy z-score. Label: next-step direction (+1/-1),
    generated from a nonlinear (not linearly separable) combination of the
    features plus noise, so a linear classical baseline is a meaningful,
    non-trivial comparator rather than a strawman.
    """

    def __init__(self, n_samples: int = 600, n_features: int = 8, noise: float = 0.15, seed: int = 7) -> None:
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

        # nonlinear decision boundary: XOR-like interaction between momentum and vol,
        # so a plain linear classifier cannot trivially reach ~100% accuracy
        score = np.sign(X[:, 0]) * np.sign(X[:, 1] - np.median(X[:, 1])) + noise * rng.normal(0, 1, n_samples)
        y = np.where(score >= 0, 1, -1)

        self._X, self._y = X, y
        self._n_features = n_features

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_directional",
            n_samples=len(self._y),
            n_features=self._n_features,
            domain="directional_classification",
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y
