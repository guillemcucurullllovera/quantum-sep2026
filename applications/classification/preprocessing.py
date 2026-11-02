"""Composable preprocessing step wiring `core.feature_engineering`
(fracdiff, purged CV) into the classification/regime/anomaly product layer.

Before this platform, `core.feature_engineering` existed but nothing under
`applications/` actually called it -- every synthetic dataset fed raw
features straight into an algorithm's own ad hoc scaling (e.g.
`algorithms.vqc_qml.encoding.AngleEncoder`'s min-max-to-[-pi,pi], which is
encoding-specific, not general financial-ML hygiene). `Preprocessor` here is
the general step that runs *before* any algorithm-specific encoding.

Leakage discipline: `Preprocessor.fit` computes standardization statistics
(mean/std) from training data only. `fracdiff` weights depend only on `d`
(not on the data), so applying it independently to a train slice and a test
slice is causally safe -- each fracdiff'd value at time t depends only on
values at or before t within the slice it's computed on. This does mean a
test slice loses its first `width-1` rows to the FFD warm-up window when
fracdiff is applied to it in isolation (see `transform`); that's a
conservative information loss, not a leakage bug, and is documented rather
than silently absorbed by recomputing fracdiff over the concatenated
train+test series (which would be leak-free too, but couples the
preprocessor's public API to slice adjacency in a way that's easy to get
wrong at the call site -- kept out of scope here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.errors import ValidationError
from core.feature_engineering.fracdiff import fracdiff
from core.feature_engineering.purged_cv import PurgedKFold


@dataclass
class PreprocessingConfig:
    fracdiff_d: float | None = None
    fracdiff_threshold: float = 1e-4
    standardize: bool = True


class Preprocessor:
    """fit/transform step: optional fractional differentiation per column,
    optional train-only z-score standardization. Stateless w.r.t. fracdiff
    (weights depend only on `d`); stateful w.r.t. standardization (mean/std
    captured at `fit` time)."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.cfg = config or PreprocessingConfig()
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._fitted = False

    def _fracdiff_columns(self, X: np.ndarray) -> np.ndarray:
        if self.cfg.fracdiff_d is None:
            return X
        out = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            series = pd.Series(X[:, j])
            fd = fracdiff(series, d=self.cfg.fracdiff_d, threshold=self.cfg.fracdiff_threshold)
            # FFD warm-up window is NaN for the first `width-1` rows; fill with
            # the column's own (post-warm-up) mean rather than dropping rows,
            # so every caller gets back an array of the same shape it put in --
            # documented, not hidden (see module docstring).
            fill_value = fd.dropna().mean() if fd.notna().any() else 0.0
            out[:, j] = fd.fillna(fill_value).to_numpy()
        return out

    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        X_train = np.asarray(X_train, dtype=float)
        X_fd = self._fracdiff_columns(X_train)
        if self.cfg.standardize:
            self._mu = X_fd.mean(axis=0)
            self._sigma = np.where(X_fd.std(axis=0) > 1e-12, X_fd.std(axis=0), 1.0)
        else:
            self._mu = np.zeros(X_fd.shape[1])
            self._sigma = np.ones(X_fd.shape[1])
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValidationError("Preprocessor.fit() must be called before transform()")
        X = np.asarray(X, dtype=float)
        X_fd = self._fracdiff_columns(X)
        return (X_fd - self._mu) / self._sigma

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


def purged_walk_forward_splits(
    n_samples: int, n_splits: int, embargo_td: int = 0, label_horizon: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged+embargoed k-fold splits over `n_samples` point-in-time rows,
    using `core.feature_engineering.purged_cv.PurgedKFold` under the hood.

    `label_horizon` models each row's label as depending on a forward-looking
    window of that many rows (e.g. a triple-barrier label's t1) -- rows whose
    label window overlaps a test fold are purged from that fold's training
    set, and `embargo_td` further rows immediately after each test fold are
    dropped. Use this instead of `FinancialDataset.split_walk_forward` when
    labels are computed over a forward window rather than at a single point
    in time (see `core.feature_engineering` package docstring).
    """
    index = pd.RangeIndex(n_samples)
    X = pd.DataFrame(index=index)
    t1 = pd.Series(index.to_numpy() + label_horizon, index=index)
    pkf = PurgedKFold(n_splits=n_splits, embargo_td=embargo_td)
    return list(pkf.split(X, t1=t1))


__all__ = ["PreprocessingConfig", "Preprocessor", "purged_walk_forward_splits"]
