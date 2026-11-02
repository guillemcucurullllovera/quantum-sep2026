"""Composable preprocessing pipeline for the time-series application layer,
wiring core.feature_engineering.fracdiff into a real, testable pipeline
instead of leaving stationarity transforms as a manual, ad hoc step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from applications.time_series.domain import FeatureSpec
from applications.time_series.errors import InsufficientDataError
from core.dataset_interfaces import FinancialDataset
from core.feature_engineering.fracdiff import fracdiff
from core.schemas import DatasetDescriptor


class ArrayDataset(FinancialDataset):
    """Wrap already-built (X, y) arrays as a FinancialDataset so downstream
    code (walk-forward splitting, method adapters) shares one contract
    regardless of whether the arrays came straight from a synthetic
    generator or passed through fracdiff/horizon preprocessing first.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, name: str, domain: str = "generic") -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X) != len(y):
            raise InsufficientDataError(f"X has {len(X)} rows, y has {len(y)}")
        self._X = X
        self._y = y
        self._name = name
        self._domain = domain

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name=self._name,
            n_samples=len(self._y),
            n_features=self._X.shape[1] if self._X.ndim == 2 else 0,
            domain=self._domain,
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y


def apply_fracdiff(X: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Apply Fixed-Width-Window fractional differentiation column-wise to a
    feature matrix, trimming the warm-up rows where the FFD window is
    incomplete. Real stationarity-inducing preprocessing
    (core.feature_engineering.fracdiff), preserving memory rather than
    destroying it the way integer differencing would -- useful specifically
    for realized-volatility series, which are persistent (long-memory) but
    non-stationary in level.
    """
    if not spec.use_fracdiff:
        return np.asarray(X, dtype=float)

    X = np.asarray(X, dtype=float)
    df = pd.DataFrame(X)
    fd_cols = []
    max_valid_start = 0
    for col in df.columns:
        fd = fracdiff(df[col], d=spec.fracdiff_d, threshold=spec.fracdiff_threshold)
        fd_cols.append(fd)
        first_valid = fd.first_valid_index()
        if first_valid is not None:
            max_valid_start = max(max_valid_start, first_valid)

    out = pd.concat(fd_cols, axis=1).iloc[max_valid_start:]
    if len(out) < 10:
        raise InsufficientDataError(
            f"fracdiff(d={spec.fracdiff_d}) leaves only {len(out)} usable rows "
            f"from {len(X)} input rows -- increase the series length or lower "
            f"fracdiff_threshold."
        )
    return out.to_numpy(dtype=float)


def apply_horizon(X: np.ndarray, y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift the target `horizon` steps ahead of its matching feature row
    (horizon=1 reproduces the original next-step alignment: y_h[t] = y[t]).
    Never looks behind: X[t] is assumed (by the upstream dataset builder) to
    already only encode information available at/before t, and
    y_h[t] = y[t + horizon - 1] only ever reaches forward, never backward,
    into the target series.
    """
    if horizon <= 1:
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    if n <= horizon:
        raise InsufficientDataError(f"Series of length {n} is too short for horizon={horizon}")
    X_h = np.asarray(X, dtype=float)[: n - horizon + 1]
    y_h = np.asarray(y, dtype=float)[horizon - 1 :]
    return X_h, y_h


def build_pipeline(X: np.ndarray, y: np.ndarray, spec: FeatureSpec, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Full feature pipeline: fracdiff (optional) -> horizon alignment.
    Order matters -- fracdiff trims warm-up rows from the front of the
    series, so horizon shifting happens afterward, against the
    already-trimmed length.
    """
    X_proc = apply_fracdiff(X, spec)
    y_proc = np.asarray(y, dtype=float)
    if spec.use_fracdiff:
        y_proc = y_proc[-len(X_proc) :]
    return apply_horizon(X_proc, y_proc, horizon)


__all__ = ["ArrayDataset", "apply_fracdiff", "apply_horizon", "build_pipeline"]
