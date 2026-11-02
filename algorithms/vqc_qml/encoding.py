"""Feature scaling/encoding utilities for VQC pipelines.

Getting this wrong (unbounded or unnormalized inputs into angle encoding) is
one of the most common real-world VQC bugs — angles wrap mod 2*pi, so an
un-scaled feature silently aliases to the wrong rotation. This module makes
the scaling an explicit, fitted, reusable step rather than inline math
scattered through training code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.errors import ValidationError


@dataclass
class AngleEncoder:
    """Maps arbitrary real-valued features to [-pi, pi] via a fitted min-max
    scale, so RY(angle) rotations use the full, non-aliased range.
    """

    feature_min_: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    feature_max_: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def fit(self, X: np.ndarray) -> "AngleEncoder":
        X = np.asarray(X, dtype=float)
        self.feature_min_ = X.min(axis=0)
        self.feature_max_ = X.max(axis=0)
        # guard against zero-range features (constant column) -> avoid div by zero
        flat_mask = (self.feature_max_ - self.feature_min_) < 1e-12
        self.feature_max_ = np.where(flat_mask, self.feature_min_ + 1.0, self.feature_max_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.feature_min_ is None:
            raise ValidationError("AngleEncoder.fit() must be called before transform()")
        X = np.asarray(X, dtype=float)
        scaled_01 = (X - self.feature_min_) / (self.feature_max_ - self.feature_min_)
        scaled_01 = np.clip(scaled_01, 0.0, 1.0)  # out-of-training-range values clip, don't alias
        return scaled_01 * 2 * np.pi - np.pi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def truncate_or_pad(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """Match feature dimensionality to n_qubits: truncate extra features,
    zero-pad if there are fewer features than qubits (explicit, not silent —
    the ansatz itself also zero-pads, but doing it here makes the applied
    feature width visible to the caller for logging/provenance)."""
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    if n_features == n_qubits:
        return X
    if n_features > n_qubits:
        return X[:, :n_qubits]
    pad = np.zeros((X.shape[0], n_qubits - n_features))
    return np.concatenate([X, pad], axis=1)


def amplitude_encode(x: np.ndarray) -> np.ndarray:
    """Normalize a feature vector to a valid quantum statevector amplitude
    encoding: pad to the next power of 2 and L2-normalize.

    Provided as an alternative to angle encoding for algorithms (e.g. some
    kernel methods) that prefer amplitude encoding; VQC in this package uses
    angle encoding by default per the v1-v5 lineage.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    padded_len = 1 << (n - 1).bit_length() if n > 1 else 1
    padded = np.zeros(padded_len)
    padded[:n] = x
    norm = np.linalg.norm(padded)
    if norm < 1e-12:
        raise ValidationError("Cannot amplitude-encode an all-zero (or near-zero) vector")
    return padded / norm
