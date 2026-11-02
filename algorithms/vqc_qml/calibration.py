"""Platt-scaling probability calibration for VQC (or any margin-based
binary classifier) outputs, validated against core.metrics.calibration_error.

A ridge-regression readout produces a real-valued score, not a calibrated
probability — feeding raw scores through a sigmoid without fitting one is a
common source of overconfident/underconfident quantum-classifier probability
claims. This module fits the sigmoid's scale/offset instead of assuming them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize as sp_optimize

from core.errors import ValidationError
from core.metrics import calibration_error


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class PlattCalibrator:
    """Fits p(y=1|score) = sigmoid(A*score + B) by minimizing binary
    cross-entropy over held-out (score, label) pairs. Binary labels only
    (use one calibrator per one-vs-rest class for multiclass)."""

    A_: float = field(default=None, repr=False)  # type: ignore[assignment]
    B_: float = field(default=None, repr=False)  # type: ignore[assignment]

    def fit(self, scores: np.ndarray, y_binary: np.ndarray) -> "PlattCalibrator":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y_binary, dtype=float)
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValidationError("PlattCalibrator requires binary {0,1} labels")

        def neg_log_likelihood(params: np.ndarray) -> float:
            A, B = params
            p = np.clip(_sigmoid(A * scores + B), 1e-12, 1 - 1e-12)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        result = sp_optimize.minimize(neg_log_likelihood, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
        self.A_, self.B_ = result.x
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self.A_ is None:
            raise ValidationError("Call fit() before predict_proba()")
        return _sigmoid(self.A_ * np.asarray(scores, dtype=float) + self.B_)

    def evaluate_calibration(self, scores: np.ndarray, y_binary: np.ndarray, n_bins: int = 10) -> float:
        """Returns Expected Calibration Error on the given (held-out) data."""
        probs = self.predict_proba(scores)
        return calibration_error(probs, np.asarray(y_binary, dtype=float), n_bins=n_bins)
