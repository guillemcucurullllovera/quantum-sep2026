"""Classical regression baseline. No classical regression baseline existed
anywhere in the platform (checked: classical/{optimization,monte_carlo,risk,
portfolio,option_pricing,reservoir,kernels,machine_learning} -- only
classification baselines in classical/machine_learning). Every quantum method
in this platform is expected to ship with an honest classical comparator, so
this is a small, real, necessary addition -- ridge regression, closed form.
Classification comparisons reuse classical.machine_learning.baselines
directly, not duplicated here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.metrics import mse, rmse


@dataclass
class RegressionBaselineResult:
    name: str
    mse: float
    rmse: float
    predictions: np.ndarray


def ridge_regression_baseline(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, alpha: float = 1.0
) -> RegressionBaselineResult:
    Xb_train = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    Xb_test = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)
    A = Xb_train.T @ Xb_train + alpha * np.eye(Xb_train.shape[1])
    w = np.linalg.solve(A, Xb_train.T @ y_train)
    preds = Xb_test @ w
    return RegressionBaselineResult(
        name="ridge_regression", mse=mse(y_test, preds), rmse=rmse(y_test, preds), predictions=preds
    )
