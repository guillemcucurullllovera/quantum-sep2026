"""Classical classifier baselines: logistic regression and a tree ensemble.

Falls back to a hand-rolled numpy logistic regression if scikit-learn is
unavailable in a given environment, so this module — and everything that
depends on it for an honest comparison — never silently skips the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.metrics import accuracy, calibration_error

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression as _SKLogisticRegression

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only in sklearn-less envs
    _HAS_SKLEARN = False


@dataclass
class BaselineResult:
    name: str
    accuracy: float
    calibration_error: float
    predictions: np.ndarray
    probabilities: np.ndarray


class _NumpyLogisticRegression:
    """Minimal hand-rolled binary logistic regression (gradient descent),
    used only when scikit-learn is not installed, so the platform's
    classical-baseline guarantee never silently degrades to "no baseline"."""

    def __init__(self, lr: float = 0.1, n_iter: int = 500, l2: float = 1e-3) -> None:
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyLogisticRegression":
        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        y01 = (y == np.unique(y)[-1]).astype(float)  # positive class = larger label value
        for _ in range(self.n_iter):
            z = X @ self.w_ + self.b_
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = X.T @ (p - y01) / n + self.l2 * self.w_
            grad_b = float(np.mean(p - y01))
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w_ + self.b_
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1 - p1, p1], axis=1)


def logistic_regression_baseline(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> BaselineResult:
    classes = np.unique(y_train)
    if _HAS_SKLEARN:
        model = _SKLogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        preds = model.predict(X_test)
    else:
        if len(classes) != 2:
            raise ValueError("Fallback numpy logistic regression only supports binary labels")
        model = _NumpyLogisticRegression().fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        preds = classes[np.argmax(proba, axis=1)]

    pos_class_proba = proba[:, -1]
    y_binary_test = (y_test == classes[-1]).astype(float)
    return BaselineResult(
        name="logistic_regression",
        accuracy=accuracy(y_test, preds),
        calibration_error=calibration_error(pos_class_proba, y_binary_test),
        predictions=preds,
        probabilities=proba,
    )


def random_forest_baseline(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_estimators: int = 200
) -> BaselineResult:
    if not _HAS_SKLEARN:
        raise ImportError(
            "random_forest_baseline requires scikit-learn; install it or use "
            "logistic_regression_baseline (which has a dependency-free fallback)."
        )
    classes = np.unique(y_train)
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)
    pos_class_proba = proba[:, -1]
    y_binary_test = (y_test == classes[-1]).astype(float)
    return BaselineResult(
        name="random_forest",
        accuracy=accuracy(y_test, preds),
        calibration_error=calibration_error(pos_class_proba, y_binary_test),
        predictions=preds,
        probabilities=proba,
    )
