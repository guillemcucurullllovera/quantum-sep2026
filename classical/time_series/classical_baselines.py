"""
Classical baselines for quantum component benchmarking (LR, LightGBM, Ridge).

Every quantum component must beat these before any advantage claim.

    LogisticBaseline  : LR  — documented Level-1 go/no-go bar is 63.8% OOS accuracy
    LGBMBaseline      : LightGBM — stronger tree baseline
    RidgeBaseline     : Ridge classifier (via decision function)

All share the same interface:
    fit(X, y)          -> self
    predict_proba(X)   -> (n, 3) array: [P(sell), P(hold), P(buy)]
    accuracy(X, y)      -> float

Requires: scikit-learn (installed), optionally lightgbm (not installed in
this environment — LGBMBaseline raises ImportError with an install hint if
used directly; benchmark_all() omits it gracefully rather than failing).
"""

from __future__ import annotations

import numpy as np


def _require_sklearn(name: str = "sklearn"):
    try:
        import sklearn
        return sklearn
    except ImportError:
        raise ImportError(
            f"{name} requires scikit-learn. Install with: pip install scikit-learn"
        )


def _require_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        raise ImportError(
            "LGBMBaseline requires lightgbm. Install with: pip install lightgbm"
        )


class _BaselineBase:
    """Shared interface for all classical baselines."""
    _label_map = {-1: 0, 0: 1, 1: 2}
    _inv_label_map = {0: -1, 1: 0, 2: 1}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaselineBase":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n, 3) proba array: columns = [P(sell), P(hold), P(buy)]."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (n,) predicted labels in {-1, 0, 1}."""
        proba = self.predict_proba(X)
        mapped = np.argmax(proba, axis=1)
        return np.array([self._inv_label_map[int(m)] for m in mapped])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class LogisticBaseline(_BaselineBase):
    """
    Logistic Regression baseline.

    The bar to beat at Level 1 go/no-go: 63.8% OOS accuracy.
    (VQC v4 achieved 57.1% on this bar historically — failed it.)
    """
    level1_threshold: float = 0.638  # Level 1 go/no-go acceptance bar

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        _require_sklearn()
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        y_mapped = np.array([self._label_map[int(v)] for v in y])
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self._model.fit(X_scaled, y_mapped)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)


class LGBMBaseline(_BaselineBase):
    """
    LightGBM baseline — stronger tree-based model.

    HISTORICAL framing (WITHDRAWN, Q10: the quantum circuits are CSIM, so
    beating a classical baseline is not evidence of advantage). If this
    fails to beat quantum components, the case is stronger. If it beats
    them, retrain quantum with better features first.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMBaseline":
        lgb = _require_lgbm()
        y_mapped = np.array([self._label_map[int(v)] for v in y])
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_state,
            verbose=-1,
        )
        self._model.fit(X, y_mapped)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict_proba(X)


class RidgeBaseline(_BaselineBase):
    """
    Ridge Classifier baseline.

    Uses the same representer-theorem justification as VQC v5's ridge readout
    (Schuld, "Machine Learning with Quantum Computers", §6.5) — provides a
    pure-linear comparison for the quantum kernel.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        _require_sklearn()
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        y_mapped = np.array([self._label_map[int(v)] for v in y])
        self._model = RidgeClassifier(alpha=self.alpha, class_weight="balanced")
        self._model.fit(X_scaled, y_mapped)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        _require_sklearn()
        X_scaled = self._scaler.transform(X)
        decisions = self._model.decision_function(X_scaled)
        decisions -= decisions.max(axis=1, keepdims=True)
        exp_d = np.exp(decisions)
        return exp_d / exp_d.sum(axis=1, keepdims=True)


def benchmark_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Fit all available baselines and return accuracy dict.

    Returns:
        {"logistic": {"accuracy": float}, "lgbm": {...}, "ridge": {...}}
        ("lgbm" is omitted, not an error, if lightgbm isn't installed.)
    """
    results = {}
    for name, cls in [("logistic", LogisticBaseline), ("ridge", RidgeBaseline)]:
        try:
            m = cls().fit(X_train, y_train)
            results[name] = {"accuracy": m.accuracy(X_test, y_test)}
        except ImportError:
            pass
    try:
        m = LGBMBaseline().fit(X_train, y_train)
        results["lgbm"] = {"accuracy": m.accuracy(X_test, y_test)}
    except ImportError:
        pass
    return results


__all__ = [
    "LogisticBaseline",
    "LGBMBaseline",
    "RidgeBaseline",
    "benchmark_all",
]
