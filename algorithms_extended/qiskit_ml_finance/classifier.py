"""QNNClassifier: binary financial classification via FinancialQNN, trained
end-to-end with exact gradients (contrast with `algorithms.vqc_qml.training
.VQCTrainer`'s ridge-closed-form + gradient-free outer loop -- see qnn.py's
module docstring for the architectural distinction this closes).

Reuses `algorithms.vqc_qml.calibration.PlattCalibrator` for probability
calibration rather than reimplementing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.seeds import seed_everything
from core.metrics import accuracy
from algorithms.vqc_qml.ansatz import VQCAnsatz
from algorithms.vqc_qml.encoding import AngleEncoder, truncate_or_pad
from algorithms.vqc_qml.calibration import PlattCalibrator

from algorithms_extended.qiskit_ml_finance.qnn import FinancialQNN
from algorithms_extended.qiskit_ml_finance.optimizers import Adam


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class ClassifierConfig:
    n_qubits: int = 4
    ansatz_version: str = "v1_linear"
    epochs: int = 80
    lr: float = 0.1
    seed: int = 42
    class_weighted: bool = True


@dataclass
class QNNClassifier:
    cfg: ClassifierConfig = field(default_factory=ClassifierConfig)

    def __post_init__(self) -> None:
        self.ansatz_ = VQCAnsatz(n_qubits=self.cfg.n_qubits, version=self.cfg.ansatz_version)
        self.qnn_ = FinancialQNN(ansatz=self.ansatz_, n_outputs=1)
        self.encoder_ = AngleEncoder()
        self.calibrator_ = PlattCalibrator()
        self.classes_ = None
        self.theta_ = self.W_ = self.b_ = None
        self.loss_history_: list[float] = []

    def _weights(self, y01: np.ndarray) -> np.ndarray:
        if not self.cfg.class_weighted:
            return np.ones_like(y01)
        n_pos, n_neg = y01.sum(), len(y01) - y01.sum()
        n_pos, n_neg = max(n_pos, 1), max(n_neg, 1)
        w = np.where(y01 == 1, len(y01) / (2 * n_pos), len(y01) / (2 * n_neg))
        return w

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QNNClassifier":
        seed_everything(self.cfg.seed)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            from core.errors import ValidationError

            raise ValidationError("QNNClassifier currently supports binary labels only")
        X = truncate_or_pad(np.asarray(X, dtype=float), self.cfg.n_qubits)
        X_encoded = self.encoder_.fit_transform(X)
        y01 = (np.asarray(y) == self.classes_[1]).astype(float)
        sw = self._weights(y01)

        theta, W, b = self.qnn_.init_params(seed=self.cfg.seed)
        opt = Adam(lr=self.cfg.lr)
        self.loss_history_ = []
        scores = None

        for _ in range(self.cfg.epochs):
            fwd = self.qnn_.forward(X_encoded, theta, W, b)
            scores = fwd.output[:, 0]
            p = _sigmoid(scores)
            eps = 1e-12
            bce = -np.mean(sw * (y01 * np.log(p + eps) + (1 - y01) * np.log(1 - p + eps)))
            self.loss_history_.append(float(bce))
            grad_scores = (sw * (p - y01) / len(y01)).reshape(-1, 1)
            grad_theta, grad_W, grad_b = self.qnn_.backward(X_encoded, theta, W, fwd, grad_scores)
            updated = opt.step(
                {"theta": theta, "W": W, "b": b},
                {"theta": grad_theta, "W": grad_W, "b": grad_b},
            )
            theta, W, b = updated["theta"], updated["W"], updated["b"]

        self.theta_, self.W_, self.b_ = theta, W, b
        # calibrate on the final training-set scores (a held-out calibration
        # set would be preferable in production; documented as a known
        # simplification, not hidden)
        self.calibrator_.fit(scores, y01)
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        X = truncate_or_pad(np.asarray(X, dtype=float), self.cfg.n_qubits)
        X_encoded = self.encoder_.transform(X)
        fwd = self.qnn_.forward(X_encoded, self.theta_, self.W_, self.b_)
        return fwd.output[:, 0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator_.predict_proba(self.decision_scores(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.where(p >= 0.5, self.classes_[1], self.classes_[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy(y, self.predict(X))
