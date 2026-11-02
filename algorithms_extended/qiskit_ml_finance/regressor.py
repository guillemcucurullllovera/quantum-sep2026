"""QNNRegressor: continuous-target financial regression via FinancialQNN.

Genuine gap this closes: `algorithms/vqc_qml` (v1-v5) is
classification-only -- `RidgeReadout`/`VQCv5` both fit one-vs-rest +-1
class targets. Nothing in the platform previously trained a VQC/QNN-style
circuit against a continuous financial target (e.g. a realized-volatility
level or an expected-return estimate) end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.seeds import seed_everything
from algorithms.vqc_qml.ansatz import VQCAnsatz
from algorithms.vqc_qml.encoding import AngleEncoder, truncate_or_pad

from algorithms_extended.qiskit_ml_finance.qnn import FinancialQNN
from algorithms_extended.qiskit_ml_finance.optimizers import Adam, cobyla_minimize


@dataclass
class RegressorConfig:
    n_qubits: int = 4
    ansatz_version: str = "v1_linear"
    epochs: int = 60
    lr: float = 0.05
    seed: int = 42


@dataclass
class QNNRegressor:
    cfg: RegressorConfig = field(default_factory=RegressorConfig)

    def __post_init__(self) -> None:
        self.ansatz_ = VQCAnsatz(n_qubits=self.cfg.n_qubits, version=self.cfg.ansatz_version)
        self.qnn_ = FinancialQNN(ansatz=self.ansatz_, n_outputs=1)
        self.encoder_ = AngleEncoder()
        self.theta_ = self.W_ = self.b_ = None
        self.loss_history_: list[float] = []

    def _mse_grad(self, out: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(y)
        return (2.0 / n) * (out[:, 0] - y).reshape(-1, 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QNNRegressor":
        seed_everything(self.cfg.seed)
        X = truncate_or_pad(np.asarray(X, dtype=float), self.cfg.n_qubits)
        X_encoded = self.encoder_.fit_transform(X)
        y = np.asarray(y, dtype=float)

        theta, W, b = self.qnn_.init_params(seed=self.cfg.seed)
        opt = Adam(lr=self.cfg.lr)
        self.loss_history_ = []

        for _ in range(self.cfg.epochs):
            fwd = self.qnn_.forward(X_encoded, theta, W, b)
            loss = float(np.mean((fwd.output[:, 0] - y) ** 2))
            self.loss_history_.append(loss)
            grad_out = self._mse_grad(fwd.output, y)
            grad_theta, grad_W, grad_b = self.qnn_.backward(X_encoded, theta, W, fwd, grad_out)
            updated = opt.step(
                {"theta": theta, "W": W, "b": b},
                {"theta": grad_theta, "W": grad_W, "b": grad_b},
            )
            theta, W, b = updated["theta"], updated["W"], updated["b"]

        self.theta_, self.W_, self.b_ = theta, W, b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            from core.errors import ValidationError

            raise ValidationError("Call fit() before predict()")
        X = truncate_or_pad(np.asarray(X, dtype=float), self.cfg.n_qubits)
        X_encoded = self.encoder_.transform(X)
        fwd = self.qnn_.forward(X_encoded, self.theta_, self.W_, self.b_)
        return fwd.output[:, 0]

    def fit_cobyla(self, X: np.ndarray, y: np.ndarray) -> "QNNRegressor":
        """Alternative training path using the gradient-free COBYLA optimizer
        on the identical loss surface, for the Adam-vs-COBYLA comparison."""
        seed_everything(self.cfg.seed)
        X = truncate_or_pad(np.asarray(X, dtype=float), self.cfg.n_qubits)
        X_encoded = self.encoder_.fit_transform(X)
        y = np.asarray(y, dtype=float)
        theta0, W0, b0 = self.qnn_.init_params(seed=self.cfg.seed)
        n_theta, n_qubits = len(theta0), self.cfg.n_qubits

        def unpack(flat: np.ndarray):
            theta = flat[:n_theta]
            W = flat[n_theta : n_theta + n_qubits].reshape(n_qubits, 1)
            b = flat[n_theta + n_qubits :]
            return theta, W, b

        def loss_fn(flat: np.ndarray) -> float:
            theta, W, b = unpack(flat)
            fwd = self.qnn_.forward(X_encoded, theta, W, b)
            return float(np.mean((fwd.output[:, 0] - y) ** 2))

        x0 = np.concatenate([theta0, W0.ravel(), b0])
        x_star, history = cobyla_minimize(loss_fn, x0, max_iter=150)
        self.theta_, self.W_, self.b_ = unpack(x_star)
        self.loss_history_ = history
        return self
