"""VQC training loop.

Readout is a ridge-regression head over the circuit's Z-expectation features
(Schuld's Representer Theorem: every quantum model is linear in its encoding
feature space, so the readout layer can be solved in closed form and kept
out of the non-convex optimization entirely — the same design choice Gen2's
v5 made, generalized here for any VQCAnsatz version).

Two gradient-free optimizer families are supported for the (non-convex)
circuit parameters: COBYLA (via scipy) and a hand-rolled SPSA, since exact
parameter-shift gradients scale as O(n_params) circuit evaluations per step
and are often impractical outside small demos/tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from scipy import optimize as sp_optimize

from core.errors import ValidationError
from core.seeds import seed_everything

from algorithms.vqc_qml.ansatz import VQCAnsatz
from algorithms.vqc_qml.encoding import AngleEncoder, truncate_or_pad


class OptimizerFamily(str, Enum):
    COBYLA = "cobyla"
    SPSA = "spsa"


NoiseFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class TrainingConfig:
    optimizer: OptimizerFamily = OptimizerFamily.COBYLA
    max_iter: int = 150
    ridge_alpha: float = 0.1
    class_weighted: bool = True
    seed: int = 42
    # SPSA-only
    spsa_a: float = 0.15
    spsa_c: float = 0.1
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    # readout-noise approximation applied to z-expectations during training
    # (a physically-motivated attenuation: E[Z]_noisy ~= (1-2p) * E[Z]_clean
    # for i.i.d. bit-flip readout error with flip probability p). For full
    # gate-level noisy-circuit simulation, use hardware.noise + Aer directly
    # and pass its output through `noise_fn` instead of this scalar.
    readout_flip_prob: float = 0.0
    noise_fn: NoiseFn | None = field(default=None, repr=False)


@dataclass
class RidgeReadout:
    weights_: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    classes_: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def fit(self, Z: np.ndarray, y: np.ndarray, alpha: float, sample_weight: np.ndarray | None = None) -> "RidgeReadout":
        self.classes_ = np.unique(y)
        Y = np.stack([(y == c).astype(float) * 2 - 1 for c in self.classes_], axis=1)  # one-vs-rest +-1 targets
        Zb = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)  # bias term
        if sample_weight is not None:
            W = np.diag(sample_weight)
            A = Zb.T @ W @ Zb + alpha * np.eye(Zb.shape[1])
            b = Zb.T @ W @ Y
        else:
            A = Zb.T @ Zb + alpha * np.eye(Zb.shape[1])
            b = Zb.T @ Y
        self.weights_ = np.linalg.solve(A, b)  # shape (n_features+1, n_classes)
        return self

    def decision_function(self, Z: np.ndarray) -> np.ndarray:
        Zb = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
        return Zb @ self.weights_

    def predict(self, Z: np.ndarray) -> np.ndarray:
        scores = self.decision_function(Z)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        scores = self.decision_function(Z)
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class VQCTrainer:
    def __init__(self, ansatz: VQCAnsatz, config: TrainingConfig | None = None) -> None:
        self.ansatz = ansatz
        self.cfg = config or TrainingConfig()
        self.encoder_ = AngleEncoder()
        self.readout_: RidgeReadout | None = None
        self.params_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def _features(self, X_encoded: np.ndarray, params: np.ndarray) -> np.ndarray:
        Z = np.array([self.ansatz.z_expectations(x, params) for x in X_encoded])
        if self.cfg.readout_flip_prob > 0:
            Z = Z * (1 - 2 * self.cfg.readout_flip_prob)
        if self.cfg.noise_fn is not None:
            Z = self.cfg.noise_fn(Z)
        return Z

    def _sample_weights(self, y: np.ndarray) -> np.ndarray | None:
        if not self.cfg.class_weighted:
            return None
        classes, counts = np.unique(y, return_counts=True)
        freq = dict(zip(classes, counts))
        return np.array([len(y) / (len(classes) * freq[label]) for label in y])

    def _loss(self, params: np.ndarray, X_encoded: np.ndarray, y: np.ndarray, sample_weight) -> float:
        Z = self._features(X_encoded, params)
        readout = RidgeReadout().fit(Z, y, self.cfg.ridge_alpha, sample_weight)
        scores = readout.decision_function(Z)
        targets = np.stack([(y == c).astype(float) * 2 - 1 for c in readout.classes_], axis=1)
        return float(np.mean((scores - targets) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCTrainer":
        seed_everything(self.cfg.seed)
        X = truncate_or_pad(np.asarray(X, dtype=float), self.ansatz.n_qubits)
        X_encoded = self.encoder_.fit_transform(X)
        y = np.asarray(y)
        sample_weight = self._sample_weights(y)

        params0 = self.ansatz.init_params(seed=self.cfg.seed)
        self.loss_history_ = []

        def tracked_loss(p: np.ndarray) -> float:
            loss = self._loss(p, X_encoded, y, sample_weight)
            self.loss_history_.append(loss)
            return loss

        if self.cfg.optimizer == OptimizerFamily.COBYLA:
            result = sp_optimize.minimize(
                tracked_loss, params0, method="COBYLA",
                options={"maxiter": self.cfg.max_iter, "rhobeg": 0.3},
            )
            best_params = result.x
        elif self.cfg.optimizer == OptimizerFamily.SPSA:
            best_params = self._spsa(tracked_loss, params0)
        else:
            raise ValidationError(f"Unknown optimizer family {self.cfg.optimizer}")

        self.params_ = best_params
        Z = self._features(X_encoded, best_params)
        self.readout_ = RidgeReadout().fit(Z, y, self.cfg.ridge_alpha, sample_weight)
        return self

    def _spsa(self, loss_fn: Callable[[np.ndarray], float], params0: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed)
        theta = params0.copy()
        a, c, alpha, gamma = self.cfg.spsa_a, self.cfg.spsa_c, self.cfg.spsa_alpha, self.cfg.spsa_gamma
        for k in range(1, self.cfg.max_iter + 1):
            ak = a / (k**alpha)
            ck = c / (k**gamma)
            delta = rng.choice([-1.0, 1.0], size=theta.shape)
            loss_plus = loss_fn(theta + ck * delta)
            loss_minus = loss_fn(theta - ck * delta)
            ghat = (loss_plus - loss_minus) / (2 * ck) * delta
            theta = theta - ak * ghat
        return theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.readout_ is None or self.params_ is None:
            raise ValidationError("Call fit() before predict()")
        X = truncate_or_pad(np.asarray(X, dtype=float), self.ansatz.n_qubits)
        Z = self._features(self.encoder_.transform(X), self.params_)
        return self.readout_.predict(Z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.readout_ is None or self.params_ is None:
            raise ValidationError("Call fit() before predict_proba()")
        X = truncate_or_pad(np.asarray(X, dtype=float), self.ansatz.n_qubits)
        Z = self._features(self.encoder_.transform(X), self.params_)
        return self.readout_.predict_proba(Z)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw ridge-readout decision scores per class (pre-softmax), for
        callers that need to fit their own calibrator on top (e.g.
        PlattCalibrator) rather than the built-in softmax probabilities."""
        if self.readout_ is None or self.params_ is None:
            raise ValidationError("Call fit() before decision_scores()")
        X = truncate_or_pad(np.asarray(X, dtype=float), self.ansatz.n_qubits)
        Z = self._features(self.encoder_.transform(X), self.params_)
        return self.readout_.decision_function(Z)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
