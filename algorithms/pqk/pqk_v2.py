"""PQK v2 -- Projected Quantum Kernel with KTA alignment (Gen2, recommended).

The encoding circuit itself lives in `feature_maps.AngleCNOTRingV2` /
`feature_maps.project_state` (deduplicated -- Gen2's original inlined its own
copy of the same angle-encode-plus-CNOT-ring circuit) rather than being
reimplemented here.

  kappa(x, x') = sum_i <Z_i>(x) . <Z_i>(x')   (projected kernel)

Training: KTA gradient ascent (kta_alignment.py) until gamma > 0.3, then a
Nystrom landmark approximation (nystrom.py) for O(m) inference, then a
closed-form ridge-regression readout head.

Deployment note carried over from Gen2's docstring: after QPU training the
intended production path is a classical-shadow extraction feeding inference
from a cache (e.g. Redis) rather than querying a QPU per prediction -- that
deployment integration is out of scope for this algorithm module and belongs
under integrations/ once built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms.pqk.feature_maps import AngleCNOTRingV2, project_state
from algorithms.pqk.kta_alignment import kta_gradient_ascent
from algorithms.pqk.nystrom import NystromApproximation
from core.errors import ValidationError


def pqk_kernel_value(x1: np.ndarray, x2: np.ndarray, n: int, encoding_params: np.ndarray | None = None) -> float:
    fmap = AngleCNOTRingV2(n_qubits=n)
    f1 = project_state(fmap.encode(x1, encoding_params), n)
    f2 = project_state(fmap.encode(x2, encoding_params), n)
    return float(np.dot(f1, f2))


@dataclass
class PQKConfig:
    n_qubits: int = 8
    kta_target: float = 0.3
    kta_steps: int = 200
    kta_lr: float = 0.05
    nystrom_m: int = 200
    ridge_alpha: float = 1.0
    seed: int | None = None
    nystrom_ridge: float = 1e-6


class PQKv2:
    """Projected Quantum Kernel v2 classifier: KTA-aligned encoding +
    Nystrom-approximated kernel + ridge-regression readout."""

    def __init__(self, config: PQKConfig | None = None) -> None:
        self.cfg = config or PQKConfig()
        self.encoding_params_: np.ndarray | None = None
        self.nystrom_: NystromApproximation | None = None
        self.readout_weights_: np.ndarray | None = None
        self.readout_bias_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.kta_history_: list[float] = []
        self.final_kta_: float | None = None

    def _make_kernel_fn(self, params: np.ndarray):
        n = self.cfg.n_qubits

        def fn(x1: np.ndarray, x2: np.ndarray) -> float:
            return pqk_kernel_value(x1, x2, n, params)

        return fn

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PQKv2":
        rng = np.random.default_rng(self.cfg.seed)
        n = self.cfg.n_qubits
        self.classes_ = np.unique(y)

        m_kta = min(50, len(X))
        idx_kta = rng.choice(len(X), size=m_kta, replace=False)
        X_kta, y_kta = X[idx_kta], y[idx_kta].astype(float)

        params_init = rng.uniform(0.5, 1.5, n)
        best_params, best_kta, history = kta_gradient_ascent(
            params_init, X_kta, y_kta, self._make_kernel_fn,
            n_steps=self.cfg.kta_steps, lr=self.cfg.kta_lr, target_kta=self.cfg.kta_target,
        )
        self.encoding_params_, self.final_kta_, self.kta_history_ = best_params, best_kta, history

        kernel_fn = self._make_kernel_fn(best_params)
        self.nystrom_ = NystromApproximation(
            m=min(self.cfg.nystrom_m, len(X)), ridge=self.cfg.nystrom_ridge, seed=self.cfg.seed
        )
        self.nystrom_.fit(X, kernel_fn)
        phi = self.nystrom_.transform(X)

        weights, biases = [], []
        for c in self.classes_:
            w, b = self._ridge_fit(phi, (y == c).astype(float))
            weights.append(w)
            biases.append(b)
        self.readout_weights_ = np.stack(weights)
        self.readout_bias_ = np.array(biases)
        return self

    def _ridge_fit(self, phi: np.ndarray, y_bin: np.ndarray) -> tuple[np.ndarray, float]:
        m = phi.shape[1]
        A = phi.T @ phi + self.cfg.ridge_alpha * np.eye(m)
        b_vec = phi.T @ y_bin
        w = np.linalg.solve(A, b_vec)
        bias = float(np.mean(y_bin) - phi.mean(axis=0) @ w)
        return w, bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.nystrom_ is None:
            raise ValidationError("Call fit() before predict_proba().")
        phi = self.nystrom_.transform(X)
        scores = phi @ self.readout_weights_.T + self.readout_bias_[None, :]
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        if self.nystrom_ is None:
            raise ValidationError("Call fit() before kernel_matrix().")
        phi1 = self.nystrom_.transform(X1)
        phi2 = phi1 if X2 is None else self.nystrom_.transform(X2)
        return phi1 @ phi2.T

    @property
    def level1_passed(self) -> bool:
        """True once final KTA >= 0.3 -- the acceptance bar this platform
        inherited from the original PQK v1 -> v2 remediation."""
        return bool(self.final_kta_ is not None and self.final_kta_ >= 0.3)
