"""
Mamba SSM Baseline — Simplified Selective State Space Model.

Reference: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with
           Selective State Spaces" arXiv:2312.00752.

HISTORICAL framing: this was the "Level-2" go/no-go
threshold. WITHDRAWN per the Q10 audit — the quantum circuits it was meant to
benchmark (PQK-v2, QRC-10, VQC-v5) are classically simulable (CSIM), so
clearing this bar is NOT evidence of quantum advantage. It remains the
scientifically honest classical comparator to report regardless: quantum vs.
HMM (1989) is not a credible 2026 comparison; quantum vs. Mamba SSM (2023) is.

Key property: unlike LSTM (fixed recurrent gates), Mamba's state transition
parameters B, C, Δ are functions of the input x — enabling input-dependent
(selective) dynamics. This gives Mamba exponentially more effective hidden
state than HMM for the same parameter count.

This implementation is pure numpy (no CUDA/PyTorch/mamba-ssm required):
    - Correct selective dynamics
    - Slower than the CUDA-optimized version (no parallel scan)
    - Trains only the linear read-out head via SGD; the SSM layers use fixed
      random features (full backprop through the recurrence needs PyTorch)

For production/benchmarking with full backprop, install the official package:
    pip install mamba-ssm   (requires CUDA)
This module has no dependency on that package and runs standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MambaConfig:
    d_model: int = 16        # input/output dimension (will match n_features)
    d_state: int = 16        # SSM state dimension (N in the paper)
    d_conv: int = 4          # local convolution width
    expand: int = 2          # inner dimension expansion factor
    n_layers: int = 2        # number of Mamba blocks stacked
    lr: float = 1e-3         # learning rate for linear classifier head
    n_epochs: int = 50       # training epochs
    batch_size: int = 32
    random_seed: int = 42


class _SelectiveSSM:
    """
    Core Mamba selective state space layer (single block).

    Parameters (trained via gradient descent):
        log_A  : (d_inner, d_state) — diagonal state transition (log scale)
        W_B    : (d_inner, d_state) — input -> B (selective)
        W_C    : (d_inner, d_state) — state -> C (selective)
        W_dt   : (d_inner, d_inner) — input -> Δ (log timescale, selective)
        dt_bias: (d_inner,)         — Δ bias
        W_out  : (d_model, d_inner) — output projection

    Selectivity: B, C, Δ are computed from the input x at each step. This
    makes the dynamics context-dependent, unlike HMM where transition
    probabilities are fixed regardless of input value.
    """

    def __init__(self, d_model: int, d_state: int, expand: int, rng):
        d_inner = d_model * expand
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner

        scale = 0.02
        self.log_A = rng.standard_normal((d_inner, d_state)) * scale
        self.W_B = rng.standard_normal((d_inner, d_state)) * scale
        self.W_C = rng.standard_normal((d_inner, d_state)) * scale
        self.W_dt = rng.standard_normal((d_inner, d_inner)) * scale
        self.dt_bias = np.zeros(d_inner)
        self.W_in = rng.standard_normal((d_inner, d_model)) * scale
        self.W_out = rng.standard_normal((d_model, d_inner)) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass over a sequence.

        Args:
            x : (T, d_model) input sequence
        Returns:
            (T, d_model) output sequence
        """
        T = x.shape[0]
        u = x @ self.W_in.T                                      # (T, d_inner)

        B = u @ self.W_B                                          # (T, d_state)
        C = u @ self.W_C                                          # (T, d_state)
        dt = _softplus(u @ self.W_dt.T + self.dt_bias)           # (T, d_inner)

        A = -np.exp(self.log_A)                                   # (d_inner, d_state), stable
        A_bar = np.exp(dt[:, :, None] * A[None, :, :])           # (T, d_inner, d_state)
        B_bar = dt[:, :, None] * B[:, None, :]                   # (T, d_inner, d_state)

        h = np.zeros((self.d_inner, self.d_state))
        outputs = np.empty((T, self.d_inner))

        for t in range(T):
            h = A_bar[t] * h + B_bar[t] * u[t, :, None]          # (d_inner, d_state)
            y_t = np.sum(C[t, None, :] * h, axis=-1)              # (d_inner,)
            outputs[t] = y_t

        out = outputs @ self.W_out.T                              # (T, d_model)
        return out + x                                             # residual connection

    def parameters(self):
        return [self.log_A, self.W_B, self.W_C, self.W_dt,
                self.dt_bias, self.W_in, self.W_out]


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -30, 30)))


class SimpleMambaSSM:
    """
    Simplified Mamba SSM for financial time-series classification.

    Architecture: n_layers x SelectiveSSM blocks + linear classifier head.
    Training: simple SGD with mean cross-entropy loss (numpy only).

    Output: (n, 3) probability matrix — [P(sell), P(hold), P(buy)]
    Labels: {-1: sell, 0: hold, 1: buy}
    """

    _label_map = {-1: 0, 0: 1, 1: 2}
    _inv_label_map = {0: -1, 1: 0, 2: 1}

    def __init__(self, config: Optional[MambaConfig] = None):
        self.config = config or MambaConfig()
        self._layers: list = []
        self._W_head: Optional[np.ndarray] = None
        self._b_head: Optional[np.ndarray] = None
        self._d_model: Optional[int] = None
        self._fitted = False

    def _build(self, n_features: int) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        d = n_features

        self._d_model = d
        self._layers = [
            _SelectiveSSM(d, cfg.d_state, cfg.expand, rng)
            for _ in range(cfg.n_layers)
        ]
        self._W_head = rng.standard_normal((3, d)) * 0.01
        self._b_head = np.zeros(3)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(shifted)
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: Optional[str] = None,
    ) -> "SimpleMambaSSM":
        """
        Train Mamba SSM on feature matrix X with labels y in {-1, 0, 1}.

        Args:
            X            : (n_samples, n_features) feature matrix
            y            : (n_samples,) labels in {-1, 0, 1}
            class_weight : None or 'balanced'. 'balanced' scales each
                           sample's loss contribution by n_samples /
                           (n_classes * class_count), matching sklearn's
                           convention.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        n_features = X.shape[1]
        self._build(n_features)

        y_mapped = np.array([self._label_map[int(v)] for v in y])
        n = len(X)
        n_classes = 3

        if class_weight == "balanced":
            counts = np.bincount(y_mapped, minlength=n_classes).astype(float)
            counts = np.where(counts == 0, 1.0, counts)
            weights_per_class = n / (n_classes * counts)
            sample_weights = weights_per_class[y_mapped]
        else:
            sample_weights = np.ones(n)

        # SSM layers use fixed random features; only the linear head is
        # trained via SGD. This captures the SSM's temporal mixing without
        # requiring backprop through the recurrence — see module docstring.
        for epoch in range(cfg.n_epochs):
            indices = rng.permutation(n)
            for start in range(0, n, cfg.batch_size):
                batch_idx = indices[start: start + cfg.batch_size]
                X_b = X[batch_idx]
                y_b = y_mapped[batch_idx]
                w_b = sample_weights[batch_idx]

                features = self._layers[0].forward(X_b)
                for layer in self._layers[1:]:
                    features = layer.forward(features)

                logits = features @ self._W_head.T + self._b_head
                proba = self._softmax(logits)

                grad = proba.copy()
                grad[np.arange(len(y_b)), y_b] -= 1.0
                grad *= w_b[:, None]
                grad /= w_b.sum()

                self._W_head -= cfg.lr * (grad.T @ features)
                self._b_head -= cfg.lr * grad.sum(axis=0)

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n, 3) probability matrix: [P(sell), P(hold), P(buy)]."""
        if not self._fitted:
            raise RuntimeError("SimpleMambaSSM is not fitted — call fit() first.")
        features = X
        for layer in self._layers:
            features = layer.forward(features)
        logits = features @ self._W_head.T + self._b_head
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (n,) predicted labels in {-1, 0, 1}."""
        proba = self.predict_proba(X)
        mapped = np.argmax(proba, axis=1)
        return np.array([self._inv_label_map[int(m)] for m in mapped])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# Backwards-compatible alias for the old class name
MambaSSMBaseline = SimpleMambaSSM

__all__ = ["SimpleMambaSSM", "MambaSSMBaseline", "MambaConfig"]
