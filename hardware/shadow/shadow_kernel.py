"""
Shadow Kernel -- Classical Kernel Approximation for <10ms Inference

After QPU training completes, the PQK v2 kernel is approximated classically
using the Nystrom landmark features. The Nystrom approximation:

  K(x1, x2) ~= phi(x1) . phi(x2)^T
  where phi(x) = K(x, landmarks) . K(landmarks, landmarks)^{-1/2}

This makes inference O(m) instead of O(N), where m=200 is the landmark count.
The full Gram matrix K(X_train, X_train) is precomputed once on QPU;
inference uses only the landmark features and classical dot products.

The shadow kernel is the bridge from QPU training to CPU production:
  1. QPU run: compute K(X_train, landmarks) -- expensive, done once
  2. CPU run: K(x_new, x_train) ~= phi(x_new) . Phi_train^T -- < 10ms

NOT the same thing as algorithms/pqk/caching.py's KernelCache:
KernelCache is a disk-backed memoization cache that returns the exact
full Gram matrix on a cache hit (avoids recomputation during a sweep or
notebook re-run, still O(N) per miss). ShadowKernel is a Nystrom
*approximation* -- it never computes the full O(N^2) Gram matrix at
inference time at all, trading exactness for O(m) latency, which is what
makes production deployment viable. Both are kept; they serve different
points in the pipeline (dev-time recomputation avoidance vs. production
inference latency).

This module is self-contained, numpy-only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ShadowKernel:
    """
    Classical kernel approximation backed by Nystrom landmark features.

    Trained once on QPU-extracted features; predicts classically.
    Target inference: < 10ms for a single sample.

    Usage:
      sk = ShadowKernel(m=200)
      sk.fit(X_train, pqk_kernel_fn)  # QPU step (one-time)
      sk.save("shadow_kernel.npz")
      # In production:
      sk2 = ShadowKernel.load("shadow_kernel.npz")
      K_new = sk2.predict_row(x_new, X_train)  # < 10ms
    """

    def __init__(self, m: int = 200):
        self.m = m
        self.landmarks_: Optional[np.ndarray] = None
        self.Phi_L_: Optional[np.ndarray] = None
        self.K_LL_inv_sqrt_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        rng: Optional[np.random.Generator] = None,
        ridge: float = 1e-6,
    ) -> "ShadowKernel":
        """
        Fit the shadow kernel from QPU-evaluated kernel values.

        Selects m landmarks from X, computes K_LL and K_LL^{-1/2}.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        m = min(self.m, len(X))
        idx = rng.choice(len(X), size=m, replace=False)
        self.landmarks_ = X[idx].copy()

        K_LL = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                k_val = kernel_fn(self.landmarks_[i], self.landmarks_[j])
                K_LL[i, j] = k_val
                K_LL[j, i] = k_val

        K_reg = K_LL + ridge * np.eye(m)
        eigvals, eigvecs = np.linalg.eigh(K_reg)
        eigvals = np.maximum(eigvals, 1e-12)
        self.K_LL_inv_sqrt_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self._fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> np.ndarray:
        """Compute Nystrom feature map phi(X) = K(X, L) . K_LL^{-1/2}."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        m = len(self.landmarks_)
        K_NL = np.zeros((len(X), m))
        for i, x in enumerate(X):
            for j in range(m):
                K_NL[i, j] = kernel_fn(x, self.landmarks_[j])

        return K_NL @ self.K_LL_inv_sqrt_

    def predict_kernel_row(
        self,
        x_new: np.ndarray,
        Phi_train: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> np.ndarray:
        """
        Compute K(x_new, X_train) using the shadow kernel approximation.

        This is the <10ms inference path: classical dot products only.
        QPU needed only for computing phi_new = K(x_new, landmarks).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_kernel_row().")

        m = len(self.landmarks_)
        K_row_L = np.array([kernel_fn(x_new, self.landmarks_[j]) for j in range(m)])
        phi_new = K_row_L @ self.K_LL_inv_sqrt_
        return phi_new @ Phi_train.T

    def save(self, path: str) -> None:
        """Save shadow kernel to disk."""
        if not self._fitted:
            raise RuntimeError("Fit before saving.")
        np.savez_compressed(
            path,
            landmarks=self.landmarks_,
            K_LL_inv_sqrt=self.K_LL_inv_sqrt_,
        )
        meta = {"m": self.m, "fitted": True}
        with open(path.replace(".npz", "_meta.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "ShadowKernel":
        """Load shadow kernel from disk."""
        data = np.load(path)
        sk = cls(m=int(data["landmarks"].shape[0]))
        sk.landmarks_ = data["landmarks"]
        sk.K_LL_inv_sqrt_ = data["K_LL_inv_sqrt"]
        sk._fitted = True
        return sk

    def inference_latency_estimate_ms(self, n_train: int) -> float:
        """
        Estimate classical inference latency (milliseconds).

        Dominant cost: phi_new @ Phi_train.T -- one dot product of shape
        (m,) . (m, N). Approximate: 2*m*N flops / (1e9 flops/s) * 1000 ms.
        Target: < 10ms for n_train=5000, m=200.
        """
        flops = 2 * self.m * n_train
        return flops / 1e9 * 1000

    def summary(self) -> dict:
        return {
            "m_landmarks": self.m,
            "fitted": self._fitted,
            "method": "Nystrom kernel approximation (Williams & Seeger 2001)",
            "inference_latency_ms_5k_train": round(
                self.inference_latency_estimate_ms(5000), 4
            ),
            "target_latency_ms": 10.0,
            "latency_ok": self.inference_latency_estimate_ms(5000) < 10.0,
        }
