"""Nystroem approximation for large-scale projected-kernel inference.

Approximates the full N x N kernel matrix using m landmark points, so
inference is O(m) per query instead of O(N) full-kernel evaluations against
every training point.

Reference: Williams & Seeger (2001), "Using the Nystrom Method to Speed Up
Kernel Machines", NIPS.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class NystromApproximation:
    def __init__(self, m: int = 200, ridge: float = 1e-6, seed: int | None = None) -> None:
        self.m = m
        self.ridge = ridge
        self.rng = np.random.default_rng(seed)
        self.landmarks_: np.ndarray | None = None
        self.K_LL_inv_: np.ndarray | None = None
        self._kernel_fn: Callable | None = None

    def fit(
        self,
        X: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], float],
        landmarks: np.ndarray | None = None,
    ) -> "NystromApproximation":
        n = len(X)
        m = min(self.m, n)
        self._kernel_fn = kernel_fn

        if landmarks is not None:
            self.landmarks_ = landmarks[:m]
        else:
            idx = self.rng.choice(n, size=m, replace=False)
            self.landmarks_ = X[idx]

        m_actual = len(self.landmarks_)
        K_LL = np.zeros((m_actual, m_actual))
        for i in range(m_actual):
            for j in range(i, m_actual):
                v = kernel_fn(self.landmarks_[i], self.landmarks_[j])
                K_LL[i, j] = v
                K_LL[j, i] = v

        K_LL_reg = K_LL + self.ridge * np.eye(m_actual)
        self.K_LL_inv_ = np.linalg.inv(K_LL_reg)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.landmarks_ is None:
            raise RuntimeError("Call fit() first.")
        n, m = len(X), len(self.landmarks_)
        K_NL = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K_NL[i, j] = self._kernel_fn(X[i], self.landmarks_[j])

        eigvals, eigvecs = np.linalg.eigh(self.K_LL_inv_)
        eigvals = np.maximum(eigvals, 0)
        K_LL_inv_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        return K_NL @ K_LL_inv_sqrt

    def predict_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        phi1 = self.transform(X1)
        phi2 = self.transform(X2)
        return phi1 @ phi2.T

    def accuracy_vs_exact(
        self, X_test: np.ndarray, kernel_fn: Callable | None = None, n_test: int = 50
    ) -> float:
        """Relative Frobenius error of the Nystrom approximation vs. the
        exact kernel on a subsample -- the acceptance bar used elsewhere in
        this platform is <=2%."""
        if kernel_fn is None:
            kernel_fn = self._kernel_fn
        X_sub = X_test[:n_test]
        n = len(X_sub)

        K_exact = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                v = kernel_fn(X_sub[i], X_sub[j])
                K_exact[i, j] = v
                K_exact[j, i] = v

        K_approx = self.predict_kernel(X_sub, X_sub)
        norm_exact = np.linalg.norm(K_exact, "fro")
        if norm_exact < 1e-12:
            return 0.0
        return float(np.linalg.norm(K_approx - K_exact, "fro") / norm_exact)
