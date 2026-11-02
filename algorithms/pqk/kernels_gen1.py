"""Gen1 fidelity/projected quantum kernel pipeline + from-scratch SMO SVM.

Ported from the Gen1 research repository.
Kept as a working historical baseline (its ZZFeatureMap has the concentration
issue documented in kta_alignment.py) -- prefer `pqk_v2.PQKv2` for new work.
`SimpleSVM` here is also reused by classical/kernels/baselines.py as the
classical comparator's SVM, since it takes an arbitrary precomputed Gram
matrix and doesn't care whether the kernel came from a quantum circuit or
an RBF/polynomial classical kernel.
"""
from __future__ import annotations

import numpy as np

from algorithms.pqk.feature_maps import project_state


def fidelity_kernel_element(state_i: np.ndarray, state_j: np.ndarray) -> float:
    """K(xi, xj) = |<psi(xi)|psi(xj)>|^2."""
    overlap = np.dot(state_i.conj(), state_j)
    return float(abs(overlap) ** 2)


def compute_fidelity_gram(X: np.ndarray, feature_map, X2: np.ndarray | None = None) -> np.ndarray:
    states_1 = [feature_map.encode(list(x)) for x in X]
    states_2 = states_1 if X2 is None else [feature_map.encode(list(x)) for x in X2]
    n1, n2 = len(states_1), len(states_2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = fidelity_kernel_element(states_1[i], states_2[j])
    return K


def compute_projected_features(X: np.ndarray, feature_map) -> np.ndarray:
    return np.array([project_state(feature_map.encode(list(x))) for x in X])


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    dist_sq = sq1 + sq2.T - 2 * X1 @ X2.T
    return np.exp(-gamma * np.clip(dist_sq, 0, None))


def compute_projected_gram(
    X: np.ndarray, feature_map, X2: np.ndarray | None = None, gamma: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    proj_X = compute_projected_features(X, feature_map)
    if X2 is None:
        return rbf_kernel(proj_X, proj_X, gamma), proj_X, None
    proj_X2 = compute_projected_features(X2, feature_map)
    return rbf_kernel(proj_X, proj_X2, gamma), proj_X, proj_X2


class SimpleSVM:
    """Kernel SVM via SMO-style coordinate ascent on a precomputed Gram
    matrix -- works for any kernel (quantum or classical), since it only
    ever sees K, never the raw features."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-4) -> None:
        self.C, self.max_iter, self.tol = C, max_iter, tol
        self.alpha: np.ndarray | None = None
        self.b = 0.0
        self.y_train: np.ndarray | None = None

    def fit(self, K: np.ndarray, y: np.ndarray) -> int:
        n = len(y)
        y = y.astype(float)
        self.alpha = np.zeros(n)
        self.y_train = y.copy()
        rng = np.random.default_rng(0)

        for _ in range(self.max_iter):
            num_changed = 0
            for i in range(n):
                f_i = np.dot(self.alpha * y, K[i]) + self.b
                e_i = f_i - y[i]
                if (y[i] * e_i < -self.tol and self.alpha[i] < self.C) or (
                    y[i] * e_i > self.tol and self.alpha[i] > 0
                ):
                    j = i
                    while j == i:
                        j = int(rng.integers(n))
                    f_j = np.dot(self.alpha * y, K[j]) + self.b
                    e_j = f_j - y[j]

                    if y[i] != y[j]:
                        lo = max(0.0, self.alpha[j] - self.alpha[i])
                        hi = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        lo = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
                        hi = min(self.C, self.alpha[i] + self.alpha[j])
                    if abs(lo - hi) < 1e-10:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha_j_new = np.clip(self.alpha[j] - y[j] * (e_i - e_j) / eta, lo, hi)
                    if abs(alpha_j_new - self.alpha[j]) < 1e-8:
                        continue
                    alpha_i_new = self.alpha[i] + y[i] * y[j] * (self.alpha[j] - alpha_j_new)

                    b1 = (
                        self.b - e_i - y[i] * (alpha_i_new - self.alpha[i]) * K[i, i]
                        - y[j] * (alpha_j_new - self.alpha[j]) * K[i, j]
                    )
                    b2 = (
                        self.b - e_j - y[i] * (alpha_i_new - self.alpha[i]) * K[i, j]
                        - y[j] * (alpha_j_new - self.alpha[j]) * K[j, j]
                    )
                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_new < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    self.alpha[i], self.alpha[j] = alpha_i_new, alpha_j_new
                    num_changed += 1
            if num_changed == 0:
                break

        return int(np.sum(self.alpha > 1e-6))

    def decision_function(self, K_test: np.ndarray) -> np.ndarray:
        return K_test @ (self.alpha * self.y_train) + self.b

    def predict_kernel(self, K_test: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(K_test))

    def accuracy(self, K_test: np.ndarray, y_test: np.ndarray) -> float:
        return float(np.mean(self.predict_kernel(K_test) == y_test))
