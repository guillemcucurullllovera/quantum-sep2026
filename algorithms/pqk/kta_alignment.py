"""Kernel-Target Alignment (KTA) gradient ascent.

Context: PQK v1 (`feature_maps.ZZFeatureMap`, this
module's sibling) suffers kernel concentration for the data this platform
targets (Thanasilp et al. 2024, "Exponential concentration in quantum kernel
methods") -- the kernel collapses toward a constant value for all pairs,
giving KTA gamma ~ 0 and making it useless for classification. Gen2's fix is
`feature_maps.AngleCNOTRingV2` (area-law entanglement instead of the ZZ
interaction's volume-law growth) trained with the gradient ascent below
until gamma > 0.3.

Reference: Hubregtsen et al. (2022), "Training Quantum Embedding Kernels on
Near-Term Quantum Computers", arXiv:2105.02276.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def compute_kta(K: np.ndarray, y: np.ndarray) -> float:
    """gamma = <K, yy^T>_F / (||K||_F ||yy^T||_F), in [-1, 1]."""
    y_outer = np.outer(y, y).astype(float)
    k_f = np.linalg.norm(K, "fro")
    y_f = np.linalg.norm(y_outer, "fro")
    if k_f < 1e-12 or y_f < 1e-12:
        return 0.0
    return float(np.sum(K * y_outer)) / (k_f * y_f)


def compute_gram_matrix(
    X: np.ndarray, kernel_fn: Callable[[np.ndarray, np.ndarray], float], symmetric: bool = True
) -> np.ndarray:
    m = len(X)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(i if symmetric else 0, m):
            val = kernel_fn(X[i], X[j])
            K[i, j] = val
            if symmetric:
                K[j, i] = val
    return K


def kta_gradient_step(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn_factory: Callable[[np.ndarray], Callable],
    eps: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Central-difference gradient of KTA w.r.t. encoding params."""
    K = compute_gram_matrix(X, kernel_fn_factory(params))
    kta_val = compute_kta(K, y)
    grad = np.zeros_like(params)
    for k in range(len(params)):
        p_plus, p_minus = params.copy(), params.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        k_plus = compute_gram_matrix(X, kernel_fn_factory(p_plus))
        k_minus = compute_gram_matrix(X, kernel_fn_factory(p_minus))
        grad[k] = (compute_kta(k_plus, y) - compute_kta(k_minus, y)) / (2 * eps)
    return grad, kta_val


def kta_gradient_ascent(
    params_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn_factory: Callable[[np.ndarray], Callable],
    n_steps: int = 200,
    lr: float = 0.05,
    target_kta: float = 0.3,
    patience: int = 20,
    verbose: bool = False,
) -> tuple[np.ndarray, float, list[float]]:
    """ADAM gradient ascent on encoding params until gamma >= target_kta,
    or n_steps / patience is exhausted. Returns (best_params, best_kta, history)."""
    params = params_init.copy()
    m_adam = np.zeros_like(params)
    v_adam = np.zeros_like(params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    best_kta = -np.inf
    best_params = params.copy()
    history: list[float] = []
    no_improve = 0

    for step in range(1, n_steps + 1):
        grad, kta_val = kta_gradient_step(params, X, y, kernel_fn_factory)
        history.append(kta_val)

        if kta_val > best_kta:
            best_kta, best_params, no_improve = kta_val, params.copy(), 0
        else:
            no_improve += 1

        if verbose and step % 20 == 0:
            print(f"  KTA step {step}: gamma={kta_val:.4f} (best {best_kta:.4f})")

        if best_kta >= target_kta or no_improve >= patience:
            break

        m_adam = beta1 * m_adam + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
        m_hat = m_adam / (1 - beta1 ** step)
        v_hat = v_adam / (1 - beta2 ** step)
        params = params + lr * m_hat / (np.sqrt(v_hat) + eps_adam)

    return best_params, best_kta, history


def kernel_concentration_measure(K: np.ndarray) -> float:
    """Std over off-diagonal Gram entries -- near-zero means the kernel has
    concentrated to a constant and carries no classification signal."""
    m = K.shape[0]
    off_diag = K[np.triu_indices(m, k=1)]
    return float(np.std(off_diag))
