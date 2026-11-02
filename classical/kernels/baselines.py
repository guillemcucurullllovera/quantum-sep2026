"""Classical kernel baselines -- the honest comparator every PQK experiment
in this platform should run against. Pure numpy (no sklearn dependency in
this environment); reuses `algorithms.pqk.kernels_gen1.SimpleSVM` as the
classifier since it operates on an arbitrary precomputed Gram matrix and has
no idea whether that matrix came from a quantum circuit or here.
"""
from __future__ import annotations

import numpy as np

from algorithms.pqk.kernels_gen1 import SimpleSVM
from core.metrics import is_positive_semidefinite
from core.registry import ALGORITHM_REGISTRY


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    dist_sq = sq1 + sq2.T - 2 * X1 @ X2.T
    return np.exp(-gamma * np.clip(dist_sq, 0, None))


def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    return (X1 @ X2.T + coef0) ** degree


def linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return X1 @ X2.T


CLASSICAL_KERNELS = {
    "rbf": rbf_kernel,
    "polynomial": polynomial_kernel,
    "linear": linear_kernel,
}


def train_classical_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    **kernel_kwargs,
) -> dict:
    """Same train/eval contract as algorithms.pqk.training.train_gen1_svm,
    so quantum-vs-classical comparisons are apples to apples: same SVM
    solver, same accuracy definition, only the kernel differs."""
    if kernel not in CLASSICAL_KERNELS:
        raise ValueError(f"Unknown classical kernel '{kernel}'. Known: {sorted(CLASSICAL_KERNELS)}")
    kfn = CLASSICAL_KERNELS[kernel]

    y_tr = 2 * y_train.astype(float) - 1
    y_va = 2 * y_val.astype(float) - 1

    K_train = kfn(X_train, X_train, **kernel_kwargs)
    K_val = kfn(X_val, X_train, **kernel_kwargs)
    assert is_positive_semidefinite(K_train) or kernel == "polynomial", (
        "classical Gram matrix should be PSD by construction for rbf/linear; "
        "a failure here indicates a bug in the kernel function, not noise"
    )

    svm = SimpleSVM(C=C)
    n_sv = svm.fit(K_train, y_tr)
    return {
        "kernel": kernel,
        "train_accuracy": svm.accuracy(K_train, y_tr),
        "val_accuracy": svm.accuracy(K_val, y_va),
        "n_support_vectors": n_sv,
    }


ALGORITHM_REGISTRY.register("classical_kernel_svm")(train_classical_svm)
