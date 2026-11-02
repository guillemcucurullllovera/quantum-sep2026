"""Kernel polarity/alignment and matrix post-processing, adapted from
PennyLane's ``pennylane.kernels`` subpackage (Apache-2.0, Xanadu Quantum
Technologies Inc.) — see ``upstream/kernels/`` for the original source and
``THIRD_PARTY_NOTICES.md`` for exactly what changed.

Reimplemented against plain numpy (no PennyLane operator/tape machinery
required) so this runs standalone in this platform's stack. Mathematics is
unchanged from upstream; only the dependency surface was rewritten.

Complements, does not replace, ``algorithms.pqk.kta_alignment`` (this
platform's existing gradient-ascent KTA trainer) and ``core.metrics``'s
single PSD-repair strategy (``nearest_psd``) — this module adds four
distinct PSD-repair strategies plus a kernel-level depolarizing-noise
mitigation technique that exist nowhere else in the platform.
"""
from __future__ import annotations

import numpy as np


def frobenius_inner_product(A: np.ndarray, B: np.ndarray, normalize: bool = False) -> float:
    """Frobenius inner product of two matrices; adapted from
    ``pennylane.math.frobenius_inner_product``."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    ip = float(np.sum(A * B))
    if not normalize:
        return ip
    norm = np.sqrt(np.sum(A * A) * np.sum(B * B))
    return ip / norm if norm > 0 else 0.0


def square_kernel_matrix(X, kernel, assume_normalized_kernel: bool = False) -> np.ndarray:
    """Gram matrix of a dataset with itself; adapted from
    ``pennylane.kernels.utils.square_kernel_matrix``."""
    n = len(X)
    K = np.ones((n, n))
    for i in range(n):
        start = i if assume_normalized_kernel else i
        for j in range(start, n):
            if i == j and assume_normalized_kernel:
                continue
            val = float(kernel(X[i], X[j]))
            K[i, j] = K[j, i] = val
    return K


def kernel_matrix(X1, X2, kernel) -> np.ndarray:
    """Rectangular kernel matrix between two datasets; adapted from
    ``pennylane.kernels.utils.kernel_matrix``."""
    return np.array([[float(kernel(x1, x2)) for x2 in X2] for x1 in X1])


def polarity(
    X, Y, kernel,
    assume_normalized_kernel: bool = False,
    rescale_class_labels: bool = True,
    normalize: bool = False,
) -> float:
    """Kernel polarity: sum_ij y_i y_j k(x_i, x_j). Adapted from
    ``pennylane.kernels.cost_functions.polarity``."""
    K = square_kernel_matrix(X, kernel, assume_normalized_kernel=assume_normalized_kernel)
    Y = np.asarray(Y, dtype=float)
    if rescale_class_labels:
        nplus = int(np.count_nonzero(Y == 1))
        nminus = len(Y) - nplus
        y_scaled = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        y_scaled = Y
    T = np.outer(y_scaled, y_scaled)
    return frobenius_inner_product(K, T, normalize=normalize)


def target_alignment(X, Y, kernel, assume_normalized_kernel: bool = False, rescale_class_labels: bool = True) -> float:
    """Normalized kernel-target alignment; adapted from
    ``pennylane.kernels.cost_functions.target_alignment``."""
    return polarity(
        X, Y, kernel,
        assume_normalized_kernel=assume_normalized_kernel,
        rescale_class_labels=rescale_class_labels,
        normalize=True,
    )


# ---------------------------------------------------------------------------
# PSD-repair strategies, adapted from pennylane.kernels.postprocessing.
# Distinct from core.metrics.nearest_psd (eigenvalue-clip) -- these are
# alternative repair strategies with different bias/variance tradeoffs.
# ---------------------------------------------------------------------------

def threshold_matrix(K: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues to zero (same family as core.metrics.nearest_psd,
    kept here for a uniform API alongside the other three strategies)."""
    K = np.asarray(K, dtype=float)
    K = (K + K.T) / 2
    w, v = np.linalg.eigh(K)
    return v @ np.diag(np.clip(w, 0, None)) @ v.T


def displace_matrix(K: np.ndarray) -> np.ndarray:
    """Shift all eigenvalues up by the magnitude of the most negative one,
    preserving eigenvectors and relative eigenvalue spacing exactly (unlike
    thresholding, which distorts the spectrum non-uniformly)."""
    K = np.asarray(K, dtype=float)
    K = (K + K.T) / 2
    w, v = np.linalg.eigh(K)
    shift = max(0.0, -float(w.min()))
    return v @ np.diag(w + shift) @ v.T


def flip_matrix(K: np.ndarray) -> np.ndarray:
    """Flip the sign of negative eigenvalues rather than clipping or shifting --
    preserves the magnitude of noise-induced negative eigenvalues as signal
    rather than discarding it, appropriate when negativity is believed to be
    symmetric sampling noise rather than a systematic bias."""
    K = np.asarray(K, dtype=float)
    K = (K + K.T) / 2
    w, v = np.linalg.eigh(K)
    return v @ np.diag(np.abs(w)) @ v.T


def closest_psd_matrix(K: np.ndarray, fix_diagonal: bool = False) -> np.ndarray:
    """Closest PSD matrix in Frobenius norm (Higham's method), optionally
    constraining the diagonal to remain 1 (as a normalized kernel requires)."""
    K = np.asarray(K, dtype=float)
    K = (K + K.T) / 2
    w, v = np.linalg.eigh(K)
    K_psd = v @ np.diag(np.clip(w, 0, None)) @ v.T
    if fix_diagonal:
        d = np.sqrt(np.clip(np.diag(K_psd), 1e-12, None))
        K_psd = K_psd / np.outer(d, d)
        np.fill_diagonal(K_psd, 1.0)
    return K_psd


def mitigate_depolarizing_noise(K: np.ndarray, num_wires: int, method: str = "average", use_entries=None) -> np.ndarray:
    """Undo the effect of a global depolarizing channel on a fidelity kernel
    matrix, adapted (math kept faithful to upstream) from
    ``pennylane.kernels.postprocessing.mitigate_depolarizing_noise``
    (Thubregtsen et al. 2022, arXiv:2105.02276, Section V).

    Precondition, important and easy to get wrong: this assumes the DIAGONAL
    ITSELF was measured on noisy hardware (k(x,x) comes back < 1 due to the
    depolarizing channel), not assumed/fixed at the ideal value 1 -- a
    diagonal that's exactly 1 by construction carries no information about
    the noise rate and this function is a no-op on it.

    Genuinely new capability -- nothing else in this platform mitigates
    noise at the kernel-matrix level (existing hardware/mitigation acts on
    circuits before measurement, not on the resulting Gram matrix).

    method: "single" uses one diagonal entry (use_entries[0], default index 0);
    "average" (default) uses the mean of several diagonal entries (use_entries,
    default all); "split_channel" estimates a per-datapoint noise rate from
    every diagonal entry individually -- more accurate when noise strength
    varies across circuits, at the cost of needing every diagonal measured.
    """
    K = np.asarray(K, dtype=float)
    dim = 2**num_wires

    if method == "single":
        idx = 0 if use_entries is None else use_entries[0]
        d = K[idx, idx]
        if d <= 1.0 / dim:
            raise ValueError("diagonal entry too small for single-method noise mitigation")
        noise_rate = (1 - d) * dim / (dim - 1)
        return (K - noise_rate / dim) / (1 - noise_rate)

    if method == "average":
        diag = np.diag(K) if use_entries is None else np.diag(K)[np.array(use_entries)]
        if diag.mean() <= 1.0 / dim:
            raise ValueError("average diagonal too small for average-method noise mitigation")
        noise_rates = (1 - diag) * dim / (dim - 1)
        mean_rate = float(noise_rates.mean())
        return (K - mean_rate / dim) / (1 - mean_rate)

    if method == "split_channel":
        diag = np.diag(K)
        if np.any(diag <= 1.0 / dim):
            raise ValueError("diagonal contains an entry too small for split_channel mitigation")
        eff_rates = np.clip((1 - diag) * dim / (dim - 1), 0.0, 1.0)
        rates = 1 - np.sqrt(1 - eff_rates)
        n = len(K)
        inv_noise = -np.outer(rates, rates) + rates.reshape(1, n) + rates.reshape(n, 1)
        return (K - inv_noise / dim) / (1 - inv_noise)

    raise ValueError(f"unknown method {method!r}: expected 'single', 'average', or 'split_channel'")
