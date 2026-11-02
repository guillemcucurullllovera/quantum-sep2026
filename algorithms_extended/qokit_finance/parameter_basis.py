"""Fourier-basis QAOA parameter transfer.

Adapted from QOKit's `qokit/parameter_utils.py` (`to_basis`/`from_basis`/
`extrapolate_parameters_in_fourier_basis`, Apache-2.0, JP Morgan Chase & Co).
Reimplemented against this platform's own dependencies (scipy.fft, already
a project dependency) -- algorithm and DST/DCT-normalization convention
matched to the original, no source lines copied.

The technique (Zhou et al. 2020's INTERP, already adapted in `algorithms_extended/qaoa_initialization/`, is a *different* depth-extension
heuristic; this is complementary): a QAOA angle schedule (gamma_1..gamma_p,
beta_1..beta_p) found to be good for one problem instance is empirically
smooth as a function of layer index for many problem classes. Representing
it in a low-frequency Fourier basis (u, v coefficients, typically far fewer
than p) both (a) gives a compact, transferable representation -- angles
found at depth p can be re-expanded to a different depth p', and angles
found on a small/cheap problem instance can be reused as a warm start on a
larger instance of the same problem class -- and (b) can serve as a reduced
search space for the classical optimizer (optimize a handful of Fourier
coefficients instead of 2p raw angles).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, dst, idct, idst


def to_fourier_basis(gamma: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QAOA (gamma, beta) angle schedule -> (u, v) Fourier coefficients."""
    u = 2 * dst(np.asarray(gamma, dtype=float), type=4, norm="forward")
    v = 2 * dct(np.asarray(beta, dtype=float), type=4, norm="forward")
    return u, v


def from_fourier_basis(u: np.ndarray, v: np.ndarray, p: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) Fourier coefficients -> QAOA (gamma, beta) angle schedule at
    depth `p` (p >= len(u) == len(v); missing high-frequency coefficients
    are zero-padded, which is exactly what makes depth-extension work)."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if len(u) != len(v):
        raise ValueError("u and v must have equal length")
    if p is None:
        p = len(u)
    if p < len(u):
        raise ValueError(f"p={p} must be >= len(u)={len(u)}")
    u_padded = np.zeros(p)
    v_padded = np.zeros(p)
    u_padded[: len(u)] = u
    v_padded[: len(v)] = v
    gamma = 0.5 * idst(u_padded, type=4, norm="forward")
    beta = 0.5 * idct(v_padded, type=4, norm="forward")
    return gamma, beta


def transfer_parameters(gamma: np.ndarray, beta: np.ndarray, target_p: int) -> tuple[np.ndarray, np.ndarray]:
    """Transfer a known-good (gamma, beta) schedule at depth len(gamma) to
    a new target depth, via the Fourier basis. Round-trips through
    to_fourier_basis/from_fourier_basis; when target_p == len(gamma) this
    recovers the original schedule (up to float error), which is what the
    round-trip test in this module's test suite checks.
    """
    if len(gamma) != len(beta):
        raise ValueError("gamma and beta must have equal length")
    u, v = to_fourier_basis(gamma, beta)
    return from_fourier_basis(u, v, p=target_p)
