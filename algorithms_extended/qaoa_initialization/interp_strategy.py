"""INTERP QAOA parameter initialization.

Zhou, Wang, Choi, Pichler, Lukin, Phys. Rev. X 10, 021067 (2020),
arXiv:1812.01041, section III. Original implementation of the published
algorithm (see THIRD_PARTY_NOTICES.md) — operates purely on angle arrays.
"""
from __future__ import annotations

import numpy as np

from core.errors import ValidationError


def interpolate_to_next_depth(gammas: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given optimized angle arrays for QAOA depth p, produce an initial
    guess for depth p+1 by linear interpolation, per the INTERP heuristic:

        gamma_i^(p+1) = ((i-1)/p) * gamma_{i-1}^(p) + ((p-i+1)/p) * gamma_i^(p)

    (and symmetrically for beta), using 1-indexed i = 1..p+1 with
    gamma_0 = gamma_{p+1} = 0 boundary conventions.
    """
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if gammas.shape != betas.shape:
        raise ValidationError("gammas and betas must have the same shape")
    p = len(gammas)
    if p < 1:
        raise ValidationError("Need at least depth-1 angles to interpolate from")

    padded_gamma = np.concatenate([[0.0], gammas, [0.0]])
    padded_beta = np.concatenate([[0.0], betas, [0.0]])

    new_gamma = np.zeros(p + 1)
    new_beta = np.zeros(p + 1)
    for i in range(1, p + 2):  # 1-indexed target layer
        w_prev = (i - 1) / p
        w_next = (p - i + 1) / p
        new_gamma[i - 1] = w_prev * padded_gamma[i - 1] + w_next * padded_gamma[i]
        new_beta[i - 1] = w_prev * padded_beta[i - 1] + w_next * padded_beta[i]
    return new_gamma, new_beta


def random_initialization(p: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Baseline uniform-random initialization, used as the comparator that
    INTERP and linear-ramp are benchmarked against.
    """
    if p < 1:
        raise ValidationError("p must be >= 1")
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0, 2 * np.pi, size=p)
    betas = rng.uniform(0, np.pi, size=p)
    return gammas, betas
