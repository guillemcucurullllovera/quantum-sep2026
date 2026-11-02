"""Exact solver for small QUBO instances: x^T Q x, x in {0,1}^n.

Ground truth for validating QAOA/annealing results on toy problems. Only
practical up to ~20-24 variables (2^n enumeration) — that's the point: it's
the thing everything else must match, not a scalable baseline.
"""
from __future__ import annotations

import itertools

import numpy as np


def qubo_objective(x: np.ndarray, Q: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x)


def brute_force_qubo(Q: np.ndarray, max_vars: int = 24) -> tuple[np.ndarray, float]:
    """Return (best_x, best_objective) minimizing x^T Q x over all x in {0,1}^n.

    Raises ValueError above max_vars to fail loudly rather than hang.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError(f"Q must be square, got {Q.shape}")
    if n > max_vars:
        raise ValueError(
            f"brute_force_qubo is exact-enumeration only; n={n} exceeds "
            f"max_vars={max_vars}. Use simulated_annealing_qubo instead."
        )
    best_x, best_obj = None, np.inf
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=float)
        obj = qubo_objective(x, Q)
        if obj < best_obj:
            best_obj, best_x = obj, x
    return best_x, best_obj
