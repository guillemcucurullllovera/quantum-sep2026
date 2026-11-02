"""Simulated annealing for QUBO problems: x^T Q x, x in {0,1}^n.

A real annealer — geometric temperature schedule, Metropolis acceptance,
single-bit-flip proposals — not a disguised random search. Used both as a
classical baseline for QAOA benchmarking and as a scalable fallback where
brute_force_qubo's exact enumeration is infeasible.
"""
from __future__ import annotations

import numpy as np

from classical.optimization.brute_force import qubo_objective


def simulated_annealing_qubo(
    Q: np.ndarray,
    *,
    n_iters: int = 5000,
    t_start: float = 10.0,
    t_end: float = 1e-3,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]

    x = rng.integers(0, 2, size=n).astype(float)
    obj = qubo_objective(x, Q)
    best_x, best_obj = x.copy(), obj

    # geometric cooling schedule
    ratio = (t_end / t_start) ** (1.0 / max(n_iters - 1, 1))
    t = t_start

    for _ in range(n_iters):
        i = rng.integers(0, n)
        x_new = x.copy()
        x_new[i] = 1.0 - x_new[i]
        obj_new = qubo_objective(x_new, Q)
        delta = obj_new - obj

        if delta <= 0 or rng.random() < np.exp(-delta / max(t, 1e-12)):
            x, obj = x_new, obj_new
            if obj < best_obj:
                best_x, best_obj = x.copy(), obj

        t *= ratio

    return best_x, best_obj
