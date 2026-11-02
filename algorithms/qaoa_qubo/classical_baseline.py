"""Classical comparators every QAOA/QUBO result is checked against.

Per the platform's benchmarking policy (see benchmarks/), no quantum result
should be reported without a classical baseline run on the same problem
instance.
"""
from __future__ import annotations

import itertools

import numpy as np


def brute_force_qubo(Q: np.ndarray, max_n: int = 20) -> tuple[np.ndarray, float]:
    """Exact global minimum of x^T Q x over x in {0,1}^n. Only feasible for
    small n; used to compute QAOA approximation ratios in tests."""
    n = Q.shape[0]
    if n > max_n:
        raise ValueError(f"brute_force_qubo: n={n} exceeds max_n={max_n} (2^n enumeration)")

    best_x, best_cost = None, float("inf")
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        cost = float(x @ Q @ x)
        if cost < best_cost:
            best_cost, best_x = cost, x
    return best_x, best_cost


def brute_force_k_hot(Q: np.ndarray, k: int, max_n: int = 24) -> tuple[np.ndarray, float]:
    """Exact minimum restricted to exactly-k-hot bitstrings (for cardinality-
    constrained problems where full 2^n enumeration is unnecessary)."""
    n = Q.shape[0]
    if n > max_n:
        raise ValueError(f"brute_force_k_hot: n={n} exceeds max_n={max_n}")

    best_x, best_cost = None, float("inf")
    for combo in itertools.combinations(range(n), k):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        cost = float(x @ Q @ x)
        if cost < best_cost:
            best_cost, best_x = cost, x
    return best_x, best_cost


def simulated_annealing_qubo(
    Q: np.ndarray,
    *,
    n_sweeps: int = 2000,
    initial_temp: float = 10.0,
    final_temp: float = 0.01,
    seed: int = 0,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Simulated annealing for QUBO minimization (bit-flip neighborhood,
    geometric cooling schedule, Metropolis acceptance). A real, general-
    purpose baseline that scales to problem sizes brute force cannot reach.
    """
    n = Q.shape[0]
    rng = np.random.default_rng(seed)
    x = (rng.random(n) > 0.5).astype(float) if x0 is None else np.asarray(x0, dtype=float).copy()

    def cost(v: np.ndarray) -> float:
        return float(v @ Q @ v)

    cur_cost = cost(x)
    best_x, best_cost = x.copy(), cur_cost
    history = [cur_cost]

    cooling_rate = (final_temp / initial_temp) ** (1.0 / max(n_sweeps, 1))
    temp = initial_temp

    for _ in range(n_sweeps):
        i = rng.integers(0, n)
        x_new = x.copy()
        x_new[i] = 1.0 - x_new[i]
        new_cost = cost(x_new)
        delta = new_cost - cur_cost

        if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            x, cur_cost = x_new, new_cost
            if cur_cost < best_cost:
                best_x, best_cost = x.copy(), cur_cost

        history.append(cur_cost)
        temp *= cooling_rate

    return best_x, best_cost, history


def simulated_annealing_k_hot(
    Q: np.ndarray,
    k: int,
    *,
    n_sweeps: int = 2000,
    initial_temp: float = 10.0,
    final_temp: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """Cardinality-preserving simulated annealing: neighborhood moves swap
    which asset is "in" vs "out" rather than flipping a single bit, so
    every visited state stays exactly k-hot (mirrors the XY-mixer's
    Hamming-weight-preserving search space in the classical baseline)."""
    n = Q.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k={k} must be in [1, {n}]")

    rng = np.random.default_rng(seed)
    selected = list(rng.choice(n, size=k, replace=False))
    x = np.zeros(n)
    x[selected] = 1.0

    def cost(v: np.ndarray) -> float:
        return float(v @ Q @ v)

    cur_cost = cost(x)
    best_x, best_cost = x.copy(), cur_cost
    history = [cur_cost]

    cooling_rate = (final_temp / initial_temp) ** (1.0 / max(n_sweeps, 1))
    temp = initial_temp
    not_selected = [i for i in range(n) if i not in selected]

    for _ in range(n_sweeps):
        if not selected or not not_selected:
            break
        out_idx = selected[rng.integers(0, len(selected))]
        in_idx = not_selected[rng.integers(0, len(not_selected))]

        x_new = x.copy()
        x_new[out_idx], x_new[in_idx] = 0.0, 1.0
        new_cost = cost(x_new)
        delta = new_cost - cur_cost

        if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            x, cur_cost = x_new, new_cost
            selected.remove(out_idx)
            selected.append(in_idx)
            not_selected.remove(in_idx)
            not_selected.append(out_idx)
            if cur_cost < best_cost:
                best_x, best_cost = x.copy(), cur_cost

        history.append(cur_cost)
        temp *= cooling_rate

    return best_x, best_cost, history
