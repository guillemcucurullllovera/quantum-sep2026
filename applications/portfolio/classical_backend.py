"""SolverChoice.CLASSICAL_BASELINE: a real, deterministic, non-quantum
comparator that never touches a QUBO, an Ising Hamiltonian, or a quantum
circuit -- distinct from the simulated-annealing/brute-force comparators
already embedded inside other generations (those still optimize the *same*
penalty-QUBO objective the quantum solvers do; this instead solves the
literal continuous Markowitz problem and projects it to a discrete
cardinality-k selection). Existing in this platform as
`classical.portfolio.markowitz`; this module is the first thing that wires
it into `applications.portfolio`'s one-interface solver dispatch.
"""
from __future__ import annotations

import time

import numpy as np

from applications.portfolio.domain import PortfolioSpec
from classical.portfolio.markowitz import markowitz_min_variance
from core.errors import ValidationError


def solve_classical_baseline(
    spec: PortfolioSpec, seed: int = 42
) -> tuple[np.ndarray, float, float, dict]:
    """Solve the continuous relaxation (min-variance, long-only, sum(w)=1)
    exactly via SLSQP, then project to a hard cardinality-k selection by
    keeping the k largest weights and re-normalizing -- a standard,
    honest heuristic for turning a continuous Markowitz solution into a
    discrete-holdings portfolio, not a disguised quantum method."""
    spec.validate_shape()
    k = spec.cardinality_k()
    if k is None:
        raise ValidationError("CLASSICAL_BASELINE requires a cardinality constraint")

    t0 = time.perf_counter()
    cov = spec.cov()
    w_continuous = markowitz_min_variance(cov, long_only=True)

    top_k = np.argsort(-w_continuous)[:k]
    x = np.zeros(spec.n_assets(), dtype=float)
    x[top_k] = 1.0
    dt = time.perf_counter() - t0

    mu = spec.expected_returns()
    q = spec.risk_aversion
    # same mean-variance objective convention as PortfolioQUBOBuilder's base
    # term, so its cost is directly comparable to every QUBO-based solver's
    cost = float(q * x @ cov @ x - (1 - q) * mu @ x)

    extra = {
        "continuous_weights": w_continuous.tolist(),
        "method": "markowitz_min_variance + top-k projection",
    }
    return x, cost, dt, extra
