"""Wall-time scaling of two algorithm families as problem size grows.
Different scope from a sibling agent's QOKit-specific scaling benchmark
(if built): this covers the platform's existing NumpyQAOASolver and
QuantumReservoir generically, not QOKit's fast-simulation path specifically.
"""
from __future__ import annotations

import time

import numpy as np

from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints, PortfolioQUBOBuilder
from algorithms.qaoa_qubo.solvers import NumpyQAOASolver
from algorithms.qrc.reservoir import QuantumReservoir, QuantumReservoirConfig
from applications.portfolio.synthetic_market import generate_synthetic_market


def qaoa_scaling(sizes: list[int] = (4, 6, 8, 10), seed: int = 0) -> list[dict]:
    out = []
    for n in sizes:
        mu, sigma, _ = generate_synthetic_market(n, seed=seed)
        Q = PortfolioQUBOBuilder(mu, sigma).build(PortfolioConstraints(cardinality_k=max(1, n // 2)))
        t0 = time.perf_counter()
        result = NumpyQAOASolver(p=2, n_iter=50, seed=seed).solve(Q)
        wall = time.perf_counter() - t0
        out.append({"n_qubits": n, "wall_time_s": wall, "best_cost": result.best_cost})
    return out


def qrc_scaling(reservoir_sizes: list[int] = (2, 4, 6, 8), seed: int = 0) -> list[dict]:
    out = []
    for n2 in reservoir_sizes:
        cfg = QuantumReservoirConfig(n1=3, n2=n2, seed=seed)
        t0 = time.perf_counter()
        reservoir = QuantumReservoir(cfg).build()
        seq = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=(20, cfg.n1))
        reservoir.sequence_features(seq)
        wall = time.perf_counter() - t0
        out.append({"n_qubits": cfg.n, "wall_time_s": wall})
    return out
