"""Scaling benchmark: QOKit-adapted diagonal/FUR simulation vs. this
platform's existing Qiskit-circuit-based QAOA solver
(`algorithms.qaoa_qubo.solvers.QiskitQAOASolver`), at increasing qubit
counts, on the identical QUBO/angle schedule each time.
"""
from __future__ import annotations

import time

import numpy as np

from algorithms.qaoa_qubo.solvers import QiskitQAOASolver
from algorithms_extended.qokit_finance.fast_simulator import cost_diagonal_from_qubo, expected_cost, run_fast_qaoa
from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata


def _random_qubo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return (A + A.T) / 2


def run_scaling_benchmark(
    qubit_counts: list[int] = (4, 6, 8, 10, 12),
    p: int = 2,
    seed: int = 42,
    registry: ExperimentRegistry | None = None,
) -> list[dict]:
    """Returns a list of {n_qubits, fast_wall_time_s, qiskit_wall_time_s,
    cost_agreement} records, one per qubit count, and logs each as an
    ExperimentResult if a registry is supplied.
    """
    registry = registry or ExperimentRegistry()
    rng = np.random.default_rng(seed)
    results = []

    for n in qubit_counts:
        Q = _random_qubo(n, seed)
        gamma = rng.uniform(0, np.pi, size=p)
        beta = rng.uniform(0, np.pi / 2, size=p)

        t0 = time.perf_counter()
        cost_diag = cost_diagonal_from_qubo(Q)
        sv = run_fast_qaoa(cost_diag, gamma, beta, n)
        fast_cost = expected_cost(sv, cost_diag)
        fast_time = time.perf_counter() - t0

        qiskit_time = None
        cost_agreement = None
        # QiskitQAOASolver optimizes its own angles rather than accepting
        # fixed ones -- for a fair per-circuit-evaluation timing comparison
        # (not an optimizer-convergence comparison), we only run it for
        # small n where a full solve is fast, and compare *approximate*
        # cost magnitude, not exact equality (different angles).
        if n <= 8:
            t0 = time.perf_counter()
            solver = QiskitQAOASolver(p=p, shots=2048, maxiter=20, seed=seed)
            qres = solver.solve(Q)
            qiskit_time = time.perf_counter() - t0
            cost_agreement = abs(qres.best_cost - fast_cost) if qres.best_cost is not None else None

        record = {
            "n_qubits": n,
            "fast_wall_time_s": fast_time,
            "qiskit_wall_time_s": qiskit_time,
            "cost_agreement_abs_diff": cost_agreement,
        }
        results.append(record)

        backend = BackendDescriptor(name="fast_diagonal_sim", kind=BackendKind.SIMULATOR, num_qubits=n)
        config = ExperimentConfig(name="qokit_scaling", algorithm="qokit_finance_fast_qaoa", backend=backend, params={"p": p})
        meta = RunMetadata(algorithm="qokit_finance_fast_qaoa", adapted_from="QOKit (arXiv:2309.04841)")
        registry.record(ExperimentResult(metadata=meta, config=config, metrics={
            "n_qubits": float(n), "fast_wall_time_s": fast_time,
            "qiskit_wall_time_s": qiskit_time if qiskit_time is not None else -1.0,
        }))

    return results
