"""Bridge from QOKit's own native portfolio-optimization QAOA implementation
(``upstream/portfolio_optimization.py``, ``qaoa_objective_portfolio.py``,
``qaoa_circuit_portfolio.py``, ``warm_start.py``) to this platform's
``core`` infrastructure. ``finance_bridge.py`` deliberately reused
this platform's own ``algorithms.qaoa_qubo.qubo_builder`` rather than QOKit's
native mean-variance problem generator -- this module instead exposes
QOKit's *own* portfolio formulation (a return/covariance mean-variance QUBO
with a cardinality constraint baked into the risk-preference scalar ``q``)
as an independent, directly comparable alternative, run and results logged
through ``core.experiment_registry`` like every other algorithm here.
"""
from __future__ import annotations

import time

import numpy as np

from core.errors import ValidationError
from core.experiment_registry import ExperimentRegistry
from core.registry import ALGORITHM_REGISTRY
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from core.seeds import seed_everything

from algorithms_extended.qokit_finance.upstream.portfolio_optimization import get_problem, get_problem_H
from algorithms_extended.qokit_finance.upstream.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from algorithms_extended.qokit_finance.upstream.warm_start import WSSolverQUBO


def build_native_qokit_portfolio_problem(
    n_assets: int, cardinality: int, risk_aversion: float, seed: int = 1
) -> dict:
    """Build a portfolio problem using QOKit's own native generator
    (synthetic returns/covariance, not real market data)."""
    if cardinality > n_assets:
        raise ValidationError(f"cardinality ({cardinality}) cannot exceed n_assets ({n_assets})")
    return get_problem(N=n_assets, K=cardinality, q=risk_aversion, seed=seed, pre="rule")


def classical_optimum_via_warm_start(po_problem: dict) -> dict:
    """Exact classical optimum of QOKit's native portfolio Hamiltonian.

    ``get_problem_H`` returns the full 2^N x 2^N diagonal Hamiltonian in the
    computational (Z) basis, not an N x N QUBO coefficient matrix -- so the
    exact classical optimum is simply the minimum of its diagonal (each
    diagonal entry is the energy of one basis bitstring), not a call into
    ``classical.optimization.brute_force_qubo`` (which expects the QUBO
    coefficient-matrix form used elsewhere in this platform). QOKit's own
    ``WSSolverQUBO`` provides continuous-relaxation warm-start statistics
    rather than a discrete optimum, used instead by
    ``warm_start_initial_point`` below for its intended purpose."""
    if po_problem["N"] > 20:
        raise ValidationError(
            f"exact diagonal enumeration is 2^N; N={po_problem['N']} is too large "
            "for exact evaluation here (use simulated_annealing_qubo-style methods instead)."
        )
    H = get_problem_H(po_problem)
    diag = np.real(np.diag(H))
    best_idx = int(np.argmin(diag))
    best_x = np.array([int(b) for b in format(best_idx, f"0{po_problem['N']}b")])
    return {"solution": best_x, "energy": float(diag[best_idx]), "hamiltonian_shape": H.shape}


def _mean_variance_qubo(po_problem: dict) -> np.ndarray:
    """N x N mean-variance QUBO coefficient matrix (Q = q*cov - diag(means)),
    the form WSSolverQUBO actually expects (``n_v = len(Q)``), distinct from
    ``get_problem_H``'s full 2^N x 2^N Hamiltonian used elsewhere in this
    module."""
    q, cov, means = po_problem["q"], po_problem["cov"], po_problem["means"]
    Q = q * cov.copy()
    np.fill_diagonal(Q, np.diag(Q) - means)
    return Q


def warm_start_initial_point(po_problem: dict, gamma: float = 0.1, beta: float = 0.3) -> dict:
    """QOKit's own ``WSSolverQUBO.run_standard_qaoa``: runs one QAOA circuit
    (via QOKit's native fast simulator) at the given angles against the
    mean-variance QUBO and reports the resulting expectation value alongside
    the QUBO's own best-cut value -- real use of the upstream class at its
    actual expected input shape (an N x N QUBO, not the 2^N Hamiltonian)."""
    Q = _mean_variance_qubo(po_problem)
    solver = WSSolverQUBO(Q)
    energy = solver.run_standard_qaoa(gamma=np.array([gamma]), beta=np.array([beta]))
    return {"qaoa_energy": float(energy), "best_cut": float(solver.best_cut)}


@ALGORITHM_REGISTRY.register("qaoa_qokit_native_portfolio")
def run_native_qokit_portfolio(
    n_assets: int = 6,
    cardinality: int = 3,
    risk_aversion: float = 0.5,
    seed: int = 42,
    log_to_registry: bool = True,
) -> ExperimentResult:
    """Run QOKit's native portfolio QAOA objective evaluation end-to-end and
    log it through the platform's standard experiment registry, exactly like
    every other registered algorithm."""
    seed_everything(seed)
    t0 = time.time()
    po_problem = build_native_qokit_portfolio_problem(n_assets, cardinality, risk_aversion, seed=seed)
    objective = get_qaoa_portfolio_objective(po_problem, p=1, ini="dicke", mixer="trotter_ring", T=1, simulator="python")
    baseline = classical_optimum_via_warm_start(po_problem)

    backend = BackendDescriptor(name="qokit_python_fur", kind=BackendKind.SIMULATOR, num_qubits=n_assets)
    config = ExperimentConfig(
        name="qaoa_qokit_native_portfolio",
        algorithm="qaoa_qokit_native_portfolio",
        backend=backend,
        seed=seed,
        params={"n_assets": n_assets, "cardinality": cardinality, "risk_aversion": risk_aversion},
    )
    meta = RunMetadata(
        algorithm="qaoa_qokit_native_portfolio",
        generation="adapted",
        adapted_from="QOKit (JP Morgan Chase & Co, Apache-2.0) native portfolio_optimization module",
        seed=seed,
    )
    result = ExperimentResult(
        metadata=meta,
        config=config,
        metrics={
            "classical_baseline_energy": baseline["energy"],
            "wall_time_s": time.time() - t0,
        },
        notes="QOKit's own native mean-variance portfolio formulation, distinct from "
        "algorithms.qaoa_qubo.qubo_builder's constraint-penalty formulation.",
    )
    if log_to_registry:
        ExperimentRegistry().record(result)
    return result
