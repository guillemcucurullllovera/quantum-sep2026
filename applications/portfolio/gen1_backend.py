"""SolverChoice.GEN1_QAOA: routes a PortfolioSpec through Gen1's original,
independent QAOA engine (`algorithms.qaoa_qubo.gen1.qaoa_portfolio`) rather
than the Gen2 NumPy engine `SolverChoice.QAOA_DIRECT` already uses.

Gen1 and Gen2 are architecturally distinct implementations (different
Ising-conversion code, different mixer/init-state construction, different
classical optimizer loop with multi-restart + adiabatic-schedule angle
initialization + amplitude amplification) that happen to solve the same
class of problem -- exactly the kind of "two independently-written encoders"
comparison this platform's multi-solver layer is built around. Before this
module, Gen1 was reachable only by importing `algorithms.qaoa_qubo.gen1`
directly; nothing under `applications/` routed a `PortfolioSpec` to it.
"""
from __future__ import annotations

import time

import numpy as np

from algorithms.qaoa_qubo.gen1.qaoa_portfolio import PortfolioQUBO, QAOACircuit, QAOAOptimizer
from applications.portfolio.domain import PortfolioSpec
from applications.portfolio.exceptions import SolverFailureError
from core.errors import ValidationError


def solve_gen1_qaoa(
    spec: PortfolioSpec,
    seed: int = 42,
    *,
    p_layers: int = 3,
    n_restarts: int = 3,
    maxiter: int = 150,
    mixer: str = "xy",
) -> tuple[np.ndarray, float, float, dict]:
    """Returns (bitstring, qubo_cost, wall_time_s, extra_diagnostics)."""
    spec.validate_shape()
    k = spec.cardinality_k()
    if k is None:
        raise ValidationError("GEN1_QAOA requires a cardinality constraint")

    penalty = 2.0
    for c in spec.constraints_of("cardinality"):
        penalty = float(c.params.get("penalty", penalty))

    qubo = PortfolioQUBO(
        expected_returns=spec.expected_returns(),
        cov_matrix=spec.cov(),
        risk_aversion=spec.risk_aversion,
        cardinality_k=k,
        penalty_lambda=penalty,
        asset_names=spec.asset_names(),
    )
    J, h, _offset = qubo.build_ising()

    n = spec.n_assets()
    if n > 16:
        raise ValidationError(
            f"GEN1_QAOA: n={n} assets -- Gen1's circuit is a dense 2**n statevector "
            "simulation (see algorithms/qaoa_qubo/gen1/qaoa_portfolio.py), impractical "
            "beyond ~16 qubits; use a different solver generation for larger universes."
        )

    circuit = QAOACircuit(n_assets=n, J=J, h=h, p_layers=p_layers, mixer=mixer, cardinality_k=k)
    optimizer = QAOAOptimizer(circuit, qubo, method="COBYLA", maxiter=maxiter, n_restarts=n_restarts)

    t0 = time.perf_counter()
    try:
        outcome = optimizer.optimize()
    except Exception as exc:  # Gen1's own optimizer is third-party-shaped legacy code
        raise SolverFailureError(f"GEN1_QAOA optimizer failed: {exc}") from exc
    dt = time.perf_counter() - t0

    x = np.array([int(b) for b in outcome["best_bitstring"]], dtype=float)
    extra = {
        "approximation_ratio": outcome["approximation_ratio"],
        "n_restarts": outcome["n_restarts"],
        "portfolio": outcome["portfolio"],
        "mixer": mixer,
    }
    return x, float(outcome["best_qubo_cost"]), dt, extra
