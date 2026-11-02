"""SolverChoice.QP_ASYMMETRICA: models a PortfolioSpec as a real
`QuadraticProgram` (named binary variables, explicit `LinearConstraint`
objects -- cardinality as an equality, turnover/leverage/sector ranges as
inequalities, factor exposure linearized against the cardinality target)
using the full-base `algorithms_extended.qiskit_optimization_asymmetrica`
package (a wholesale, real port of upstream qiskit-optimization), then
solves it with a genuine meta-optimizer from that package rather than this
platform's own penalty-QUBO machinery.

Unlike every other solver generation here, constraints modeled this way are
HARD (structural to the QuadraticProgram and its converter chain), not
penalty terms competing with the objective -- the real difference this
generation is meant to demonstrate. Two constraint kinds this platform's
`ConstraintSpec` vocabulary supports (`risk_budget`, `tracking_error`) are
genuinely quadratic and are NOT added to the QuadraticProgram here: the
standard `QuadraticProgramToQubo` converter chain
(`LinearInequalityToPenalty`/`LinearEqualityToPenalty`) only accepts LINEAR
constraints -- see `converters/quadratic_program_to_qubo.py`. Encoding a
quadratic constraint would require a bespoke quadratic-penalty step this
module does not (yet) build; those two kinds remain validated post-hoc only
(`applications.portfolio.validation`), same as `transaction_cost` already is
platform-wide -- an honest, documented gap, not a silent one.
"""
from __future__ import annotations

import time
from enum import Enum

import numpy as np

from algorithms_extended.qiskit_optimization_asymmetrica import QuadraticProgram
from algorithms_extended.qiskit_optimization_asymmetrica.algorithms.admm_optimizer import ADMMOptimizer
from algorithms_extended.qiskit_optimization_asymmetrica.algorithms.minimum_eigen_optimizer import MinimumEigenOptimizer
from algorithms_extended.qiskit_optimization_asymmetrica.algorithms.recursive_minimum_eigen_optimizer import (
    RecursiveMinimumEigenOptimizer,
)
from algorithms_extended.qiskit_optimization_asymmetrica.algorithms.slsqp_optimizer import SlsqpOptimizer
from algorithms_extended.qiskit_optimization_asymmetrica.algorithms.warm_start_qaoa_optimizer import (
    WarmStartQAOAOptimizer,
)
from algorithms_extended.qiskit_optimization_asymmetrica.minimum_eigensolvers.numpy_minimum_eigensolver import (
    NumPyMinimumEigensolver,
)
from algorithms_extended.qiskit_optimization_asymmetrica.minimum_eigensolvers.qaoa import QAOA
from algorithms_extended.qiskit_optimization_asymmetrica.optimizers import COBYLA
from applications.portfolio.domain import PortfolioSpec
from applications.portfolio.exceptions import SolverFailureError
from core.errors import ValidationError


class QPStrategy(str, Enum):
    EXACT = "exact"                      # NumPyMinimumEigensolver-backed MinimumEigenOptimizer -- ground truth
    RECURSIVE_QAOA = "recursive_qaoa"    # RecursiveMinimumEigenOptimizer wrapping a real Aer-Sampler QAOA
    WARM_START_QAOA = "warm_start_qaoa"  # WarmStartQAOAOptimizer: LP-relaxation-informed QAOA (Egger et al. 2020)
    ADMM = "admm"                        # ADMMOptimizer: alternating-direction heuristic (Gambella & Simonetto 2020)


def build_quadratic_program(spec: PortfolioSpec) -> QuadraticProgram:
    if spec.long_short:
        raise ValidationError(
            "QP_ASYMMETRICA does not model long/short portfolios (the 2n-variable encoding "
            "used elsewhere) -- use MODELED_REPAIR with PortfolioSpec.long_short=True instead."
        )
    spec.validate_shape()
    n = spec.n_assets()
    mu = spec.expected_returns()
    cov = spec.cov()
    q = spec.risk_aversion
    names = spec.asset_names()

    turnover_prev = None
    for c in spec.constraints_of("turnover"):
        turnover_prev = np.asarray(c.params["previous_holdings"], dtype=float)

    linear: dict[str, float] = {names[i]: -(1 - q) * float(mu[i]) for i in range(n)}
    for c in spec.constraints_of("transaction_cost"):
        bps = c.params.get("bps", 0.0)
        if bps > 0 and turnover_prev is not None:
            cost_per_new_trade = bps * 1e-4
            for i in range(n):
                if turnover_prev[i] < 0.5:
                    linear[names[i]] = linear.get(names[i], 0.0) + cost_per_new_trade

    quadratic: dict[tuple[str, str], float] = {}
    for i in range(n):
        for j in range(n):
            if cov[i, j] != 0:
                quadratic[(names[i], names[j])] = q * float(cov[i, j])

    qp = QuadraticProgram(name="portfolio_qp_asymmetrica")
    for name in names:
        qp.binary_var(name=name)
    qp.minimize(linear=linear, quadratic=quadratic)

    cardinality_k = spec.cardinality_k()
    for c in spec.constraints:
        if c.kind == "cardinality":
            qp.linear_constraint(linear={nm: 1 for nm in names}, sense="==", rhs=c.params["k"], name="cardinality")
        elif c.kind == "turnover":
            prev = np.asarray(c.params["previous_holdings"], dtype=float)
            limit = c.params["limit"]
            coeffs = {names[i]: float(1 - 2 * prev[i]) for i in range(n)}
            rhs = limit - float(prev.sum())
            qp.linear_constraint(linear=coeffs, sense="<=", rhs=rhs, name="turnover")
        elif c.kind == "transaction_cost":
            continue  # folded into the linear objective above
        elif c.kind == "sector_range":
            idx_names = c.params["asset_names"]
            lo, hi = c.params.get("min_count", 0), c.params.get("max_count")
            cname = c.params["name"]
            if hi is not None:
                qp.linear_constraint(linear={nm: 1 for nm in idx_names}, sense="<=", rhs=hi, name=f"sector_max_{cname}")
            if lo:
                qp.linear_constraint(linear={nm: 1 for nm in idx_names}, sense=">=", rhs=lo, name=f"sector_min_{cname}")
        elif c.kind == "liquidity_floor":
            floor = c.params["floor"]
            scores = spec.liquidity_scores()
            for i in range(n):
                if scores[i] < floor:
                    qp.linear_constraint(linear={names[i]: 1}, sense="<=", rhs=0, name=f"liquidity_excl_{names[i]}")
        elif c.kind == "leverage":
            qp.linear_constraint(
                linear={nm: 1 for nm in names}, sense="<=", rhs=c.params["max_gross_positions"], name="leverage"
            )
        elif c.kind == "factor_exposure":
            if cardinality_k is None:
                raise ValidationError(
                    "factor_exposure in QP_ASYMMETRICA requires a cardinality constraint to "
                    "linearize per-asset weight under the equal-weight-per-holding assumption."
                )
            loadings = np.asarray(c.params["loadings"], dtype=float)
            coeffs = {names[i]: float(loadings[i]) for i in range(n)}
            lo, hi = c.params.get("min_exposure"), c.params.get("max_exposure")
            if hi is not None:
                qp.linear_constraint(linear=coeffs, sense="<=", rhs=hi * cardinality_k, name="factor_max")
            if lo is not None:
                qp.linear_constraint(linear=coeffs, sense=">=", rhs=lo * cardinality_k, name="factor_min")
        elif c.kind in ("net_exposure", "concentration_limit", "risk_budget", "tracking_error"):
            continue  # quadratic or long/short-only -- validated post-hoc, see module docstring
        else:
            raise ValidationError(f"unknown constraint kind {c.kind!r}")

    return qp


def solve_qp_asymmetrica(
    spec: PortfolioSpec,
    seed: int = 42,
    *,
    strategy: QPStrategy = QPStrategy.EXACT,
    p: int = 1,
    maxiter: int = 60,
) -> tuple[np.ndarray, float, float, dict]:
    """Returns (bitstring, native_qp_objective_cost, wall_time_s, extra_diagnostics)."""
    n = spec.n_assets()
    if strategy == QPStrategy.EXACT and n > 18:
        raise ValidationError(
            f"QP_ASYMMETRICA strategy=EXACT: n={n} assets -- NumPyMinimumEigensolver diagonalizes "
            "a dense 2**n_qubits matrix after QUBO conversion, impractical much beyond ~18 qubits "
            "(and slack variables from inequality constraints can add several more)."
        )
    qp = build_quadratic_program(spec)

    t0 = time.perf_counter()
    try:
        if strategy == QPStrategy.EXACT:
            optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        elif strategy in (QPStrategy.RECURSIVE_QAOA, QPStrategy.WARM_START_QAOA):
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            # AerSamplerV2 (unlike StatevectorSampler) needs the QAOA ansatz
            # transpiled to a basis it recognizes before it can run --
            # without a pass_manager it fails with "unknown instruction:
            # QAOA" (Aer sees the un-decomposed QAOAAnsatz label, not gates).
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
            qaoa = QAOA(sampler=AerSamplerV2(), optimizer=COBYLA(maxiter=maxiter), reps=p, pass_manager=pass_manager)
            if strategy == QPStrategy.RECURSIVE_QAOA:
                optimizer = RecursiveMinimumEigenOptimizer(MinimumEigenOptimizer(qaoa), min_num_vars=2)
            else:
                optimizer = WarmStartQAOAOptimizer(
                    pre_solver=SlsqpOptimizer(), relax_for_pre_solver=True, qaoa=qaoa, epsilon=0.25
                )
        elif strategy == QPStrategy.ADMM:
            optimizer = ADMMOptimizer()
        else:
            raise ValidationError(f"unknown QPStrategy {strategy!r}")
        result = optimizer.solve(qp)
    except (ValidationError,):
        raise
    except Exception as exc:
        raise SolverFailureError(f"QP_ASYMMETRICA (strategy={strategy.value}) failed: {exc}") from exc
    dt = time.perf_counter() - t0

    x = np.asarray(result.x, dtype=float)
    extra = {
        "strategy": strategy.value,
        "status": str(result.status),
        "n_constraints": len(qp.linear_constraints) + len(qp.quadratic_constraints),
    }
    return x, float(result.fval), dt, extra
