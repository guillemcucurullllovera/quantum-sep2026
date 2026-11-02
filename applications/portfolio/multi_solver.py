"""Multi-backend portfolio solving: the same PortfolioSpec can be routed
through eight genuinely different solver architectures built in this
platform, through one call convention (`solve_portfolio` / `compare_solvers`):

- SolverChoice.QAOA_DIRECT          -- Gen2 NumPy engine: raw QUBO + correlation-XY-mixer QAOA
- SolverChoice.GEN1_QAOA            -- Gen1's original hand-rolled QAOA engine
- SolverChoice.QOKIT_FAST           -- QOKit-adapted diagonal simulator + SK/Fourier parameter transfer
- SolverChoice.MODELED_REPAIR       -- qiskit-optimization-finance: model + converter + feasibility repair
- SolverChoice.QP_ASYMMETRICA       -- qiskit-optimization-asymmetrica: real QuadraticProgram + meta-optimizers
- SolverChoice.XY_MIXER_QAOA        -- cardinality-preserving XY-mixer QAOA circuit, real Aer execution
- SolverChoice.TRAINING_PIPELINE_TQA -- IBM qaoa-training-pipeline TQA trainer + statevector evaluator
- SolverChoice.CLASSICAL_BASELINE   -- Markowitz min-variance projected to cardinality-k, no quantum step at all

This module does not reimplement any solver -- it translates a
`PortfolioSpec` into whichever wire format each backend already expects and
normalizes every result into one `PortfolioResult` shape. Every result's
`cost` field is recomputed against ONE shared reference QUBO
(`PortfolioQUBOBuilder` over the spec's constraints) regardless of which
architecture produced the bitstring, so costs are directly comparable across
generations even though each solver internally optimizes a differently
-encoded objective (a solver's own native objective value, which may include
architecture-specific penalty/repair effects or be on an entirely different
scale -- e.g. QP_ASYMMETRICA's hard-constrained objective has no penalty term
at all -- is preserved in `extra["native_cost"]`, never discarded). Every
result's `feasible`/`violations` fields come from an independent post-hoc
check, never from the solver's own claim (see `validation.py`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints, PortfolioQUBOBuilder, SectorLimit
from algorithms.qaoa_qubo.solvers import NumpyQAOASolver
from algorithms_extended.qiskit_optimization_finance.converters import FixedPenaltyStrategy
from algorithms_extended.qiskit_optimization_finance.model import FinancePortfolioModel
from algorithms_extended.qiskit_optimization_finance.runner import solve_model
from algorithms_extended.qokit_finance.finance_bridge import solve_portfolio_fast
from applications.portfolio.domain import PortfolioResult, PortfolioSpec, SolverChoice
from core.errors import ValidationError
from core.schemas import BenchmarkResult

# Constraint kinds that (a) apply only in the long/short 2n-variable encoding
# (net_exposure), or (b) are genuinely quadratic / weight-dependent and are
# validated post-hoc only, never encoded as a hard constraint by the
# long-only bitstring solvers below -- see domain.py's ConstraintSpec
# docstring for the full vocabulary and applications.portfolio.qp_backend's
# module docstring for exactly why risk_budget/tracking_error can't be
# folded into QP_ASYMMETRICA's converter chain either.
_POST_HOC_ONLY_KINDS = frozenset({"concentration_limit", "risk_budget", "tracking_error", "factor_exposure"})


def _to_portfolio_constraints(spec: PortfolioSpec) -> PortfolioConstraints:
    kwargs: dict = {}
    sector_limits: list[SectorLimit] = []
    for c in spec.constraints:
        if c.kind == "cardinality":
            kwargs["cardinality_k"] = c.params["k"]
        elif c.kind == "turnover":
            kwargs["previous_holdings"] = np.asarray(c.params["previous_holdings"], dtype=float)
            kwargs["turnover_limit"] = c.params["limit"]
        elif c.kind == "transaction_cost":
            kwargs["transaction_cost_bps"] = c.params["bps"]
        elif c.kind == "sector_range":
            idx = [spec.asset_names().index(n) for n in c.params["asset_names"]]
            sector_limits.append(
                SectorLimit(name=c.params["name"], asset_indices=idx,
                            min_count=c.params.get("min_count", 0), max_count=c.params.get("max_count"))
            )
        elif c.kind == "liquidity_floor":
            kwargs["liquidity_scores"] = spec.liquidity_scores()
            kwargs["liquidity_floor"] = c.params["floor"]
        elif c.kind == "leverage":
            # No dedicated field on PortfolioConstraints for a bare gross-
            # position cap; a SectorLimit over every asset index with
            # max_count=limit is an exact, honest encoding of the same
            # constraint (reuses tested machinery instead of adding a
            # parallel penalty term for the same thing).
            sector_limits.append(SectorLimit(
                name="leverage", asset_indices=list(range(spec.n_assets())),
                min_count=0, max_count=c.params["max_gross_positions"],
            ))
        elif c.kind == "net_exposure":
            continue  # long/short-only; meaningless for a long-only bitstring encoding
        elif c.kind in _POST_HOC_ONLY_KINDS:
            continue
        else:
            raise ValidationError(f"unknown constraint kind {c.kind!r}")
    if sector_limits:
        kwargs["sector_limits"] = sector_limits
    return PortfolioConstraints(**kwargs)


def _to_finance_model(spec: PortfolioSpec) -> FinancePortfolioModel:
    m = FinancePortfolioModel(
        spec.asset_names(), spec.expected_returns(), spec.cov(),
        risk_aversion=spec.risk_aversion, long_short=spec.long_short,
    )
    for c in spec.constraints:
        if c.kind == "cardinality":
            m.add_cardinality(c.params["k"])
        elif c.kind == "turnover":
            m.add_turnover(np.asarray(c.params["previous_holdings"], dtype=float), c.params["limit"])
        elif c.kind == "transaction_cost":
            m.add_transaction_cost(c.params["bps"])
        elif c.kind == "sector_range":
            m.add_sector_range(c.params["name"], c.params["asset_names"],
                                c.params.get("min_count", 0), c.params.get("max_count"))
        elif c.kind == "liquidity_floor":
            m.add_liquidity_floor(spec.liquidity_scores(), c.params["floor"])
        elif c.kind == "leverage":
            if spec.long_short:
                m.add_leverage_limit(c.params["max_gross_positions"])
            else:
                m.add_sector_range("leverage", spec.asset_names(), 0, c.params["max_gross_positions"])
        elif c.kind == "net_exposure":
            m.add_net_exposure_target(c.params["target"])
        elif c.kind in _POST_HOC_ONLY_KINDS:
            continue
        else:
            raise ValidationError(f"unknown constraint kind {c.kind!r}")
    return m


SingleSolveResult = PortfolioResult  # backward-compatible alias: same type, richer fields


def _finalize(spec: PortfolioSpec, name: str, x: np.ndarray, native_cost: float, dt: float, extra: dict) -> PortfolioResult:
    from applications.portfolio.validation import validate_bitstring

    x = np.asarray(x, dtype=float)
    extra = dict(extra)
    extra.setdefault("native_cost", native_cost)

    n = spec.n_assets()
    if len(x) == n:
        constraints = _to_portfolio_constraints(spec)
        Q = PortfolioQUBOBuilder(spec.expected_returns(), spec.cov(), risk_aversion=spec.risk_aversion).build(constraints)
        cost = float(x @ Q @ x)
    else:
        # long/short 2n-variable encoding: no single n x n reference QUBO
        # applies, so the solver's own native objective is reported as-is.
        cost = native_cost

    v = validate_bitstring(spec, x)
    return PortfolioResult(name, x, cost, dt, extra, feasible=v.ok, violations=v.violations)


def _solve_qaoa_direct(spec: PortfolioSpec, seed: int) -> PortfolioResult:
    spec.validate_shape()
    constraints = _to_portfolio_constraints(spec)
    if constraints.cardinality_k is None:
        raise ValidationError("QAOA_DIRECT requires a cardinality constraint")
    builder = PortfolioQUBOBuilder(spec.expected_returns(), spec.cov(), risk_aversion=spec.risk_aversion)
    Q = builder.build(constraints)
    t0 = time.perf_counter()
    solver = NumpyQAOASolver(p=2, n_iter=150, seed=seed)
    result = solver.solve(Q, cardinality_k=constraints.cardinality_k)
    dt = time.perf_counter() - t0
    x = np.array([int(b) for b in result.best_bitstring], dtype=float)
    return _finalize(spec, "qaoa_direct", x, float(x @ Q @ x), dt, {"approximation_ratio": result.approximation_ratio})


def _solve_qokit_fast(spec: PortfolioSpec, seed: int) -> PortfolioResult:
    spec.validate_shape()
    constraints = _to_portfolio_constraints(spec)
    if constraints.cardinality_k is None:
        raise ValidationError("QOKIT_FAST requires a cardinality constraint")
    t0 = time.perf_counter()
    r = solve_portfolio_fast(spec.expected_returns(), spec.cov(), constraints, risk_aversion=spec.risk_aversion)
    dt = time.perf_counter() - t0
    x = (r.probabilities.argmax())
    n = spec.n_assets()
    bits = np.array([(x >> i) & 1 for i in range(n)], dtype=float)[::-1]
    return _finalize(spec, "qokit_fast", bits, float(r.expected_cost), dt, {"gamma": r.gamma.tolist(), "beta": r.beta.tolist()})


def _solve_modeled_repair(spec: PortfolioSpec, seed: int) -> PortfolioResult:
    model = _to_finance_model(spec)
    for c in spec.constraints:
        if c.kind == "cardinality":
            model.add_cardinality(c.params["k"])
    t0 = time.perf_counter()
    solver = NumpyQAOASolver(p=2, n_iter=150, seed=seed)
    result = solve_model(model, FixedPenaltyStrategy(), solver, repair=True, seed=seed)
    dt = time.perf_counter() - t0
    return _finalize(spec, "modeled_repair", result.bitstring, result.cost, dt,
                      {"feasibility": result.feasibility, "was_repaired": result.was_repaired})


def _solve_gen1_qaoa(spec: PortfolioSpec, seed: int) -> PortfolioResult:
    from applications.portfolio.gen1_backend import solve_gen1_qaoa

    x, cost, dt, extra = solve_gen1_qaoa(spec, seed)
    return _finalize(spec, "gen1_qaoa", x, cost, dt, extra)


def _solve_qp_asymmetrica(spec: PortfolioSpec, seed: int) -> PortfolioResult:
    from applications.portfolio.qp_backend import solve_qp_asymmetrica

    x, cost, dt, extra = solve_qp_asymmetrica(spec, seed)
    return _finalize(spec, "qp_asymmetrica", x, cost, dt, extra)


def _solve_xy_mixer_qaoa(spec: PortfolioSpec, seed: int) -> PortfolioResult:
    from applications.portfolio.xy_mixer_backend import solve_xy_mixer_qaoa

    x, cost, dt, extra = solve_xy_mixer_qaoa(spec, seed)
    return _finalize(spec, "xy_mixer_qaoa", x, cost, dt, extra)


def _solve_training_pipeline_tqa(spec: PortfolioSpec, seed: int) -> PortfolioResult:
    from applications.portfolio.training_pipeline_backend import solve_training_pipeline_tqa

    x, cost, dt, extra = solve_training_pipeline_tqa(spec, seed)
    return _finalize(spec, "training_pipeline_tqa", x, cost, dt, extra)


def _solve_classical_baseline(spec: PortfolioSpec, seed: int) -> PortfolioResult:
    from applications.portfolio.classical_backend import solve_classical_baseline

    x, cost, dt, extra = solve_classical_baseline(spec, seed)
    return _finalize(spec, "classical_baseline", x, cost, dt, extra)


_SOLVERS = {
    SolverChoice.QAOA_DIRECT: _solve_qaoa_direct,
    SolverChoice.GEN1_QAOA: _solve_gen1_qaoa,
    SolverChoice.QOKIT_FAST: _solve_qokit_fast,
    SolverChoice.MODELED_REPAIR: _solve_modeled_repair,
    SolverChoice.QP_ASYMMETRICA: _solve_qp_asymmetrica,
    SolverChoice.XY_MIXER_QAOA: _solve_xy_mixer_qaoa,
    SolverChoice.TRAINING_PIPELINE_TQA: _solve_training_pipeline_tqa,
    SolverChoice.CLASSICAL_BASELINE: _solve_classical_baseline,
}

_DEFAULT_COMPARISON_CHOICES = (SolverChoice.QAOA_DIRECT, SolverChoice.QOKIT_FAST, SolverChoice.MODELED_REPAIR)
ALL_SOLVER_CHOICES: tuple[SolverChoice, ...] = tuple(_SOLVERS)


def solve_portfolio(spec: PortfolioSpec, choice: SolverChoice, seed: int = 42) -> PortfolioResult:
    if choice == SolverChoice.ALL:
        raise ValidationError("solve_portfolio: use compare_solvers() for SolverChoice.ALL")
    return _SOLVERS[choice](spec, seed)


@dataclass
class MultiSolverComparison:
    results: dict[str, PortfolioResult]
    benchmarks: list[BenchmarkResult]
    best: str
    best_feasible: str | None = None


def compare_solvers(
    spec: PortfolioSpec, seed: int = 42, choices: list[SolverChoice] | None = None
) -> MultiSolverComparison:
    """Run every requested solver backend on the identical PortfolioSpec and
    produce pairwise BenchmarkResults plus a best-by-cost verdict. Real runs,
    not mocks -- each backend independently builds its own QUBO/model/circuit
    and solves it. `choices` defaults to the original three-way comparison
    (QAOA_DIRECT, QOKIT_FAST, MODELED_REPAIR); pass
    `applications.portfolio.multi_solver.ALL_SOLVER_CHOICES` (or any subset)
    for the full six-generation-plus-classical comparison.
    """
    resolved = list(choices) if choices is not None else list(_DEFAULT_COMPARISON_CHOICES)
    results = {c.value: _SOLVERS[c](spec, seed) for c in resolved}

    names = list(results)
    benchmarks = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]], results[names[j]]
            benchmarks.append(BenchmarkResult(
                method_a=a.solver, method_b=b.solver, metric_name="qubo_cost",
                value_a=a.cost, value_b=b.cost,
                wall_time_a_s=a.wall_time_s, wall_time_b_s=b.wall_time_s,
                seed=seed,
            ))

    best = min(results, key=lambda k: results[k].cost)
    feasible_names = [n for n in results if results[n].feasible]
    best_feasible = min(feasible_names, key=lambda k: results[k].cost) if feasible_names else None
    return MultiSolverComparison(results=results, benchmarks=benchmarks, best=best, best_feasible=best_feasible)
