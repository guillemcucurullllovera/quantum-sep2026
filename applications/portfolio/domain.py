"""Typed domain models for the portfolio application layer.

Callers should build a `PortfolioSpec` + `ConstraintSpec` list instead of
hand-assembling raw QUBO matrices or a `PortfolioConstraints` dataclass
directly -- those remain the underlying wire format each solver backend
consumes, but this is the intended user-facing entry point, matching
`core.schemas`'s pydantic style.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from core.errors import ValidationError


class SolverChoice(str, Enum):
    """Every solver architecture reachable through this platform's portfolio
    application layer, spanning five algorithm generations plus a classical
    comparator -- one enum, one `solve_portfolio`/`compare_solvers` call
    convention, regardless of which underlying package actually does the work.
    """

    QAOA_DIRECT = "qaoa_direct"          # Gen2 NumPy engine: raw QUBO matrix + correlation-XY-mixer QAOA
    GEN1_QAOA = "gen1_qaoa"              # Gen1's original hand-rolled QAOA engine (algorithms/qaoa_qubo/gen1)
    QOKIT_FAST = "qokit_fast"            # QOKit-adapted diagonal simulator + SK/Fourier warm start
    MODELED_REPAIR = "modeled_repair"    # qiskit-optimization-finance: model + converter + feasibility repair
    QP_ASYMMETRICA = "qp_asymmetrica"    # qiskit-optimization-asymmetrica: real QuadraticProgram + meta-optimizers
    XY_MIXER_QAOA = "xy_mixer_qaoa"      # cardinality-preserving XY-mixer QAOA circuit, real Aer execution
    TRAINING_PIPELINE_TQA = "training_pipeline_tqa"  # IBM qaoa-training-pipeline TQA trainer + statevector evaluator
    CLASSICAL_BASELINE = "classical_baseline"        # Markowitz min-variance projected to cardinality-k, no quantum step
    ALL = "all"                          # run every solver above and compare


class ConstraintSpec(BaseModel):
    """One named constraint. Recognized `kind` values and their `params`:

    - cardinality        {k: int}                                  -- hard, every solver
    - turnover           {previous_holdings: list[float], limit: int} -- hard (bitstring solvers + QP_ASYMMETRICA)
    - transaction_cost   {bps: float}                               -- soft objective term only (never a hard constraint)
    - sector_range       {name, asset_names: list[str], min_count, max_count} -- hard; also used for categorical
                          "sector/factor" groupings (equities/bonds/... or a discrete factor bucket)
    - liquidity_floor    {floor: float}                             -- hard: no held asset may score below floor
    - leverage           {max_gross_positions: int}                 -- hard: total held (long+short) positions <= limit
    - net_exposure       {target: float}                            -- long/short only; requires PortfolioSpec.long_short=True
    - concentration_limit {max_weight: float}                       -- validated post-hoc (equal-weight-per-holding assumption)
    - risk_budget        {max_variance: float}                      -- validated post-hoc (portfolio variance ceiling)
    - tracking_error     {benchmark_weights: list[float], max_te: float} -- validated post-hoc
    - factor_exposure    {loadings: list[float], min_exposure, max_exposure} -- hard in QP_ASYMMETRICA (needs
                          cardinality set to linearize per-asset weight), validated post-hoc otherwise

    Every kind is checked post-hoc by `applications.portfolio.validation`
    regardless of whether the solver that produced the bitstring encodes it
    as a hard constraint -- see that module's docstring for why "the solver
    claims feasibility" is never trusted on its own.
    """

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class AssetSpec(BaseModel):
    name: str
    expected_return: float
    liquidity_score: float | None = None


class PortfolioSpec(BaseModel):
    """A portfolio optimization problem: assets + covariance + constraints,
    independent of which solver backend eventually handles it."""

    assets: list[AssetSpec]
    cov_matrix: list[list[float]]
    risk_aversion: float = 0.5
    long_short: bool = False
    constraints: list[ConstraintSpec] = Field(default_factory=list)

    @field_validator("cov_matrix")
    @classmethod
    def _check_square(cls, v: list[list[float]], info) -> list[list[float]]:
        n = len(v)
        if any(len(row) != n for row in v):
            raise ValueError("cov_matrix must be square")
        return v

    def n_assets(self) -> int:
        return len(self.assets)

    def asset_names(self) -> list[str]:
        return [a.name for a in self.assets]

    def expected_returns(self) -> np.ndarray:
        return np.array([a.expected_return for a in self.assets], dtype=float)

    def cov(self) -> np.ndarray:
        return np.asarray(self.cov_matrix, dtype=float)

    def validate_shape(self) -> None:
        n = self.n_assets()
        cov = self.cov()
        if cov.shape != (n, n):
            raise ValidationError(
                f"PortfolioSpec: {n} assets but cov_matrix is {cov.shape}, must be ({n}, {n})"
            )

    def constraints_of(self, kind: str) -> list[ConstraintSpec]:
        return [c for c in self.constraints if c.kind == kind]

    def cardinality_k(self) -> int | None:
        cs = self.constraints_of("cardinality")
        return cs[0].params["k"] if cs else None

    def liquidity_scores(self) -> np.ndarray:
        return np.array(
            [a.liquidity_score if a.liquidity_score is not None else 0.0 for a in self.assets],
            dtype=float,
        )

    def with_turnover(self, previous_holdings: np.ndarray, limit: int) -> "PortfolioSpec":
        """Return a copy of this spec with a turnover constraint referencing
        `previous_holdings` -- the mechanism walk-forward rebalancing
        (`applications.portfolio.scenario`) uses to carry the PRIOR period's
        holdings into the NEXT period's problem instance."""
        new_constraints = [c for c in self.constraints if c.kind != "turnover"]
        new_constraints.append(ConstraintSpec(
            kind="turnover",
            params={"previous_holdings": list(np.asarray(previous_holdings, dtype=float)), "limit": limit},
        ))
        return self.model_copy(update={"constraints": new_constraints})


@dataclass
class PortfolioResult:
    """Canonical result shape every solver backend normalizes into --
    callers never need to know which of the six solver architectures under
    `applications.portfolio` produced a given result, only that it has these
    fields. `feasible`/`violations` are always computed by an independent
    post-hoc check (`applications.portfolio.validation.validate_bitstring`),
    never taken on a solver's own say-so.
    """

    solver: str
    bitstring: np.ndarray
    cost: float
    wall_time_s: float
    extra: dict = field(default_factory=dict)
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
