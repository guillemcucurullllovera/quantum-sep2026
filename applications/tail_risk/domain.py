"""Typed domain models for the tail-risk application layer.

`TailRiskSpec` is the intended caller-facing entry point (confidence levels,
horizon, an optional position/portfolio definition, and every knob the
multi-method comparison in `applications.tail_risk.comparison` needs) --
matching `core.schemas`'s pydantic style, the same way
`applications.portfolio.domain.PortfolioSpec` does for the portfolio
application.

`MethodResult` / `TailRiskComparisonReport` are the normalized result shape
every one of the five (or six, with the quantum method) VaR/CVaR estimators
in this package reports into, so a caller diffing methods against each
other is diffing one schema, not five ad hoc return types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DistributionAssumption(str, Enum):
    """What `pnl_samples` is taken to represent. EMPIRICAL is the default and
    the only assumption every method in this package can honor directly;
    GAUSSIAN documents that the caller intends the sample as noisy draws
    around a Gaussian ground truth (informational only today -- every
    estimator still reads the empirical sample, exactly like the platform's
    other 'distribution assumption' fields document intent without forcing
    a synthetic-generation code path).
    """

    EMPIRICAL = "empirical"
    GAUSSIAN = "gaussian"


class PositionSpec(BaseModel):
    """One position contributing to the P&L sample under assessment. Purely
    descriptive metadata at this platform (the comparison operates on the
    already-aggregated `pnl_samples` array) -- present so a caller can
    attach a real portfolio/position definition to a `TailRiskSpec` and have
    it flow through into reports/registry logging, and so a future work can
    wire per-position P&L decomposition without changing the spec shape.
    """

    name: str
    notional: float = 1.0
    weight: float | None = None


class TailRiskSpec(BaseModel):
    """A tail-risk assessment request: confidence levels, horizon, an
    optional position/portfolio definition, a distributional assumption
    tag, and every knob the multi-method comparison needs (quantum on/off,
    QAE grid resolution, Monte Carlo path count, bootstrap draws, seed).
    """

    confidence_levels: list[float] = Field(default_factory=lambda: [0.95])
    horizon_days: int = 1
    positions: list[PositionSpec] = Field(default_factory=list)
    distribution: DistributionAssumption = DistributionAssumption.EMPIRICAL
    seed: int = 42

    run_quantum: bool = True
    n_grid_qubits: int = 3
    epsilon: float = 0.03
    shots: int = 1024

    n_mc_paths: int = 20_000
    n_bootstrap: int = 5_000

    @field_validator("confidence_levels")
    @classmethod
    def _check_levels(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("confidence_levels must be non-empty")
        for lvl in v:
            if not (0.5 <= lvl < 1.0):
                raise ValueError(f"confidence level {lvl} must be in [0.5, 1.0)")
        return v

    @field_validator("horizon_days")
    @classmethod
    def _check_horizon(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"horizon_days must be >= 1; got {v}")
        return v

    def primary_alpha(self) -> float:
        return self.confidence_levels[0]

    def total_notional(self) -> float:
        return float(sum(p.notional for p in self.positions))


@dataclass(frozen=True)
class MethodResult:
    """One estimator's VaR/CVaR for a P&L sample at a given confidence
    level, plus wall-clock cost and any method-specific diagnostics
    (skew/kurtosis for Cornish-Fisher, bootstrap draw stats for the
    meta-agent estimator, the located threshold for QAE, ...).

    `var`/`cvar` are positive loss magnitudes, consistent with
    `core.metrics.value_at_risk`'s convention -- every method in this
    package normalizes into that sign convention even when its native
    implementation uses the opposite one (e.g. `classical.meta_agent`'s
    MonteCarloCVaR reports negative-number losses internally).
    """

    method: str
    var: float
    cvar: float
    wall_time_s: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TailRiskComparisonReport:
    """The normalized output of a multi-method VaR/CVaR comparison: one
    `MethodResult` per method, keyed by method name, plus cross-method
    diagnostics (spreads, skew/kurtosis, notes explaining where and why
    methods disagree) that make the comparison the deliverable, not any one
    method's number.
    """

    alpha: float
    n_samples: int
    prob_of_loss: float
    results: dict[str, MethodResult]
    diagnostics: dict[str, float]
    notes: list[str]

    def var_spread(self) -> float:
        values = [m.var for m in self.results.values()]
        return float(max(values) - min(values))

    def cvar_spread(self) -> float:
        values = [m.cvar for m in self.results.values()]
        return float(max(values) - min(values))

    def method_names(self) -> list[str]:
        return sorted(self.results)
