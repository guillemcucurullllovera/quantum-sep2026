"""Converter pipeline: FinancePortfolioModel -> QUBO.

Adapted idea from qiskit-optimization: separate "what are the constraints"
(the model) from "how are they penalty-encoded" (a pluggable strategy) --
qiskit-optimization calls its analogous step `QuadraticProgramToQubo`.

Deliberate design choice, stated plainly: this converter does NOT re-derive
the penalty-term math (the (sum x_i - k)^2 expansions, the turnover
linearization, etc.) -- that math already exists, correct and tested, in
`algorithms.qaoa_qubo.qubo_builder`. Re-deriving it a third time here would
be exactly the kind of pointless duplication this platform's build
guidelines warn against. Instead, a PenaltyStrategy computes penalty
*coefficients* from the model, this converter packages the model's
constraints into `qubo_builder`'s existing `PortfolioConstraints` /
`LongShortConstraints` dataclasses using those coefficients, and delegates
the actual QUBO construction to `PortfolioQUBOBuilder` / `LongShortQUBOBuilder`.
The value this layer adds is (1) the named/inspectable model object itself,
(2) the pluggable, data-driven penalty-selection strategy below, which
`qubo_builder` alone has no notion of (its penalties are just constructor
defaults), and (3) `test_model_to_qubo_matches_direct_build` in this
module's test file, which proves the packaging step is faithful.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from algorithms.qaoa_qubo.qubo_builder import (
    LongShortConstraints,
    LongShortQUBOBuilder,
    PortfolioConstraints,
    PortfolioQUBOBuilder,
    SectorLimit,
)
from algorithms_extended.qiskit_optimization_finance.model import FinancePortfolioModel


class PenaltyStrategy(ABC):
    """Computes penalty coefficients for a model's constraints."""

    @abstractmethod
    def penalties(self, model: FinancePortfolioModel) -> dict[str, float]:
        """Return a dict with keys among: cardinality, turnover,
        transaction_cost, sector, liquidity, leverage, net_exposure."""


class FixedPenaltyStrategy(PenaltyStrategy):
    """The constant defaults `algorithms.qaoa_qubo.qubo_builder` itself
    uses (cardinality=2.0, turnover=1.0, transaction_cost=1.0, sector=1.5,
    liquidity=0.5, leverage=2.0, net_exposure=1.0) -- reasonable when the
    objective's own coefficient magnitudes are O(1). Breaks down when the
    objective is scaled much larger or smaller
    than these constants (see AdaptivePenaltyStrategy)."""

    def __init__(
        self,
        cardinality: float = 2.0,
        turnover: float = 1.0,
        transaction_cost: float = 1.0,
        sector: float = 1.5,
        liquidity: float = 0.5,
        leverage: float = 2.0,
        net_exposure: float = 1.0,
    ) -> None:
        self._p = dict(
            cardinality=cardinality, turnover=turnover, transaction_cost=transaction_cost,
            sector=sector, liquidity=liquidity, leverage=leverage, net_exposure=net_exposure,
        )

    def penalties(self, model: FinancePortfolioModel) -> dict[str, float]:
        return dict(self._p)


class AdaptivePenaltyStrategy(PenaltyStrategy):
    """Scales every penalty to the model's own objective coefficient
    magnitude, so constraints stay enforced regardless of the objective's
    scale -- a badly-calibrated (too-small) penalty relative to the
    objective is a classic, well-documented QAOA/QUBO failure mode (the
    optimizer happily violates a "soft" constraint because doing so
    improves the objective by more than the penalty costs). k is a safety
    multiplier on top of the objective's own largest coefficient magnitude.
    """

    def __init__(self, k: float = 3.0) -> None:
        self.k = k

    def penalties(self, model: FinancePortfolioModel) -> dict[str, float]:
        scale = float(max(np.abs(model.mu).max() if model.mu.size else 0.0,
                           np.abs(model.sigma).max() if model.sigma.size else 0.0,
                           1e-9))
        base = self.k * scale
        return dict(
            cardinality=base, turnover=base, transaction_cost=base,
            sector=base * 1.1, liquidity=base * 0.5, leverage=base, net_exposure=base * 0.5,
        )


class ModelToQUBOConverter:
    def convert(
        self, model: FinancePortfolioModel, strategy: PenaltyStrategy
    ) -> tuple[np.ndarray, "PortfolioConstraints | LongShortConstraints"]:
        pens = strategy.penalties(model)

        if model.long_short:
            builder = LongShortQUBOBuilder(model.mu, model.sigma, model.risk_aversion)
            constraints = LongShortConstraints(
                leverage_limit=model.leverage_limit if model.leverage_limit is not None else 1.5,
                leverage_penalty=pens["leverage"],
                net_exposure_target=model.net_exposure_target,
                net_exposure_penalty=pens["net_exposure"],
            )
            Q = builder.build(constraints)
            return Q, constraints

        builder = PortfolioQUBOBuilder(model.mu, model.sigma, model.risk_aversion)
        sector_limits = [
            SectorLimit(
                name=sr.name,
                asset_indices=model.asset_indices(sr.asset_names),
                min_count=sr.min_count,
                max_count=sr.max_count,
            )
            for sr in model.sector_ranges
        ]
        constraints = PortfolioConstraints(
            cardinality_k=model.cardinality_k,
            cardinality_penalty=pens["cardinality"],
            previous_holdings=model.previous_holdings,
            turnover_limit=model.turnover_limit,
            turnover_penalty=pens["turnover"],
            transaction_cost_bps=model.transaction_cost_bps,
            transaction_cost_penalty=pens["transaction_cost"],
            sector_limits=sector_limits,
            sector_penalty=pens["sector"],
            liquidity_scores=model.liquidity_scores,
            liquidity_penalty=pens["liquidity"],
            liquidity_floor=model.liquidity_floor,
        )
        Q = builder.build(constraints)
        return Q, constraints
