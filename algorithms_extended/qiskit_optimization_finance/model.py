"""Named-variable, explicit-constraint finance model -- the object-oriented
modeling layer qiskit-optimization's `QuadraticProgram` provides and
direct-QUBO-matrix approach deliberately skips.

This is NOT a `QuadraticProgram` reimplementation: it has no generic
variable-type machinery, no arbitrary linear/quadratic expression parser,
and no standalone Ising translator (this platform already has one, in
`algorithms.qaoa_qubo.qubo_builder` and
`algorithms_extended.qubo_constraints_qiskit_optimization`). What's adapted
from qiskit-optimization is narrower and specific: the idea of representing
constraints as named, inspectable, first-class objects attached to a model,
separate from the penalty coefficients used to encode them -- so a caller
can ask "what constraints does this problem have" without already knowing
how they were penalty-encoded. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.errors import ValidationError


@dataclass
class SectorRange:
    """A named group of assets with an explicit [min_count, max_count] range
    -- represented here as two independent constraint intents (a lower bound
    and an upper bound), not baked into a single penalty coefficient yet.
    Whether both bounds can actually be enforced simultaneously is a
    property of the *converter* (see converters.py's honest limitation
    note), not of this model object.
    """
    name: str
    asset_names: list[str]
    min_count: int = 0
    max_count: int | None = None


@dataclass
class _ConstraintLog:
    kind: str
    detail: dict


class FinancePortfolioModel:
    """A finance portfolio problem as named variables + explicit constraint
    objects. Call `.to_qubo(strategy)` (via ModelToQUBOConverter) to obtain
    the QUBO matrix this platform's solvers already consume.
    """

    def __init__(
        self,
        asset_names: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 0.5,
        long_short: bool = False,
    ) -> None:
        self.asset_names = list(asset_names)
        self.mu = np.asarray(expected_returns, dtype=float)
        self.sigma = np.asarray(cov_matrix, dtype=float)
        self.risk_aversion = risk_aversion
        self.long_short = long_short
        n = len(self.asset_names)
        if len(self.mu) != n or self.sigma.shape != (n, n):
            raise ValidationError(
                f"FinancePortfolioModel: asset_names has {n} entries but "
                f"expected_returns has {len(self.mu)} and cov_matrix is "
                f"{self.sigma.shape} -- all three must agree on n_assets."
            )
        self._name_to_idx = {name: i for i, name in enumerate(self.asset_names)}
        self._log: list[_ConstraintLog] = []

        # populated by add_* methods, consumed by converters.ModelToQUBOConverter
        self.cardinality_k: int | None = None
        self.previous_holdings: np.ndarray | None = None
        self.turnover_limit: int | None = None
        self.transaction_cost_bps: float = 0.0
        self.sector_ranges: list[SectorRange] = []
        self.liquidity_scores: np.ndarray | None = None
        self.liquidity_floor: float = 0.0
        self.leverage_limit: float | None = None
        self.net_exposure_target: float | None = None

    def asset_indices(self, names: list[str]) -> list[int]:
        missing = [n for n in names if n not in self._name_to_idx]
        if missing:
            raise ValidationError(f"FinancePortfolioModel: unknown asset name(s) {missing}")
        return [self._name_to_idx[n] for n in names]

    def add_cardinality(self, k: int) -> "FinancePortfolioModel":
        if not (0 < k <= len(self.asset_names)):
            raise ValidationError(f"cardinality k={k} out of range [1, {len(self.asset_names)}]")
        self.cardinality_k = k
        self._log.append(_ConstraintLog("cardinality", {"k": k}))
        return self

    def add_turnover(self, previous_holdings: np.ndarray, limit: int) -> "FinancePortfolioModel":
        p = np.asarray(previous_holdings, dtype=float)
        if len(p) != len(self.asset_names):
            raise ValidationError("previous_holdings length must match n_assets")
        self.previous_holdings = p
        self.turnover_limit = limit
        self._log.append(_ConstraintLog("turnover", {"limit": limit}))
        return self

    def add_transaction_cost(self, bps: float) -> "FinancePortfolioModel":
        self.transaction_cost_bps = bps
        self._log.append(_ConstraintLog("transaction_cost", {"bps": bps}))
        return self

    def add_sector_range(self, name: str, asset_names: list[str], min_count: int = 0, max_count: int | None = None) -> "FinancePortfolioModel":
        self.asset_indices(asset_names)  # validate names exist
        self.sector_ranges.append(SectorRange(name, list(asset_names), min_count, max_count))
        self._log.append(_ConstraintLog("sector_range", {"name": name, "min": min_count, "max": max_count}))
        return self

    def add_liquidity_floor(self, scores: np.ndarray, floor: float) -> "FinancePortfolioModel":
        s = np.asarray(scores, dtype=float)
        if len(s) != len(self.asset_names):
            raise ValidationError("liquidity scores length must match n_assets")
        self.liquidity_scores = s
        self.liquidity_floor = floor
        self._log.append(_ConstraintLog("liquidity_floor", {"floor": floor}))
        return self

    def add_leverage_limit(self, limit: float) -> "FinancePortfolioModel":
        if not self.long_short:
            raise ValidationError("add_leverage_limit requires long_short=True at construction")
        self.leverage_limit = limit
        self._log.append(_ConstraintLog("leverage", {"limit": limit}))
        return self

    def add_net_exposure_target(self, target: float) -> "FinancePortfolioModel":
        if not self.long_short:
            raise ValidationError("add_net_exposure_target requires long_short=True at construction")
        self.net_exposure_target = target
        self._log.append(_ConstraintLog("net_exposure", {"target": target}))
        return self

    def constraint_summary(self) -> list[dict]:
        """Inspect what constraints this model carries without needing to
        know how they'll be penalty-encoded -- the point of this layer."""
        return [{"kind": c.kind, **c.detail} for c in self._log]

    def n_assets(self) -> int:
        return len(self.asset_names)
