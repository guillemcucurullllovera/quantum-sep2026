"""AMC-shaped discrete option-structure selection instance.

This is the SINGLE shared problem representation for QUANTUM-1B: Track A
(classical comparator) and Track B (CVaR-QAOA) both consume
`OptionSelectionInstance` and both build their QUBO/constraint model
through `experiments.quantum1b.qubo_encoding`, so neither track can drift
into an easier objective or looser constraints than the other -- see
docs/quantum/QUANTUM1B_PLAN.md section 1.

This is a research comparator schema. It is deliberately NOT
`applications.portfolio.domain.PortfolioSpec` (that schema is continuous
weight-allocation/Markowitz-shaped; the AMC candidate-selection problem is
discrete knapsack/selection-shaped -- see QUANTUM1B_PLAN.md section 0) and
it is never imported by or written into the production portfolio
application layer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

Direction = Literal["long", "short", "neutral"]

_DIRECTION_SIGN: dict[str, int] = {"long": 1, "short": -1, "neutral": 0}


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate defined-risk option structure."""

    candidate_id: str
    underlying: str
    expected_utility: float
    premium_cost: float
    max_loss: float
    delta: float | None = None
    vega: float | None = None
    theta: float | None = None
    liquidity_score: float | None = None
    direction: Direction = "neutral"
    incompatible_with: tuple[str, ...] = ()

    def direction_sign(self) -> int:
        return _DIRECTION_SIGN[self.direction]


@dataclass(frozen=True)
class OptionSelectionInstance:
    """A frozen candidate-selection problem instance: select a subset of
    `candidates` (0/1 per candidate) maximizing total expected utility
    minus a quadratic co-selection risk penalty, subject to a bounded
    derivatives risk budget and the constraint surface below.

    The all-zero selection (NO TRADE) always satisfies every constraint
    here by construction -- it is the valid reference/floor outcome the
    brief requires.

    `pairwise_risk` encodes the quadratic risk-interaction term: entries
    `(candidate_id_i, candidate_id_j, weight)` contribute
    `risk_aversion * weight * premium_i * premium_j * x_i * x_j` to the
    (minimized) risk term. This is what makes the objective genuinely
    quadratic (a QUBO, not a trivial linear knapsack) -- it represents
    compounding concentration/correlation risk when positions are
    co-selected, which is why both a quantum QUBO method AND a real
    MIQP/MILP classical method are meaningful comparators for this
    problem (see QUANTUM1B_PLAN.md section 1-2).
    """

    instance_id: str
    candidates: tuple[CandidateSpec, ...]
    risk_budget: float
    family: str
    seed: int
    provenance: Literal["synthetic", "replay_readonly"] = "synthetic"
    max_positions: int | None = None
    max_per_underlying: dict[str, int] | None = None
    max_concentration: float | None = None
    max_directional_exposure: float | None = None
    liquidity_floor: float | None = None
    risk_aversion: float = 0.0
    pairwise_risk: tuple[tuple[str, str, float], ...] = ()

    def n(self) -> int:
        return len(self.candidates)

    def candidate_index(self) -> dict[str, int]:
        return {c.candidate_id: i for i, c in enumerate(self.candidates)}

    def underlyings(self) -> tuple[str, ...]:
        seen: list[str] = []
        for c in self.candidates:
            if c.underlying not in seen:
                seen.append(c.underlying)
        return tuple(seen)

    def fingerprint(self) -> str:
        """Deterministic sha256 fingerprint of every field that defines
        the problem (not of any solution) -- used to bind classical and
        quantum results to the exact same instance in every result
        manifest and chart contract."""
        payload = {
            "instance_id": self.instance_id,
            "family": self.family,
            "seed": self.seed,
            "provenance": self.provenance,
            "risk_budget": self.risk_budget,
            "max_positions": self.max_positions,
            "max_per_underlying": self.max_per_underlying,
            "max_concentration": self.max_concentration,
            "max_directional_exposure": self.max_directional_exposure,
            "liquidity_floor": self.liquidity_floor,
            "risk_aversion": self.risk_aversion,
            "pairwise_risk": sorted(self.pairwise_risk),
            "candidates": [asdict(c) for c in self.candidates],
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class ConstraintViolation:
    kind: str
    detail: str
    amount: float


@dataclass(frozen=True)
class SelectionEvaluation:
    """Ground-truth evaluation of one 0/1 selection against the TRUE
    constrained objective (never the penalized QUBO) -- both Track A's
    exact solvers and Track B's post-hoc feasibility check call this, so
    "feasible"/"objective" mean the same thing regardless of which solver
    produced the bitstring."""

    selection: tuple[int, ...]
    feasible: bool
    objective: float
    linear_utility: float
    quadratic_risk: float
    total_premium: float
    risk_budget_utilization: float
    n_positions: int
    violations: tuple[ConstraintViolation, ...]

    def selected_ids(self, instance: OptionSelectionInstance) -> tuple[str, ...]:
        return tuple(
            c.candidate_id for c, bit in zip(instance.candidates, self.selection) if bit
        )


def evaluate_selection(
    instance: OptionSelectionInstance, x: "list[int] | tuple[int, ...]"
) -> SelectionEvaluation:
    n = instance.n()
    x = [int(bit) for bit in x]
    if len(x) != n:
        raise ValueError(f"selection length {len(x)} != instance n={n}")

    candidates = instance.candidates
    violations: list[ConstraintViolation] = []

    total_premium = sum(c.premium_cost * bit for c, bit in zip(candidates, x))
    n_positions = sum(x)

    if total_premium > instance.risk_budget + 1e-9:
        violations.append(
            ConstraintViolation(
                "risk_budget",
                f"total_premium={total_premium} > risk_budget={instance.risk_budget}",
                total_premium - instance.risk_budget,
            )
        )

    if instance.max_positions is not None and n_positions > instance.max_positions:
        violations.append(
            ConstraintViolation(
                "max_positions",
                f"n_positions={n_positions} > max_positions={instance.max_positions}",
                n_positions - instance.max_positions,
            )
        )

    if instance.max_per_underlying is not None:
        counts: dict[str, int] = {}
        for c, bit in zip(candidates, x):
            if bit:
                counts[c.underlying] = counts.get(c.underlying, 0) + 1
        for underlying, cap in instance.max_per_underlying.items():
            got = counts.get(underlying, 0)
            if got > cap:
                violations.append(
                    ConstraintViolation(
                        "max_per_underlying",
                        f"{underlying}: {got} > cap {cap}",
                        got - cap,
                    )
                )

    if instance.max_concentration is not None and instance.risk_budget > 0:
        per_underlying_premium: dict[str, float] = {}
        for c, bit in zip(candidates, x):
            if bit:
                per_underlying_premium[c.underlying] = (
                    per_underlying_premium.get(c.underlying, 0.0) + c.premium_cost
                )
        for underlying, spend in per_underlying_premium.items():
            frac = spend / instance.risk_budget
            if frac > instance.max_concentration + 1e-9:
                violations.append(
                    ConstraintViolation(
                        "max_concentration",
                        f"{underlying}: {frac:.4f} > {instance.max_concentration}",
                        frac - instance.max_concentration,
                    )
                )

    if instance.max_directional_exposure is not None:
        directional = sum(
            c.direction_sign() * c.premium_cost * bit for c, bit in zip(candidates, x)
        )
        if abs(directional) > instance.max_directional_exposure + 1e-9:
            violations.append(
                ConstraintViolation(
                    "max_directional_exposure",
                    f"|{directional}| > {instance.max_directional_exposure}",
                    abs(directional) - instance.max_directional_exposure,
                )
            )

    if instance.liquidity_floor is not None:
        for c, bit in zip(candidates, x):
            if bit and (c.liquidity_score is not None) and c.liquidity_score < instance.liquidity_floor:
                violations.append(
                    ConstraintViolation(
                        "liquidity_floor",
                        f"{c.candidate_id}: liquidity {c.liquidity_score} < floor {instance.liquidity_floor}",
                        instance.liquidity_floor - c.liquidity_score,
                    )
                )

    idx = instance.candidate_index()
    for c in candidates:
        if x[idx[c.candidate_id]] == 0:
            continue
        for other_id in c.incompatible_with:
            j = idx.get(other_id)
            if j is not None and x[j] == 1:
                violations.append(
                    ConstraintViolation(
                        "incompatible_pair",
                        f"{c.candidate_id} and {other_id} both selected",
                        1.0,
                    )
                )

    linear_utility = sum(c.expected_utility * bit for c, bit in zip(candidates, x))
    quadratic_risk = 0.0
    if instance.pairwise_risk:
        premium_by_id = {c.candidate_id: c.premium_cost for c in candidates}
        for i_id, j_id, w in instance.pairwise_risk:
            bi = x[idx[i_id]] if i_id in idx else 0
            bj = x[idx[j_id]] if j_id in idx else 0
            if bi and bj:
                quadratic_risk += w * premium_by_id[i_id] * premium_by_id[j_id]
    quadratic_risk *= instance.risk_aversion

    objective = linear_utility - quadratic_risk

    return SelectionEvaluation(
        selection=tuple(x),
        feasible=(len(violations) == 0),
        objective=objective,
        linear_utility=linear_utility,
        quadratic_risk=quadratic_risk,
        total_premium=total_premium,
        risk_budget_utilization=(total_premium / instance.risk_budget if instance.risk_budget else 0.0),
        n_positions=n_positions,
        violations=tuple(violations),
    )


def no_trade_evaluation(instance: OptionSelectionInstance) -> SelectionEvaluation:
    return evaluate_selection(instance, [0] * instance.n())
