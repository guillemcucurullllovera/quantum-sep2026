"""Synthetic OptionSelectionInstance benchmark-ladder generator.

Deterministic (seeded), labeled `provenance="synthetic"` on every instance
-- never a fabricated historical AMC candidate record (the brief's
explicit prohibition). A read-only replay-derived generator reading real
persisted paper-candidate evidence would set `provenance="replay_readonly"`
and is deliberately NOT implemented in this phase (no such read-only
loader exists yet; documented as future work in
`QUANTUM1B_CLASSICAL_BENCHMARK.md`, not faked here).

Underlyings match the AMC's actual root set: SPY, QQQ, TLT, IEF, GLD, USO.
"""
from __future__ import annotations

import numpy as np

from experiments.quantum1b.instance import CandidateSpec, OptionSelectionInstance

UNDERLYINGS: tuple[str, ...] = ("SPY", "QQQ", "TLT", "IEF", "GLD", "USO")

FAMILIES: tuple[str, ...] = (
    "independent",
    "correlated",
    "tight_budget",
    "loose_budget",
    "degenerate",
    "concentrated",
    "conflicting_constraints",
    "sparse",
    "dense",
)


def _round_unit(value: float, unit: float = 0.25) -> float:
    return round(round(value / unit) * unit, 6)


def generate_instance(
    family: str,
    n: int,
    seed: int,
    *,
    instance_id: str | None = None,
) -> OptionSelectionInstance:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}, expected one of {FAMILIES}")
    rng = np.random.default_rng(seed)

    n_underlyings = 2 if family == "concentrated" else len(UNDERLYINGS)
    underlyings_used = UNDERLYINGS[:n_underlyings]

    utilities = rng.uniform(0.5, 3.0, size=n)
    premiums = rng.uniform(0.5, 2.5, size=n)
    liquidity = rng.uniform(0.0, 1.0, size=n)
    directions = rng.choice(["long", "short", "neutral"], size=n)
    underlying_choices = rng.choice(underlyings_used, size=n)

    candidates = []
    for i in range(n):
        candidates.append(
            CandidateSpec(
                candidate_id=f"cand_{i:03d}",
                underlying=str(underlying_choices[i]),
                expected_utility=_round_unit(float(utilities[i])),
                premium_cost=_round_unit(float(premiums[i])),
                max_loss=_round_unit(float(premiums[i]) * rng.uniform(1.0, 3.0)),
                liquidity_score=_round_unit(float(liquidity[i]), 0.01),
                direction=str(directions[i]),
            )
        )

    total_premium = sum(c.premium_cost for c in candidates)

    if family == "tight_budget":
        risk_budget = _round_unit(total_premium * 0.25)
    elif family == "loose_budget":
        risk_budget = _round_unit(total_premium * 0.85)
    else:
        risk_budget = _round_unit(total_premium * 0.45)

    max_positions = max(2, n // 3)

    risk_aversion = 0.0
    pairwise_risk: tuple[tuple[str, str, float], ...] = ()
    if family in ("correlated", "dense", "concentrated"):
        risk_aversion = 0.15
        pairs = []
        density = 0.8 if family in ("dense", "concentrated") else 0.4
        for i in range(n):
            for j in range(i + 1, n):
                if candidates[i].underlying == candidates[j].underlying and rng.random() < density:
                    pairs.append((candidates[i].candidate_id, candidates[j].candidate_id, 1.0))
        pairwise_risk = tuple(pairs)
    elif family == "sparse":
        risk_aversion = 0.05
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.05:
                    pairs.append((candidates[i].candidate_id, candidates[j].candidate_id, 0.5))
        pairwise_risk = tuple(pairs)
    elif family == "independent":
        risk_aversion = 0.0
        pairwise_risk = ()
    elif family == "degenerate":
        # force many near-identical utility/premium ratios -> many
        # near-optimal solutions
        ratio = rng.uniform(1.2, 1.4)
        for idx in range(n):
            candidates[idx] = CandidateSpec(
                candidate_id=candidates[idx].candidate_id,
                underlying=candidates[idx].underlying,
                expected_utility=_round_unit(candidates[idx].premium_cost * ratio),
                premium_cost=candidates[idx].premium_cost,
                max_loss=candidates[idx].max_loss,
                liquidity_score=candidates[idx].liquidity_score,
                direction=candidates[idx].direction,
            )
        risk_aversion = 0.02
    elif family == "conflicting_constraints":
        risk_aversion = 0.1
        max_positions = max(2, n // 4)  # tight cardinality AND tight budget together
        risk_budget = _round_unit(total_premium * 0.2)

    incompatible_pairs: list[tuple[int, int]] = []
    if family == "conflicting_constraints":
        # mark several of the best-utility candidates mutually exclusive,
        # deliberately creating tension with the cardinality/budget caps
        ranked = sorted(range(n), key=lambda i: -candidates[i].expected_utility)
        top = ranked[: max(2, n // 3)]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                if rng.random() < 0.5:
                    incompatible_pairs.append((top[a], top[b]))

    if incompatible_pairs:
        incompat_map: dict[int, list[str]] = {}
        for a, b in incompatible_pairs:
            incompat_map.setdefault(a, []).append(candidates[b].candidate_id)
            incompat_map.setdefault(b, []).append(candidates[a].candidate_id)
        candidates = [
            CandidateSpec(
                candidate_id=c.candidate_id,
                underlying=c.underlying,
                expected_utility=c.expected_utility,
                premium_cost=c.premium_cost,
                max_loss=c.max_loss,
                liquidity_score=c.liquidity_score,
                direction=c.direction,
                incompatible_with=tuple(incompat_map.get(i, ())),
            )
            for i, c in enumerate(candidates)
        ]

    liquidity_floor = 0.1 if family != "degenerate" else None

    return OptionSelectionInstance(
        instance_id=instance_id or f"{family}_n{n}_s{seed}",
        candidates=tuple(candidates),
        risk_budget=max(risk_budget, min(c.premium_cost for c in candidates)),
        family=family,
        seed=seed,
        provenance="synthetic",
        max_positions=max_positions,
        liquidity_floor=liquidity_floor,
        risk_aversion=risk_aversion,
        pairwise_risk=pairwise_risk,
    )


def generate_ladder(
    families: tuple[str, ...],
    sizes: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[OptionSelectionInstance]:
    out = []
    for family in families:
        for n in sizes:
            for seed in seeds:
                out.append(generate_instance(family, n, seed))
    return out
